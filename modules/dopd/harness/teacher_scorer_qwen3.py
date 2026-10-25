#!/usr/bin/env python3
"""TeacherScorer réel de d-OPD sur Qwen3-0.6B — inférence seule (J17).

Implémentation de production du contrat `academiia.dopd.TeacherScorer`
(J13) pour un professeur AR gelé Hugging Face. Le composant vit **sous
`modules/dopd/harness/`** : ADR-001 interdit `torch` / `transformers` dans
le paquet `academiia.dopd`, qui reste pur (stdlib + NumPy + `academiia.core`).

Ce que le module calcule (DOPD_SPEC §3.2-§3.6, App. B du papier) :

* `r_i(v) = p_T(v | q, z, Â_i)` — prior causal sur **tout** le vocabulaire ;
* `s_i(v) = log p_T(Ĉ_i | q, z, Â_i, v)` — score de compatibilité du futur,
  une branche par candidat ;
* `Δs_i(v) = s_i(v) − s_i(x̂_i)` — delta relatif, produit par la cible NumPy
  existante (`academiia.dopd.compute_future_aware_target`).

Deux chemins de calcul sont fournis :

* **cache** : le préfixe causal est prérempli **une fois** ; le même forward
  fournit le prior causal et le KV cache, qui est ensuite dupliqué sur les
  branches (`batch_repeat_interleave`) — chaque branche ne calcule que
  `[v, Ĉ_i]`. C'est la géométrie de l'App. B, mesurée en J14 ;
* **rejeu** : le contexte complet `(préfixe, v, Ĉ_i)` est rejoué par
  branche, sans cache — c'est le **chemin de référence** qui sert de contrôle
  d'exactitude.

Le cache de préfixe est **immuable** : le préremplissage produit une copie
figée des tenseurs KV, et chaque lot de branches reconstruit son cache à
partir de cette copie. Un `DynamicCache` de Hugging Face est mutable (le
forward des branches y ajoute les tokens du candidat et du suffixe) : le
réutiliser tel quel fausserait les jobs suivants.

Ce que le module ne fait jamais : aucun `optimizer`, aucun `backward`, aucun
`requires_grad_(True)`, aucune écriture de checkpoint. Le professeur est
gelé (`eval()`, `requires_grad=False`) et lu seulement.

Commande GPU prévue (cache Hugging Face redirigé dans le dépôt, règle 9) :

    HF_HOME="$PWD/_artefact/hf-cache" python \
        modules/dopd/harness/run_teacher_scorer_smoke.py \
        --model Qwen/Qwen3-0.6B --device cuda --dtype bfloat16 \
        --output modules/dopd/eval/teacher_scorer_qwen3_0.6b/results.json
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HF_HOME_AT_IMPORT = os.environ.get("HF_HOME")
_DEFAULT_HF_CACHE = _REPO_ROOT / "_artefact" / "hf-cache"
if not _HF_HOME_AT_IMPORT:
    os.environ["HF_HOME"] = str(_DEFAULT_HF_CACHE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

from academiia.dopd.interfaces import ScoringRequest, ScoringResult, TeacherScorer  # noqa: E402

__all__ = [
    "DEFAULT_TEACHER_MODEL",
    "MAX_PREFIX_CACHE_BYTES",
    "Qwen3TeacherScorer",
    "freeze_teacher",
    "load_teacher",
    "resolve_model_source",
    "teacher_topk",
]

#: Checkpoint professeur gelé de J17 (DOPD_SPEC §6.3, §6.10.3).
DEFAULT_TEACHER_MODEL = "Qwen/Qwen3-0.6B"

#: Plafond du **cache de préfixe étendu** (DOPD_SPEC §3.6) : au-delà, le lot de
#: branches est réduit. 16 GiB = règle de l'App. B du papier.
MAX_PREFIX_CACHE_BYTES = 16 * 1024**3

#: Correspondance nom de dtype → dtype torch (le checkpoint est BF16).
_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def freeze_teacher(model: object) -> None:
    """Gèle un modèle professeur : `eval()` et `requires_grad=False` partout.

    Aucun paramètre n'est jamais remoté : le professeur est un **lecteur**.
    """
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _hf_cache_dirs() -> list[Path]:
    """Répertoires de cache Hugging Face candidats (règle 9)."""
    candidates: list[Path] = []
    for env in ("HF_HOME", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(env)
        if value:
            candidates.append(Path(value) / "hub")
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def resolve_model_source(model: str) -> str:
    """Retrouve le snapshot local du checkpoint, en privilégiant le dépôt.

    Un chemin de répertoire est renvoyé tel quel. Un identifiant Hugging Face
    est cherché dans les caches **du dépôt d'abord** : c'est la seule manière
    de garantir la règle 9 sans téléchargement.
    """
    direct = Path(model)
    if direct.is_dir():
        return str(direct)
    if "/" not in model:
        return model
    org, name = model.split("/", 1)
    repo_dir = f"models--{org}--{name}"
    for hub in _hf_cache_dirs():
        snapshots = hub / repo_dir / "snapshots"
        if not snapshots.is_dir():
            continue
        candidates = sorted(
            (path for path in snapshots.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for snapshot in candidates:
            if (snapshot / "config.json").is_file():
                return str(snapshot)
    return model


def load_teacher(
    model_id: str,
    *,
    device: str | torch.device = "cpu",
    dtype: str = "bfloat16",
    local_files_only: bool = False,
) -> tuple[object, str]:
    """Charge le professeur gelé et rend `(modèle, source effective)`.

    Le modèle est chargé en une seule précision, jamais mélangé : le dtype
    du checkpoint (BF16 pour Qwen3) est celui du GPU ; le contrôle FP32 de
    J14 recharge explicitement le modèle en FP32.
    """
    if dtype not in _DTYPES:
        raise ValueError(f"dtype doit être l'un de {sorted(_DTYPES)}, reçu {dtype!r}")
    source = resolve_model_source(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=_DTYPES[dtype],
        local_files_only=local_files_only,
    )
    model.to(torch.device(device))
    freeze_teacher(model)
    return model, source


def teacher_topk(prior: np.ndarray, k: int) -> np.ndarray:
    """Indices des `k` tokens les plus probables du prior causal.

    Le tri est **stable** : à probabilités égales, l'ordre des identifiants
    départage. La sortie est triée par identifiant croissant pour rester
    déterministe, comme `candidate_union` (DOPD_SPEC §3.4).
    """
    if k < 1:
        raise ValueError(f"k doit être >= 1, reçu {k}")
    values = np.asarray(prior, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"prior doit être 1-D, reçu {values.ndim}-D")
    top = np.argsort(-values, kind="stable")[:k]
    return np.sort(top.astype(np.int64))


class Qwen3TeacherScorer(TeacherScorer):
    """Professeur AR gelé (Qwen3) implémentant le contrat `TeacherScorer`.

    Le calcul de `s_i(v)` utilise le **cache de préfixe partagé** de l'App. B :
    le préfixe causal n'est prérempli qu'une fois et son KV cache est
    dupliqué sur les branches. Le même préremplissage fournit le prior causal
    : un job complet (prior + scores) coûte **un** forward de préremplissage
    et **un** forward de branches.

    Le préfixe courant est mémorisé (`_prefix_key`) : un appel à
    `causal_prior` suivi d'un appel à `suffix_loglikelihood` sur le même
    préfixe — ce que fait `TeacherScorer.score` — ne préremplit qu'une fois.
    :meth:`reset_prefix_cache` invalide cette mémoire pour repartir d'un
    préremplissage neuf (mesure de latence d'un job isolé).

    Attributes:
        model: modèle Hugging Face gelé (`eval()`, `requires_grad=False`).
        device: périphérique de calcul.
        max_prefix_cache_bytes: plafond du cache étendu (§3.6).
    """

    def __init__(
        self,
        model: object,
        *,
        device: str | torch.device = "cpu",
        max_prefix_cache_bytes: int = MAX_PREFIX_CACHE_BYTES,
    ) -> None:
        if not isinstance(max_prefix_cache_bytes, int) or max_prefix_cache_bytes < 1:
            raise ValueError("max_prefix_cache_bytes doit être un entier >= 1")
        self.model = model
        self.device = torch.device(device)
        self.max_prefix_cache_bytes = int(max_prefix_cache_bytes)
        freeze_teacher(model)
        self._prefix_key: tuple[int, ...] | None = None
        self._prefix_prior: np.ndarray | None = None
        self._prefix_snapshot: list[tuple[torch.Tensor, torch.Tensor]] | None = None

    # ------------------------------------------------------------------
    # Contrat `TeacherScorer`
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Taille du vocabulaire du professeur (151 936 pour Qwen3)."""
        return int(self.model.config.vocab_size)

    def causal_prior(self, prefix_ids: Sequence[int]) -> np.ndarray:
        """Prior causal `r_i(v) = p_T(v | q, z, Â_i)` (eq. 11).

        Un seul forward sur le préfixe ; le softmax est calculé en float32
        (§3.6) et rendu en float64 NumPy, distribution complète sur tout le
        vocabulaire. Le préremplissage produit **aussi** le KV cache, réutilisé
        par `suffix_loglikelihood` sur le même préfixe.
        """
        prefix = self._as_ids(prefix_ids, "prefix_ids")
        if not prefix:
            raise ValueError("prefix_ids est vide : le prior causal exige au moins un token")
        if self._prefix_key != prefix or self._prefix_prior is None:
            self._prefill(prefix)
        assert self._prefix_prior is not None
        return self._prefix_prior.copy()

    def suffix_loglikelihood(
        self,
        prefix_ids: Sequence[int],
        candidate_ids: Sequence[int],
        suffix_ids: Sequence[int],
    ) -> np.ndarray:
        """Score de suffixe `s_i(v) = log p_T(Ĉ_i | q, z, Â_i, v)` (eq. 14).

        Une branche par candidat. Le préfixe est mis en cache **une fois** puis
        dupliqué ; chaque branche ne calcule que `[v, Ĉ_i]`.
        """
        prefix = self._as_ids(prefix_ids, "prefix_ids")
        candidates = self._as_ids(candidate_ids, "candidate_ids")
        suffix = self._as_ids(suffix_ids, "suffix_ids")
        if not prefix:
            raise ValueError("prefix_ids est vide")
        if not candidates:
            raise ValueError("candidate_ids est vide")
        if self._prefix_key != prefix or self._prefix_snapshot is None:
            self._prefill(prefix)
        return self._score_branches(prefix, candidates, suffix)

    def score(self, request: ScoringRequest) -> ScoringResult:
        """Exécute une requête complète (hérité du contrat J13).

        Identique à `TeacherScorer.score` : le prior est calculé une fois,
        puis toutes les branches (candidats + référence) sont scorées dans un
        seul lot. La surcharge explicite documente que **toute** la logique
        numérique reste dans le contrat : le scorer de production ne l'
        altère pas.
        """
        return super().score(request)

    # ------------------------------------------------------------------
    # Mémoire de préfixe
    # ------------------------------------------------------------------

    def reset_prefix_cache(self) -> None:
        """Invalide le préfixe mémorisé : le prochain job préremplit à neuf."""
        self._prefix_key = None
        self._prefix_prior = None
        self._prefix_snapshot = None

    # ------------------------------------------------------------------
    # Chemin de référence : rejeu intégral, sans cache
    # ------------------------------------------------------------------

    def replay_scores(
        self,
        prefix_ids: Sequence[int],
        candidate_ids: Sequence[int],
        suffix_ids: Sequence[int],
    ) -> np.ndarray:
        """Scores de suffixe par **rejeu intégral** du contexte, sans cache.

        Pour chaque branche, le contexte complet `(préfixe, v, Ĉ_i)` est
        rejoué. C'est le chemin de référence du contrôle d'exactitude : il
        ne dépend d'aucune mécanique de cache.
        """
        _, scores = self.replay_job(prefix_ids, candidate_ids, suffix_ids)
        return scores

    def replay_job(
        self,
        prefix_ids: Sequence[int],
        candidate_ids: Sequence[int],
        suffix_ids: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rejeu intégral : rend `(prior, scores de suffixe)`, sans cache.

        Le prior est extrait des **mêmes logs** que les scores : dans le
        contexte complet, le logit à la position `P - 1` prédit le candidat,
        donc il porte `r_i(v)`. Le prior est ainsi recalculé par un chemin de
        code distinct de `causal_prior` — le contrôle compare deux calculs
        indépendants, pas deux appels de la même fonction.
        """
        prefix = self._as_ids(prefix_ids, "prefix_ids")
        candidates = self._as_ids(candidate_ids, "candidate_ids")
        suffix = self._as_ids(suffix_ids, "suffix_ids")
        if not prefix:
            raise ValueError("prefix_ids est vide")
        if not candidates:
            raise ValueError("candidate_ids est vide")
        branches = len(candidates)
        prefix_tensor = torch.tensor([prefix], dtype=torch.long, device=self.device)
        candidate_tensor = torch.tensor(
            [[int(token)] for token in candidates], dtype=torch.long, device=self.device
        )
        suffix_tensor = torch.tensor([suffix], dtype=torch.long, device=self.device)
        full = torch.cat(
            (
                prefix_tensor.expand(branches, -1),
                candidate_tensor,
                suffix_tensor.expand(branches, -1),
            ),
            dim=1,
        )
        # Il faut S + 2 logits : celui qui prédit le candidat (position P - 1)
        # et ceux qui prédisent le suffixe (positions P .. P + S - 1).
        keep = len(suffix) + 2
        with torch.no_grad():
            out = self.model(input_ids=full, use_cache=False, logits_to_keep=keep)
            logits = out.logits.float()
        # Indice 0 = position P - 1 (prédit le candidat) ; indices 1..S
        # prédisent les tokens du suffixe.
        prior = torch.softmax(logits[:, 0, :], dim=-1)[0].cpu().numpy().astype(np.float64)
        if len(suffix) == 0:
            scores = np.zeros(branches, dtype=np.float64)
        else:
            log_probs = torch.log_softmax(logits[:, 1 : len(suffix) + 1, :], dim=-1)
            targets = suffix_tensor.expand(branches, -1)
            token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            scores = token_log_probs.sum(dim=-1).cpu().numpy().astype(np.float64)
            del log_probs, token_log_probs
        del out, logits, full
        return prior, scores

    # ------------------------------------------------------------------
    # Interne
    # ------------------------------------------------------------------

    @staticmethod
    def _as_ids(values: Sequence[int], name: str) -> tuple[int, ...]:
        if isinstance(values, (str, bytes)):
            raise TypeError(f"{name} doit être une séquence d'entiers")
        out: list[int] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} doit contenir des entiers, reçu {type(value).__name__}")
            out.append(int(value))
        return tuple(out)

    def _prefill(self, prefix: tuple[int, ...]) -> None:
        """Préremplit le préfixe : prior causal + instantané figé du KV cache.

        L'instantané est **immuable** : les tenseurs sont clonés et détachés.
        Le cache du modèle est mutable (le forward des branches y ajoute le
        candidat et le suffixe) ; le réutiliser tel quel fausserait les jobs
        suivants.
        """
        self.reset_prefix_cache()
        input_ids = torch.tensor([prefix], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
            logits = out.logits[:, -1, :].float()
            prior = torch.softmax(logits, dim=-1)
            cache = out.past_key_values
        self._prefix_prior = prior[0].cpu().numpy().astype(np.float64)
        self._prefix_snapshot = [
            (layer.keys.detach().clone(), layer.values.detach().clone())
            for layer in cache.layers
        ]
        self._prefix_key = prefix
        del out, logits, prior, cache

    def _prefix_cache_bytes(self, prefix_length: int, suffix_length: int) -> int:
        """Taille (en octets) du cache étendu à `n` branches.

        Formule de `bench_budget_rtx5090.py` : 2 (K et V) × couches × lot ×
        longueur × têtes KV × dim × taille d'un élément.
        """
        config = self.model.config
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        elements = (
            2
            * config.num_hidden_layers
            * (prefix_length + suffix_length + 1)
            * config.num_key_value_heads
            * int(head_dim)
        )
        element_size = torch.finfo(self._param_dtype()).bits // 8
        return elements * element_size

    def _param_dtype(self) -> torch.dtype:
        for parameter in self.model.parameters():
            return parameter.dtype
        return torch.float32

    def _branch_batch_size(self, prefix_length: int, suffix_length: int) -> int:
        """Nombre maximal de branches tenables sous le plafond de cache (§3.6)."""
        per_branch = self._prefix_cache_bytes(prefix_length, suffix_length)
        if per_branch <= 0:
            return 1
        return max(1, int(self.max_prefix_cache_bytes // per_branch))

    def _score_branches(
        self,
        prefix: tuple[int, ...],
        candidates: tuple[int, ...],
        suffix: tuple[int, ...],
    ) -> np.ndarray:
        """Calcule les log-vraisemblances de suffixe de toutes les branches.

        Le lot est découpé pour ne jamais dépasser le plafond de cache étendu
        (§3.6) : au-delà, le nombre de branches par forward est réduit.
        """
        scores: list[np.ndarray] = []
        limit = self._branch_batch_size(len(prefix), len(suffix))
        for start in range(0, len(candidates), limit):
            chunk = candidates[start : start + limit]
            scores.append(self._score_chunk(prefix, chunk, suffix))
        return np.concatenate(scores) if scores else np.empty(0, dtype=np.float64)

    def _score_chunk(
        self,
        prefix: tuple[int, ...],
        candidates: tuple[int, ...],
        suffix: tuple[int, ...],
    ) -> np.ndarray:
        """Un forward **avec cache de préfixe** sur un lot de branches."""
        branches = len(candidates)
        assert self._prefix_snapshot is not None
        candidate_tensor = torch.tensor(
            [[int(token)] for token in candidates], dtype=torch.long, device=self.device
        )
        suffix_tensor = torch.tensor([suffix], dtype=torch.long, device=self.device)

        # Cache reconstruit depuis l'instantané figé : chaque lot repart du
        # préfixe exact, jamais d'un cache pollué par un job précédent.
        # `DynamicCache` est construit **sans** configuration : toutes les
        # couches de Qwen3 sont à attention pleine, la reconstruction ne
        # dépend donc d'aucun type de couche particulier.
        cache = DynamicCache(ddp_cache_data=self._prefix_snapshot)
        cache.batch_repeat_interleave(branches)
        branch_input = torch.cat(
            (candidate_tensor, suffix_tensor.expand(branches, -1)), dim=1
        )
        with torch.no_grad():
            out = self.model(
                input_ids=branch_input,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=branch_input.shape[1],
            )
        # Le logit à la position du candidat prédit le premier token du
        # suffixe : les S premiers logits du forward prédisent le suffixe.
        logits = out.logits[:, : len(suffix), :].float()
        del out, cache

        if len(suffix) == 0:
            return np.zeros(branches, dtype=np.float64)
        log_probs = torch.log_softmax(logits, dim=-1)
        targets = suffix_tensor.expand(branches, -1)
        token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        result = token_log_probs.sum(dim=-1).cpu().numpy().astype(np.float64)
        del logits, log_probs
        return result
