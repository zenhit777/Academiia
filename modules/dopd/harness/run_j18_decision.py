#!/usr/bin/env python3
"""Expérience de décision à budget borné de d-OPD (J18) — inférence seule.

Mesure l'effet future-aware (ΔKL agrégé + IC, médiane, proportion de prompts
significatifs, test des signes) avec le `TeacherScorer` réel en **FP32**. Le
verdict GO / NO-GO / GO conditionnel porte sur l'ouverture d'un `StudentPolicy`
réduit en J19.

Ce que le module calcule (J18_PRE §0-§3, DOPD_SPEC §3.2-§3.6) :

* pour chaque prompt du sous-ensemble versionné, un bloc complété x̂ de N = 4
  tokens (généré greedily) ;
* pour chaque position masquée i ∈ {0, 1, 2} (toujours à futur visible), le
  posterior exact π_i conditionné sur l'état visible (énumération du support
  restreint, eq. 7-8) ;
* la causale r_i(v) = p_T(v | préfixe) et la future-aware p̃_i(v) ∝ r_i(v)·exp(
  Δs_i(v)) (eq. 15-18), toutes deux sur l'ensemble de candidats V_i (union
  top-k professeur, k = 16) ;
* ΔKL = KL(π_i ‖ r_i) − KL(π_i ‖ p̃_i) (eq. 28), positif quand la correction
  future-aware se rapproche du posterior exact.

Aucun `StudentPolicy`, aucun entraînement, aucun `backward`/`optimizer`/`step`,
aucun poids modifié : le professeur est gelé (`eval()`, `requires_grad=False`)
et lu seulement. La géométrie est stricte (J18_PRE §1) : préfixes mesurés 12 à
21 tokens, suffixes 1 à 3, pas de cellule de référence longue.

Commande GPU prévue (cache Hugging Face redirigé dans le dépôt, règle 9) :

    HF_HOME="$PWD/_artefact/hf-cache" python \
        modules/dopd/harness/run_j18_decision.py \
        --model Qwen/Qwen3-0.6B --device cuda --dtype float32 \
        --prompt-file modules/dopd/eval/j18_decision/prompts.json \
        --blocks 120 --seed 20261026 --bootstrap-seed 20261026 \
        --output modules/dopd/eval/j18_decision/results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HF_HOME_AT_IMPORT = os.environ.get("HF_HOME")
_DEFAULT_HF_CACHE = _REPO_ROOT / "_artefact" / "hf-cache"
if not _HF_HOME_AT_IMPORT:
    os.environ["HF_HOME"] = str(_DEFAULT_HF_CACHE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from academiia.core import sha256_file  # noqa: E402
from academiia.dopd import DOPDConfig  # noqa: E402
from academiia.dopd.diagnostic import (  # noqa: E402
    exact_token_posteriors,
    kl_exact_to_target,
)
from academiia.dopd.interfaces import (  # noqa: E402
    ArtifactRef,
    LineageManifest,
    LineageWriter,
)

__all__ = [
    "BLOCK_SIZE",
    "DEFAULT_OUTPUT",
    "DEFAULT_PROMPT_FILE",
    "DEFAULT_TEACHER_MODEL",
    "DEFAULT_DTYPE",
    "K_CANDIDATES",
    "MASK_POSITIONS",
    "MAX_GPU_BYTES",
    "MAX_GPU_SECONDS",
    "MIN_BLOCKS_PER_CONFIGURATION",
    "MIN_MARGINALS",
    "SEED",
    "VOCAB_WIDTH",
]

# ---------------------------------------------------------------------------
# Protocole figé (J18_PRE §1, §3) — aucune valeur ici n'est un hyperparamètre
# ajusté a posteriori : ce sont les cibles de la décision.
# ---------------------------------------------------------------------------

#: Checkpoint professeur gelé (DOPD_SPEC §6.3).
DEFAULT_TEACHER_MODEL = "Qwen/Qwen3-0.6B"

#: Précision de production (décision R1 de J17_CLOSE) : FP32, pas BF16.
DEFAULT_DTYPE = "float32"

#: Bloc d-OPD (§6.10.3) : N = 4 tokens par bloc.
BLOCK_SIZE = 4

#: Candidats : union top-k professeur / top-k élève (pas d'élève en J18).
K_CANDIDATES = 16

#: Largeur du support restreint = largeur de l'ensemble de candidats V_i. Elle
#: vaut k (= 16) pour que le posterior exact et les cibles vivent sur le même
#: support : ΔKL est ainsi comparable token pour token (eq. 28).
VOCAB_WIDTH = 16

#: Positions masquées par bloc : {0, 1, 2} — chacune a un futur visible
#: (suffixe de 3, 2 puis 1 token) ; la position 3 n'en a pas et est exclue.
#: Le masquage déterministe (et non aléatoire) garantit que **tous** les
#: marginaux ont un futur visible : la médiane n'est pas noyée par des zéros
#: structurels, contrairement au masquage randomisé de J15.
MASK_POSITIONS = (0, 1, 2)

#: Graine figée (§1, "partout") : blocs, support et bootstrap.
SEED = 20261026

#: Itérations de bootstrap (J14/J15 : 10 000).
BOOTSTRAP_ITERATIONS = 10_000

#: Blocs de base échantillonnés par prompt. ≥ 100 pour C5 (≥ 100 blocs de base
#: par configuration) ; 120 laisse une marge au-dessous du seuil.
DEFAULT_BLOCKS_PER_PROMPT = 120

#: Seils C5 (J18_PRE §3).
MIN_BLOCKS_PER_CONFIGURATION = 100
MIN_MARGINALS = 2_000

#: Budget strict (J18_PRE §5) : plafond dur, au-delà duquel le run s'interrompt
#: sans écrire de résultat partiel.
MAX_GPU_SECONDS = 4 * 60 * 60
MAX_GPU_BYTES = 16 * 1024**3

#: Sous-ensemble de prompts versionné (Phase 1).
DEFAULT_PROMPT_FILE = "modules/dopd/eval/j18_decision/prompts.json"
DEFAULT_OUTPUT = "modules/dopd/eval/j18_decision/results.json"

#: Code producteur : son commit et son empreinte sont consigné dans le manifeste
#: ADR-002 ; un résultat produit depuis une version de travail non versionnée
#: est refusé.
PRODUCER_SOURCES = (
    "modules/dopd/harness/run_j18_decision.py",
    "modules/dopd/harness/teacher_scorer_qwen3.py",
    "modules/dopd/dopd/diagnostic.py",
    "modules/dopd/dopd/interfaces.py",
)

#: Cibles de géométrie mesurée (J18_PRE §1).
PREFIX_LENGTH_MIN = 12
PREFIX_LENGTH_MAX = 21
SUFFIX_LENGTH_MIN = 1
SUFFIX_LENGTH_MAX = 3


# ---------------------------------------------------------------------------
# Outils de base
# ---------------------------------------------------------------------------


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _gpu_peak_bytes() -> int:
    """Mémoire GPU allouée cumulée (octets), si CUDA disponible."""
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return 0


def _gpu_peak_gib() -> float:
    """Mémoire GPU allouée cumulée (GiB), 0 si hors CUDA."""
    return _gpu_peak_bytes() / 1024**3

def _check_budget(started: float) -> None:
    """Arrêt automatique si le plafond temps/mémoire est dépassé (J18_PRE §5).

    Le run lève `RuntimeError` : aucun résultat n'est alors écrit.
    """
    if time.time() - started > MAX_GPU_SECONDS:
        raise RuntimeError(
            f"budget GPU temps dépassé ({time.time() - started:.0f}s > {MAX_GPU_SECONDS:.0f}s)"
        )
    peak = _gpu_peak_bytes()
    if peak > MAX_GPU_BYTES:
        raise RuntimeError(
            f"budget mémoire GPU dépassé {peak / 1024**3:.2f} GiB "
            f"> {MAX_GPU_BYTES / 1024**3:.0f} GiB"
        )


# ---------------------------------------------------------------------------
# Support restreint (Annexe D.1) — reconstruit à partir des sorties du professeur
# ---------------------------------------------------------------------------


def _build_restricted_support(
    model: object,
    prompt_ids: torch.Tensor,
    block_size: int,
    vocab_width: int,
    device: torch.device,
    batch_size: int = 32,
) -> object:
    """Construit le support fini position par position (Annexe D.1).

    À chaque position, la distribution teacher est la mixture pondérée par la
    probabilité des préfixes restreints ; les ``vocab_width`` meilleurs tokens
    définissent le support de la position suivante. Adaptation directe du
    harnais Annexe D (J15), réimplémentée ici pour garder le producteur J18
    autonome : cette fonction ne dépend que des sorties du modèle.
    """
    from academiia.dopd.diagnostic import RestrictedSupport

    supports: list[np.ndarray] = []
    depth_prefixes: list[tuple[tuple[int, ...], ...]] = []
    conditional_log_probs: list[np.ndarray] = []

    prefix_candidates: list[tuple[int, ...]] = [()]
    prefix_log_weights = np.asarray([0.0], dtype=np.float64)
    for _position in range(block_size):
        depth_prefixes.append(tuple(prefix_candidates))
        mixture = np.zeros(int(model.config.vocab_size), dtype=np.float64)
        full_conditionals: list[np.ndarray] = []
        for start in range(0, len(prefix_candidates), batch_size):
            batch_prefixes = prefix_candidates[start : start + batch_size]
            rows = []
            for prefix in batch_prefixes:
                prefix_tensor = torch.tensor(prefix, dtype=torch.long, device=device)
                rows.append(torch.cat((prompt_ids, prefix_tensor)))
            input_ids = torch.stack(rows)
            with torch.no_grad():
                logits = model(input_ids=input_ids).logits[:, -1, :].float()
                batch_log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy().astype(
                    np.float64
                )
            del logits
            weights = np.exp(prefix_log_weights[start : start + len(batch_prefixes)])
            mixture += np.sum(np.exp(batch_log_probs) * weights[:, None], axis=0)
            full_conditionals.append(batch_log_probs)
        conditional = np.concatenate(full_conditionals, axis=0)
        selected = np.sort(
            np.argsort(-mixture, kind="stable")[:vocab_width]
        ).astype(np.int64)
        restricted_conditional = conditional[:, selected]
        supports.append(selected)
        conditional_log_probs.append(restricted_conditional)

        next_prefixes: list[tuple[int, ...]] = []
        next_log_weights: list[float] = []
        for prefix, prefix_weight, row in zip(
            prefix_candidates, prefix_log_weights, conditional, strict=True
        ):
            for candidate_index, token_id in enumerate(selected):
                next_prefixes.append(prefix + (int(token_id),))
                next_log_weights.append(float(prefix_weight + row[candidate_index]))
        prefix_candidates = next_prefixes
        prefix_log_weights = np.asarray(next_log_weights, dtype=np.float64)
        prefix_log_weights -= np.logaddexp.reduce(prefix_log_weights)

    return RestrictedSupport(
        supports=tuple(supports),
        prefixes=tuple(depth_prefixes),
        conditional_log_probs=tuple(conditional_log_probs),
    )


# ---------------------------------------------------------------------------
# Décision : table §4 de J18_PRE (C1-C5 + C6-C10)
# ---------------------------------------------------------------------------


def _binominal_two_tailed_p_value(positive: int, negative: int) -> float:
    """Test des signes : binomiale exacte bilatérale (J18_PRE §3, C4).

    Sous l'hypothèse nulle (médiane de ``ΔKL`` nulle), le nombre de signes
    positifs suit une binomiale ``(n, 0,5)``. La valeur en p est calculée par
    `scipy.stats.binomtest` (exact), complément non paramétrique au critère
    primaire fondé sur la moyenne.
    """
    effective = positive + negative
    if effective == 0:
        return float("nan")
    return float(binomtest(positive, effective, p=0.5, alternative="two-sided").pvalue)


def _decide_j18_gate(
    *,
    statistics: dict,
    sign_p_value: float,
    blocks_per_configuration: dict[str, int],
    total_marginals: int,
    numerical_integrity_met: bool,
    manifest_present: bool,
    teacher_frozen_met: bool,
    model_source_in_repo: bool,
    no_regression_met: bool,
) -> dict:
    """Applique la table de décision §4 de J18_PRE, sans GPU.

    | Condition | Décision |
    |---|---|
    | C1-C5 + C6-C10 tenus | GO pour ouvrir un `StudentPolicy` réduit en J19 |
    | C1 non tenu | NO-GO |
    | C2 ou C3 fragiles mais C1 tenu | GO conditionnel (réserve « effet porté par une queue ») |
    | tout C6-C10 échoue | décision explicite écrite ; le silence n'est pas un GO (R1) |

    Les critères C1-C5 sont recalculés à partir des statistiques publiées, de
    façon reproductible et versionnée. Le critère C3 utilise la lecture stricte
    de J18_PRE §3 : proportion de prompts dont l'IC 95 % per-prompt **exclut
    zéro** (sens quel que soit).
    """
    lower, upper = statistics["aggregate_ci95_delta_kl"]
    mean_delta_kl = statistics["mean_delta_kl"]
    median_delta_kl = statistics["median_delta_kl"]

    c1 = bool(
        np.isfinite(mean_delta_kl)
        and mean_delta_kl > 0.0
        and np.isfinite(lower)
        and lower > 0.0
    )
    c2 = bool(np.isfinite(median_delta_kl) and median_delta_kl > 0.0)
    n_prompts = int(statistics["n_prompts"])
    n_excludes_zero = int(statistics["n_prompts_ci_excludes_zero"])
    c3_value = (n_excludes_zero / n_prompts) if n_prompts else float("nan")
    c3 = bool(np.isfinite(c3_value) and n_prompts > 0 and c3_value >= 0.5)
    c4 = bool(np.isfinite(sign_p_value) and sign_p_value < 0.05)
    blocks_ok = bool(blocks_per_configuration) and all(
        count >= MIN_BLOCKS_PER_CONFIGURATION
        for count in blocks_per_configuration.values()
    )
    c5 = bool(blocks_ok and total_marginals >= MIN_MARGINALS)

    c6 = numerical_integrity_met
    c7 = manifest_present
    c8 = teacher_frozen_met
    c9 = model_source_in_repo
    c10 = no_regression_met

    reserves: list[str] = []
    if not c1:
        status = "NO-GO"
    elif all((c1, c2, c3, c4, c5, c6, c7, c8, c9, c10)):
        status = "GO"
    else:
        status = "GO conditionnel"
        if not c2:
            reserves.append("médiane de ΔKL ≤ 0 (effet porté par une queue)")
        if not c3:
            reserves.append(
                f"proportion de prompts à IC exclut zéro {c3_value:.3f} < 0,5 "
                "(porté par moins de la moitié des prompts)"
            )
        if not c4:
            reserves.append(f"test des signes non significatif (p = {sign_p_value:.3g} ≥ 0,05)")
        if not c5:
            reserves.append(
                f"effectif insuffisant : "
                f"{blocks_per_configuration} blocs/config, {total_marginals} marginaux"
            )
        failed_process = [
            name
            for ok, name in (
                (c6, "C6 intégrité numérique"),
                (c7, "C7 traçabilité ADR-002"),
                (c8, "C8 professeur gelé"),
                (c9, "C9 provenance du modèle"),
                (c10, "C10 non-régression"),
            )
            if not ok
        ]
        if failed_process:
            reserves.append("critères de processus échoués : " + ", ".join(failed_process))

    rationale = (
        f"Critère primaire (C1) {'TENU' if c1 else 'NON TENU'} ; "
        f"médiane (C2) {'TENUE' if c2 else 'FRAGILE'} ; "
        f"proportion de prompts à IC exclut zéro (C3) {n_excludes_zero}/{n_prompts} = {c3_value:.3f} "
        f"({'TENUE' if c3 else 'FRAGILE'}) ; "
        f"test des signes (C4) p = {sign_p_value:.3g} "
        f"({'TENU' if c4 else 'NON TENU'}) ; "
        f"effectif (C5) {'TENU' if c5 else 'FRAGILE'}. "
        f"Décision : {status}."
    )
    if reserves:
        rationale += " Réserves : " + "; ".join(reserves) + "."

    return {
        "status": status,
        "criteria": {
            "C1_aggregate_delta_kl_positive_ci95_excludes_zero": {
                "met": c1,
                "value": float(mean_delta_kl),
                "ci95": [float(lower), float(upper)],
            },
            "C2_median_delta_kl_positive": {"met": c2, "value": float(median_delta_kl)},
            "C3_proportion_prompts_ci_excludes_zero_ge_0.5": {
                "met": c3,
                "value": float(c3_value),
                "n_prompts_significant": n_excludes_zero,
                "n_prompts": n_prompts,
            },
            "C4_binomial_two_sided_p_lt_0.05": {"met": c4, "value": float(sign_p_value)},
            "C5_blocks_ge_100_and_marginals_ge_2000": {
                "met": c5,
                "blocks_per_configuration": dict(sorted(blocks_per_configuration.items())),
                "minimum_blocks_per_configuration": MIN_BLOCKS_PER_CONFIGURATION,
                "total_marginals": int(total_marginals),
                "minimum_marginals": MIN_MARGINALS,
            },
        },
        "process_criteria": {
            "C6_numerical_integrity": {"met": c6},
            "C7_lineage_manifest_present": {"met": c7},
            "C8_teacher_frozen": {"met": c8},
            "C9_model_source_in_repo": {"met": c9},
            "C10_no_regression": {"met": c10},
        },
        "rationale": rationale,
        "reserves": reserves,
    }


# ---------------------------------------------------------------------------
# Manifeste ADR-002
# ---------------------------------------------------------------------------


class _ManifestBuilder(LineageWriter):
    """Construit le manifeste ADR-002 avec le commit du code producteur."""

    def __init__(self, commit: str) -> None:
        self._commit = commit

    def write(self, manifest: LineageManifest) -> str:
        raise NotImplementedError(
            "le résultat et son manifeste sont écrits ensemble dans le JSON"
        )

    def build_manifest(self, **kwargs) -> LineageManifest:
        manifest = super().build_manifest(**kwargs)
        producer = dict(manifest.producer)
        producer["commit"] = self._commit
        return LineageManifest(
            artifact=manifest.artifact,
            parent=manifest.parent,
            producer=producer,
            config=manifest.config,
            data=manifest.data,
            verification=manifest.verification,
            schema=manifest.schema,
        )


def _model_revision(model_source: str) -> str | None:
    path = Path(model_source)
    if not path.is_dir():
        return None
    parent = path.parent.name
    if parent == "snapshots":
        return path.name
    return path.name


def _checkpoint_hashes(model_source: str) -> dict[str, str]:
    """Empreintes SHA-256 de fichier des poids du checkpoint (ADR-002)."""
    root = Path(model_source)
    if not root.is_dir():
        raise ValueError(
            f"checkpoint illisible : {model_source!r} n'est pas un répertoire ; "
            "impossible de calculer les empreintes de fichier exigées par ADR-002."
        )
    hashes: dict[str, str] = {}
    for pattern in ("*.safetensors", "*.safetensors.index.json", "*.bin", "*.bin.index.json"):
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                hashes[path.name] = sha256_file(str(path))
    if not hashes:
        raise ValueError(
            f"checkpoint sans fichier de poids hashable : {model_source!r}. "
            "ADR-002 exige les sha256 de fichier des weights lus ; le run est refusé."
        )
    return hashes


def _cache_environment(model_source: str) -> dict:
    """Décrit la redirection de cache réellement en vigueur (règle 9)."""
    try:
        Path(model_source).resolve().relative_to(_REPO_ROOT.resolve())
        in_repo = True
    except (ValueError, OSError):
        in_repo = False
    return {
        "hf_home": os.environ.get("HF_HOME"),
        "hf_home_source": "process_environment" if _HF_HOME_AT_IMPORT else "harness_default",
        "huggingface_hub_cache": os.environ.get("HUGGINGFACE_HUB_CACHE"),
        "model_source": model_source,
        "model_source_in_repo": bool(in_repo),
        "model_revision": _model_revision(model_source),
    }


def _warn_if_cache_is_outside_repo(environment: dict) -> None:
    """Refuse un snapshot lu hors du dépôt (règle 9)."""
    if environment["model_source_in_repo"]:
        return
    raise RuntimeError(
        "regle 9 : le modele serait lu (et potentiellement ecrit) hors du depot : "
        f"{environment['model_source']}; HF_HOME={environment['hf_home']} "
        f"({environment['hf_home_source']}). Poser HF_HOME=$PWD/_artefact/hf-cache "
        "et y placer le snapshot, puis relancer. Le run est refuse avant tout chargement."
    )


def _assert_producer_committed() -> str:
    """Exige que le code producteur soit présent et intact dans ``HEAD``."""
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=_REPO_ROOT,
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()
    for relative in PRODUCER_SOURCES:
        path = _REPO_ROOT / relative
        head_blob = subprocess.run(
            ["git", "rev-parse", f"HEAD:{relative}"],
            cwd=_REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if head_blob.returncode != 0:
            raise RuntimeError(
                f"le code producteur {relative} n'est pas dans le commit {commit} ; "
                "versionner le code avant d'executer J18"
            )
        worktree_blob = subprocess.run(
            ["git", "hash-object", str(path)],
            cwd=_REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
        if worktree_blob != head_blob.stdout.strip():
            raise RuntimeError(
                f"le code producteur {relative} differe du commit {commit} ; "
                "committer les modifications avant d'executer J18"
            )
    return commit


def _producer_source_hashes() -> dict[str, str]:
    """Empreintes sha256 de fichier des sources productrices (ADR-002)."""
    out: dict[str, str] = {}
    for relative in PRODUCER_SOURCES:
        path = _REPO_ROOT / relative
        if path.is_file():
            out[relative] = sha256_file(str(path))
    return out


def _no_regression_delta() -> tuple[bool, str]:
    """C10 : `target.py`, `diagnostic.py` et `modules/a2d/eval/dream_a2d/` inchangés."""
    paths = (
        "modules/dopd/dopd/target.py",
        "modules/dopd/dopd/diagnostic.py",
        "modules/a2d/eval/dream_a2d/",
    )
    diff = subprocess.run(
        ["git", "diff", "--no-color", "659e32a7..HEAD", "--", *paths],
        cwd=_REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ).stdout
    return (len(diff.strip()) == 0, diff[:2000])


def _teacher_is_frozen(scorer: object) -> bool:
    """C8 : `model.training is False` et aucun paramètre émittable."""
    model = scorer.model
    try:
        params = list(model.parameters())
    except AttributeError:
        return False
    return bool(model.training is False and all(not p.requires_grad for p in params))


# ---------------------------------------------------------------------------
# Expérience
# ---------------------------------------------------------------------------


def _generate_block(
    model: object,
    prompt_ids: torch.Tensor,
    block_size: int,
) -> list[int]:
    """Génère un bloc de `N` tokens greedily (décodage argmax), sans aléatoire."""
    generated: list[int] = []
    current = prompt_ids.view(1, -1)
    past = None
    with torch.no_grad():
        for _ in range(block_size):
            out = model(
                input_ids=current,
                past_key_values=past,
                use_cache=True,
                logits_to_keep=1,
            )
            past = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1)
            generated.append(int(next_token.item()))
            current = next_token.view(1, 1)
            del out
    del past
    return generated


def _marginal_for_position(
    scorer: object,
    prompt_ids: tuple[int, ...],
    block: tuple[int, ...],
    position: int,
    support: object,
    sequences: np.ndarray,
    log_joint: np.ndarray,
    block_index: int,
    prompt_id: str,
) -> dict | None:
    """ΔKL d'un marginal : posterior exact vs causale vs future-aware (eq. 28).

    ``sequences`` / ``log_joint`` sont énumérés une fois par bloc (partagés par
    les positions du même bloc). Rend `None` si la position ne produit pas de
    marginal admissible (futur visible nul, ou distribution invalide : C6).
    """
    block_size = len(block)
    visible = {p: int(block[p]) for p in range(block_size) if p != position}
    exact = exact_token_posteriors(
        sequences, log_joint, visible, tuple(support.supports)
    )[position]

    prefix = prompt_ids + block[:position]
    suffix = block[position + 1 :]
    reference = int(block[position])
    candidates = [int(token) for token in support.supports[position]]
    if not suffix or not candidates:
        return None

    # Cible causale r_i(v) = p_T(v | prefix), restreinte à V_i et renormalisée.
    prior_full = scorer.causal_prior(prefix)
    causal = prior_full[np.array(candidates, dtype=np.int64)]
    causal_total = float(np.sum(causal))
    if not np.isfinite(causal_total) or causal_total <= 0.0:
        return None
    causal = causal / causal_total

    # Cible future-aware : repondération par exp(Δs_i(v)) (eq. 15-18).
    score_set = sorted(set(candidates) | {reference})
    scores = scorer.suffix_loglikelihood(prefix, tuple(score_set), suffix)
    lookup = {token: float(score) for token, score in zip(score_set, scores)}
    reference_score = lookup.get(reference)
    if reference_score is None or not np.isfinite(reference_score):
        return None
    deltas = np.array([lookup[v] - reference_score for v in candidates], dtype=np.float64)
    shifted = deltas - float(np.max(deltas))
    weights = causal * np.exp(shifted)
    weight_total = float(np.sum(weights))
    if not np.isfinite(weight_total) or weight_total <= 0.0:
        future = causal.copy()
    else:
        future = weights / weight_total

    # Vérification C6 : chaque distribution somme à 1 à 1e-5 près, sans NaN/Inf.
    sum_errors = [
        abs(float(np.sum(causal)) - 1.0),
        abs(float(np.sum(future)) - 1.0),
        abs(float(np.sum(exact)) - 1.0),
    ]
    if not (
        np.all(np.isfinite(causal))
        and np.all(np.isfinite(future))
        and np.all(np.isfinite(exact))
        and all(error <= 1e-5 for error in sum_errors)
    ):
        return None

    causal_kl = kl_exact_to_target(exact, causal)
    future_kl = kl_exact_to_target(exact, future)
    return {
        "config_index": f"N{BLOCK_SIZE}_V{len(candidates)}",
        "block_index": int(block_index),
        "position": position,
        "prompt": prompt_id,
        "prefix_length": len(prefix),
        "suffix_length": len(suffix),
        "visible_future": bool(suffix),
        "causal_kl": causal_kl,
        "future_kl": future_kl,
        "delta_kl": causal_kl - future_kl,
        "max_sum_error": float(max(sum_errors)),
    }


def _run_experiment(
    args: argparse.Namespace,
    model: object,
    tokenizer: object,
    scorer: object,
    device: torch.device,
) -> tuple[list[dict], dict]:
    """Parcours les prompts, bloc par bloc, position par position.

    Rend les lignes de marginaux et un résumé de géométrie. Le support est
    construit une fois par prompt ; le joint énuméré aussi ; les blocs sont
    échantillonnés depuis ce joint (loi conditionnelle du professeur).
    """
    started = time.time()
    payload = json.loads(Path(args.prompt_file).read_text(encoding="utf-8"))
    prompts = payload["prompts"]
    rng = np.random.default_rng(args.seed)

    records: list[dict] = []
    max_sum_error = 0.0
    prefix_lengths: list[int] = []
    suffix_lengths: list[int] = []

    for prompt in prompts:
        prompt_id = prompt["id"]
        encoded = tokenizer(prompt["text"], return_tensors="pt")
        prompt_ids_tensor = encoded["input_ids"].to(device)[0]
        prompt_ids = tuple(int(value) for value in prompt_ids_tensor.tolist())

        support = _build_restricted_support(
            model, prompt_ids_tensor, BLOCK_SIZE, int(args.k), device
        )
        # Le joint ne dépend que du support (donc du prompt) : énuméré une fois.
        sequences, log_joint = support.enumerate_joint()

        for block_index in range(args.blocks):
            idx = int(rng.integers(0, sequences.shape[0]))
            # Une ligne du joint énuméré = un bloc complété x̂ réel (tokens du
            # support restreint) : c'est ce bloc qui fournit Â_i / Ĉ_i / x̂_i et
            # dont l'état visible conditionne le posterior exact. Prendre quatre
            # indices à la place aurait généré des séquences incompatibles.
            block = tuple(int(token) for token in sequences[idx].tolist())
            for position in MASK_POSITIONS:
                record = _marginal_for_position(
                    scorer,
                    prompt_ids,
                    block,
                    position,
                    support,
                    sequences,
                    log_joint,
                    block_index,
                    prompt_id,
                )
                if record is None:
                    continue
                records.append(record)
                prefix_lengths.append(record["prefix_length"])
                suffix_lengths.append(record["suffix_length"])
                max_sum_error = max(max_sum_error, record["max_sum_error"])
                _check_budget(started)
                if args.progress_interval > 0 and (block_index + 1) % args.progress_interval == 0:
                    elapsed = time.monotonic() - started
                    print(
                        f"[j18] blocs {block_index + 1}/{args.blocks} | "
                        f"{len(records)} marginaux | {elapsed:6.1f}s | "
                        f"VRAM pic {_gpu_peak_gib():.1f} GiB",
                        flush=True,
                    )

    summary = {
        "n_prompts": len(prompts),
        "n_marginals": len(records),
        "prefix_length_min": min(prefix_lengths) if prefix_lengths else None,
        "prefix_length_max": max(prefix_lengths) if prefix_lengths else None,
        "suffix_length_min": min(suffix_lengths) if suffix_lengths else None,
        "suffix_length_max": max(suffix_lengths) if suffix_lengths else None,
        "max_sum_error": float(max_sum_error),
    }
    return records, summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="J18 : expérience de décision à budget borné (FP32 TeacherScorer)."
    )
    parser.add_argument("--model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16", "float16"), default=DEFAULT_DTYPE
    )
    parser.add_argument("--k", type=int, default=K_CANDIDATES)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS_PER_PROMPT)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=20,
        help="impression de progression toutes les N blocs (0 pour désactiver)",
    )
    parser.add_argument(
        "--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE, help="sous-ensemble de prompts"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap-seed", type=int, default=SEED)
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _command_line() -> str:
    """Commande exacte du run (ADR-002 : `producer.command`)."""
    parts = [f"python {sys_executable_name()}"]
    argv = sys.argv[1:]
    parts.append(argv[0] if argv else Path(__file__).name)
    parts.extend(argv[1:])
    return " ".join(shlex.quote(part) for part in parts)


def sys_executable_name() -> str:
    import sys

    return Path(sys.executable).name


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.time()
    device = torch.device(args.device)

    # Intégrité du producteur (code versionné avant le run).
    commit = _assert_producer_committed()

    # Sous-ensemble de prompts (Phase 1) : taille + hash immuable.
    payload = json.loads(Path(args.prompt_file).read_text(encoding="utf-8"))
    prompts = payload["prompts"]
    if len(prompts) > 50:
        raise ValueError(f"J18 : le sous-ensemble a {len(prompts)} prompts (> 50)")
    subset_bytes = Path(args.prompt_file).read_bytes()
    subset_hash = hashlib.sha256(subset_bytes).hexdigest()

    # Chargement du professeur gelé en FP32 (règle 9 + R1).
    from teacher_scorer_qwen3 import (  # noqa: E402
        Qwen3TeacherScorer,
        load_teacher,
        assert_supported_transformers,
    )

    assert_supported_transformers()
    model, source = load_teacher(
        args.model, device=device, dtype=args.dtype, local_files_only=args.local_files_only
    )
    environment = _cache_environment(source)
    _warn_if_cache_is_outside_repo(environment)
    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=args.local_files_only
    )
    scorer = Qwen3TeacherScorer(model, device=device)

    # Expérience.
    records, geometry = _run_experiment(args, model, tokenizer, scorer, device)
    wall_seconds = round(time.time() - started, 3)

    if not records:
        raise RuntimeError(
            "J18 : aucun marginal produit — géométrie ou sous-ensemble invalide"
        )
    # Garde géométrique : les préfixes/suffixes mesurés tombent dans [12,21]/[1,3].
    if not (PREFIX_LENGTH_MIN <= geometry["prefix_length_min"]
            and geometry["prefix_length_max"] <= PREFIX_LENGTH_MAX):
        raise RuntimeError(
            f"J18 : préfix hors géométrie "
            f"[{geometry['prefix_length_min']}, {geometry['prefix_length_max']}]"
        )
    if not (SUFFIX_LENGTH_MIN <= geometry["suffix_length_min"]
            and geometry["suffix_length_max"] <= SUFFIX_LENGTH_MAX):
        raise RuntimeError(
            f"J18 : suffix hors géométrie "
            f"[{geometry['suffix_length_min']}, {geometry['suffix_length_max']}]"
        )

    # Agrégation (fonctions de package, sans GPU).
    from academiia.dopd.diagnostic import evaluate_multiprompt  # noqa: E402

    statistics = evaluate_multiprompt(
        records, seed=args.bootstrap_seed, iterations=args.bootstrap_iterations
    )
    sign_p_value = _binominal_two_tailed_p_value(
        int(statistics["sign_test"]["positive"]),
        int(statistics["sign_test"]["negative"]),
    )

    distinct: dict[str, set[int]] = {}
    for record in records:
        distinct.setdefault(record["config_index"], set()).add(record["block_index"])
    blocks_per_configuration = {name: len(values) for name, values in distinct.items()}
    total_marginals = len(records)

    # C6 : intégrité numérique.
    finite = all(
        np.isfinite(record["causal_kl"])
        and np.isfinite(record["future_kl"])
        and np.isfinite(record["delta_kl"])
        for record in records
    )
    numerical_integrity_met = bool(finite and geometry["max_sum_error"] <= 1e-5)

    # C10 : non-régression.
    no_regression, _regression_diff = _no_regression_delta()

    # Décision (table §4).
    decision = _decide_j18_gate(
        statistics=statistics,
        sign_p_value=sign_p_value,
        blocks_per_configuration=blocks_per_configuration,
        total_marginals=total_marginals,
        numerical_integrity_met=numerical_integrity_met,
        manifest_present=True,
        teacher_frozen_met=_teacher_is_frozen(scorer),
        model_source_in_repo=environment["model_source_in_repo"],
        no_regression_met=no_regression,
    )

    # Manifeste ADR-002.
    builder = _ManifestBuilder(commit)
    manifest = builder.build_manifest(
        artifact=ArtifactRef(path=str(args.output), kind="eval", hashes={}),
        parent=ArtifactRef(
            path=args.model, kind="checkpoint", hashes=_checkpoint_hashes(source)
        ),
        config=LineageWriter.config_block(
            DOPDConfig(),
            device=str(device),
            dtype=args.dtype,
            block_size=BLOCK_SIZE,
            k=int(args.k),
            vocab_width=int(args.k),
            mask_positions=list(MASK_POSITIONS),
            arithmetic_dtype="float32",
            seed=int(args.seed),
            bootstrap_seed=int(args.bootstrap_seed),
            bootstrap_iterations=int(args.bootstrap_iterations),
            blocks_per_prompt=int(args.blocks),
            min_blocks_per_configuration=MIN_BLOCKS_PER_CONFIGURATION,
            min_marginals=MIN_MARGINALS,
            budget_max_gpu_seconds=MAX_GPU_SECONDS,
            budget_max_gpu_bytes=MAX_GPU_BYTES,
            # Précision de CHARGEMENT du professeur en production (FP32, J17 R1),
            # distincte de scorer_dtype (= dtype natif du checkpoint, BF16).
            scorer_load_precision="float32",
        ),
        data={
            "prompt_file": str(args.prompt_file),
            "prompt_subset_hash": subset_hash,
            "n_prompts": len(prompts),
            "model_source": source,
            "model_source_in_repo": environment["model_source_in_repo"],
            "model_revision": environment["model_revision"],
            "training.performed": False,
            "training.backward": False,
            "training.weights_updated": False,
            "training.requires_grad_all_false": bool(
                _teacher_is_frozen(scorer)
            ),
        },
        verification={
            "numerical_integrity_met": numerical_integrity_met,
            "no_regression": no_regression,
            "geometry": geometry,
            "wall_seconds": wall_seconds,
            "cuda_allocated_peak_bytes": _gpu_peak_bytes(),
        },
        command=_command_line(),
    )

    # Assemblage du résultat (le manifeste ADR-002 est intégré au JSON).
    result = {
        "kind": "dopd_j18_decision",
        "protocol": {
            "spec": "modules/dopd/docs/reports/J18_PRE.md",
            "model": args.model,
            "model_source": source,
            "model_source_in_repo": environment["model_source_in_repo"],
            "model_revision": environment["model_revision"],
            "device": str(device),
            "dtype": args.dtype,
            "torch_version": transformers.__version__,
            "python_version": platform.python_version(),
            "seed": int(args.seed),
            "bootstrap_seed": int(args.bootstrap_seed),
            "bootstrap_iterations": int(args.bootstrap_iterations),
            "block_size": BLOCK_SIZE,
            "k": int(args.k),
            "mask_positions": list(MASK_POSITIONS),
            "blocks_per_prompt": int(args.blocks),
            "prompt_file": str(args.prompt_file),
            "prompt_subset_hash": subset_hash,
            "n_prompts": len(prompts),
            "min_blocks_per_configuration": MIN_BLOCKS_PER_CONFIGURATION,
            "min_marginals": MIN_MARGINALS,
            "budget": {
                "max_gpu_seconds": MAX_GPU_SECONDS,
                "max_gpu_bytes": MAX_GPU_BYTES,
            },
            "producer_commit": commit,
            "source_sha256": _producer_source_hashes(),
        },
        "runtime": {
            "wall_seconds": wall_seconds,
            "max_rss_mib": round(_rss_mib(), 1),
            "cuda_allocated_peak_mib": round(_gpu_peak_bytes() / 1024**2, 1),
            "cuda_reserved_peak_mib": round(
                (torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0)
                / 1024**2,
                1,
            ),
        },
        "geometry": geometry,
        "numerical_integrity": {
            "finite": bool(finite),
            "max_distribution_sum_error": geometry["max_sum_error"],
            "met": numerical_integrity_met,
        },
        "statistics": {
            **statistics,
            "sign_test_exact_binomial": {
                "p_value": float(sign_p_value),
                "positive": int(statistics["sign_test"]["positive"]),
                "negative": int(statistics["sign_test"]["negative"]),
                "n": int(statistics["sign_test"]["n"]),
            },
        },
        "decision": decision,
        "lineage": manifest.to_dict(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(text, encoding="utf-8")
    result_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

    print(json.dumps({k: result[k] for k in ("kind", "decision", "geometry")}, indent=2))
    print(f"résultat écrit : {args.output}")
    print(f"sha256(results.json) = {result_sha256}")
    print(f"décision : {decision['status']}")
    return 0


def _rss_mib() -> float:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


if __name__ == "__main__":
    raise SystemExit(main())
