#!/usr/bin/env python3
"""Smoke et mesures du `TeacherScorer` réel d-OPD (J17) — inférence seule.

Ce harnais exerce le composant `modules/dopd/harness/teacher_scorer_qwen3.py`
sur Qwen3-0.6B et produit :

* le **contrôle d'exactitude** : chemin cache (préfixe partagé) vs chemin de
  **rejeu intégral** (contexte complet rejoué par branche) ;
* la **latence** : cellule de référence J14 (préfixe 1 024, suffixe 3, 16 et
  33 branches) et géométrie réelle d-OPD (états réels, 17 branches) ;
* le **contrôle FP32** : le même job rejoué en FP32, pour distinguer l'écart
  de précision BF16 d'un défaut de logique ;
* la **chaîne scorer → cible** : `compute_future_aware_target` (NumPy pur)
  alimenté par les sorties du scorer ;
* le **manifeste ADR-002** : `producer.module = "dopd"`, commit du code
  producteur, commande, versions, seeds, empreintes.

Ce que le harnais ne fait **jamais** : aucun `optimizer`, aucun `backward`,
aucun `requires_grad_(True)`, aucune écriture de checkpoint. Le professeur
est gelé et lu seulement ; l'empreinte sha256 de ses tenseurs est vérifiée
avant et après le run.

Ce que le harnais ne fait pas non plus : **aucun téléchargement**. Le
checkpoint doit déjà être présent dans le cache du dépôt
(`HF_HOME=$PWD/_artefact/hf-cache`) ou dans un répertoire local.

Commandes :

    HF_HOME="$PWD/_artefact/hf-cache" python \
        modules/dopd/harness/run_teacher_scorer_smoke.py \
        --model Qwen/Qwen3-0.6B --device cuda --dtype bfloat16 \
        --output modules/dopd/eval/teacher_scorer_qwen3_0.6b/results.json

    HF_HOME="$PWD/_artefact/hf-cache" python \
        modules/dopd/harness/run_teacher_scorer_smoke.py \
        --model Qwen/Qwen3-0.6B --device cpu --dtype float32 \
        --output modules/dopd/eval/teacher_scorer_qwen3_0.6b_cpu/results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
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
from transformers import AutoTokenizer  # noqa: E402

from academiia.core.common import sha256_file  # noqa: E402
from academiia.dopd import (  # noqa: E402
    ArtifactRef,
    DOPDConfig,
    LineageManifest,
    LineageWriter,
    ScoringRequest,
    compute_future_aware_target,
)

from teacher_scorer_qwen3 import (  # noqa: E402
    DEFAULT_TEACHER_MODEL,
    MAX_PREFIX_CACHE_BYTES,
    Qwen3TeacherScorer,
    load_teacher,
    resolve_model_source,
    teacher_topk,
)

#: Code producteur : son commit est consigné dans le manifeste ADR-002 et son
#: empreinte dans les résultats. Un résultat produit depuis une version de
#: travail non versionnée est **refusé**.
PRODUCER_SOURCES = (
    "modules/dopd/harness/teacher_scorer_qwen3.py",
    "modules/dopd/harness/run_teacher_scorer_smoke.py",
)

#: Seed figée (DOPD_SPEC §6.10.3) : aucune composante aléatoire dans le
#: scoring, la seed ne sert qu'à la cellule de référence J14.
TEACHER_SEED = 20261025

#: Géométrie de référence J14 (DOPD_SPEC §6.10.6).
REFERENCE_PREFIX_LENGTH = 1024
REFERENCE_SUFFIX_LENGTH = 3
REFERENCE_BRANCH_COUNTS = (16, 33)

#: Bloc d-OPD (§6.10.3) : N = 4 tokens par bloc.
BLOCK_SIZE = 4

#: Prompts du jeu versionné de J15 : quatre prompts, dont `p18` (prompt dégradé
#: connu de J15/J16) — le smoke doit fonctionner aussi sur lui.
DEFAULT_PROMPT_IDS = ("p01", "p05", "p12", "p18")
DEFAULT_PROMPT_FILE = (
    "modules/dopd/eval/annexe_d_qwen3_0.6b_multiprompt/prompts.json"
)

#: Nombre de répétitions de la mesure de latence (médiane).
LATENCY_REPEATS = 5

#: Nombre d'échauffements avant la mesure (comme la cellule J14).
WARMUP_REPEATS = 3


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke du TeacherScorer reel d-OPD (J17) : exactitude cache/rejeu, "
            "latence, chaine scorer -> cible. Aucun entrainement."
        )
    )
    parser.add_argument("--model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--prompt-file", type=Path, default=Path(DEFAULT_PROMPT_FILE))
    parser.add_argument(
        "--prompt-ids",
        default=",".join(DEFAULT_PROMPT_IDS),
        help="identifiants de prompts du fichier, séparés par des virgules",
    )
    parser.add_argument(
        "--positions",
        type=int,
        default=3,
        help=(
            "positions scorées par bloc : 3 = positions 0,1,2 (futur visible) ; "
            "4 = toutes les positions du bloc (la dernière n'a pas de futur)"
        ),
    )
    parser.add_argument("--seed", type=int, default=TEACHER_SEED)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-reference-geometry",
        action="store_true",
        help="saute la cellule de référence J14 (préfixe 1024) — réservé au CPU",
    )
    parser.add_argument(
        "--skip-fp32-control",
        action="store_true",
        help="saute le contrôle FP32 (exige un second chargement du modèle)",
    )
    return parser.parse_args(argv)


def _cache_environment(model_source: str) -> dict:
    """Décrit la redirection de cache réellement en vigueur (règle 9)."""
    try:
        Path(model_source).resolve().relative_to(_REPO_ROOT.resolve())
        in_repo = True
    except (ValueError, OSError):
        in_repo = False
    return {
        "hf_home": os.environ.get("HF_HOME"),
        "hf_home_source": (
            "process_environment" if _HF_HOME_AT_IMPORT else "harness_default"
        ),
        "huggingface_hub_cache": os.environ.get("HUGGINGFACE_HUB_CACHE"),
        "model_source": model_source,
        "model_source_in_repo": bool(in_repo),
    }


def _warn_if_cache_is_outside_repo(environment: dict) -> None:
    """Avertit si le snapshot lu est hors du dépôt (règle 9)."""
    if environment["model_source_in_repo"]:
        return
    print(
        "ATTENTION (regle 9) : le modele est lu hors du depot : "
        f"{environment['model_source']} ; HF_HOME={environment['hf_home']} "
        f"({environment['hf_home_source']}). Ecriture hors depot inventoriee dans "
        "modules/a2d/docs/reports/OUT_OF_WORKSPACE.md ; pour un prochain run, "
        "poser HF_HOME=$PWD/_artefact/hf-cache et y placer le snapshot.",
        file=sys.stderr,
    )


def _assert_producer_committed() -> str:
    """Exige que le code producteur soit présent et intact dans ``HEAD``.

    Un résultat doit désigner un commit dont le code correspond : sinon le
    manifeste ADR-002 pointerait vers un producteur introuvable.
    """
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
                "versionner le code avant d'executer le smoke"
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
                "committer les modifications avant d'executer le smoke"
            )
    return commit


def _weights_sha256(model: object) -> str:
    """Empreinte sha256 **du contenu** des poids, dans l'ordre des noms.

    Le parcours est déterministe (noms triés) et ne matérialise jamais
    l'ensemble des poids en mémoire : chaque tenseur est haché à la suite.
    """
    digest = hashlib.sha256()
    for name, parameter in sorted(model.state_dict().items(), key=lambda item: item[0]):
        tensor = parameter.detach().to("cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


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


def _load_prompts(path: Path, wanted: list[str]) -> list[dict]:
    """Charge les prompts demandés du jeu versionné de J15.

    Le champ `text` est utilisé, comme dans `diagnostic_annexe_d.py` : c'est
    la mise en forme exacte (`Question: …\\nAnswer:`) qui a été tokenisée en
    J15/J16. Le champ `question` n'est que la question brute.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    available = {entry["id"]: entry for entry in payload["prompts"]}
    selected = []
    for prompt_id in wanted:
        if prompt_id not in available:
            raise ValueError(
                f"prompt {prompt_id!r} absent de {path} ; ids disponibles : "
                f"{sorted(available)}"
            )
        entry = available[prompt_id]
        text = entry.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"prompt {prompt_id!r} sans champ 'text' exploitable")
        selected.append(entry)
    return selected


def _generate_block(
    model: object,
    prompt_ids: torch.Tensor,
    block_size: int,
) -> list[int]:
    """Génère un bloc de `N` tokens **greedily** (décodage argmax).

    Aucune composante aléatoire : le bloc est reproductible. Les paramètres
    de rollout du papier (température 1,0, top-p 1,0, top-k 0) sont consignés
    dans le manifeste ; un décodage greedy ne les mobilise pas.
    """
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


def _warmup(
    scorer: Qwen3TeacherScorer,
    prefix: tuple[int, ...],
    candidates: tuple[int, ...],
    suffix: tuple[int, ...],
    device: torch.device,
) -> None:
    """Échauffement du chemin cache, comme dans la cellule de référence J14.

    La première exécution d'un forward CUDA alloue les workspaces et compile
    les noyaux : la mesurer compterait ce coût unique dans la médiane. J14
    échauffait explicitement sur un cache jetable ; le même principe s'applique
    ici.
    """
    for _ in range(WARMUP_REPEATS):
        scorer.reset_prefix_cache()
        scorer.causal_prior(prefix)
        scorer.suffix_loglikelihood(prefix, candidates, suffix)
    _synchronize(device)


def _time_cache_job(
    scorer: Qwen3TeacherScorer,
    prefix: tuple[int, ...],
    candidates: tuple[int, ...],
    suffix: tuple[int, ...],
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Chronomètre un **job complet** (prior + scores) via le chemin cache.

    Le préremplissage du préfixe est invalidé avant chaque mesure : chaque
    répétition mesure un job isolé, comme la cellule de référence J14.
    """
    scorer.reset_prefix_cache()
    _synchronize(device)
    start = time.perf_counter()
    prior = scorer.causal_prior(prefix)
    scores = scorer.suffix_loglikelihood(prefix, candidates, suffix)
    _synchronize(device)
    return time.perf_counter() - start, prior, scores


def _time_replay_job(
    scorer: Qwen3TeacherScorer,
    prefix: tuple[int, ...],
    candidates: tuple[int, ...],
    suffix: tuple[int, ...],
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Chronomètre le chemin de rejeu intégral (sans cache)."""
    _synchronize(device)
    start = time.perf_counter()
    prior, scores = scorer.replay_job(prefix, candidates, suffix)
    _synchronize(device)
    return time.perf_counter() - start, prior, scores


def _reference_geometry(
    scorer: Qwen3TeacherScorer,
    *,
    seed: int,
    device: torch.device,
    repeats: int,
) -> list[dict]:
    """Cellule de référence J14 : préfixe 1 024, suffixe 3, 16 et 33 branches."""
    generator = torch.Generator(device=device).manual_seed(seed)
    entries: list[dict] = []
    for branches in REFERENCE_BRANCH_COUNTS:
        prefix = torch.randint(
            1000,
            100000,
            (1, REFERENCE_PREFIX_LENGTH),
            dtype=torch.long,
            device=device,
            generator=generator,
        )
        suffix = torch.randint(
            1000,
            100000,
            (1, REFERENCE_SUFFIX_LENGTH),
            dtype=torch.long,
            device=device,
            generator=generator,
        )
        candidates = torch.randint(
            1000,
            100000,
            (branches, 1),
            dtype=torch.long,
            device=device,
            generator=generator,
        )
        prefix_ids = tuple(int(value) for value in prefix[0].tolist())
        suffix_ids = tuple(int(value) for value in suffix[0].tolist())
        candidate_ids = tuple(int(value) for value in candidates[:, 0].tolist())

        cache_times: list[float] = []
        replay_times: list[float] = []
        cache_scores = np.empty(0, dtype=np.float64)
        replay_scores = np.empty(0, dtype=np.float64)
        cache_prior = np.empty(0, dtype=np.float64)
        replay_prior = np.empty(0, dtype=np.float64)
        _warmup(scorer, prefix_ids, candidate_ids, suffix_ids, device)
        for _ in range(repeats):
            elapsed, cache_prior, cache_scores = _time_cache_job(
                scorer, prefix_ids, candidate_ids, suffix_ids, device
            )
            cache_times.append(elapsed)
            elapsed, replay_prior, replay_scores = _time_replay_job(
                scorer, prefix_ids, candidate_ids, suffix_ids, device
            )
            replay_times.append(elapsed)

        entry = {
            "prefix_length": REFERENCE_PREFIX_LENGTH,
            "suffix_length": REFERENCE_SUFFIX_LENGTH,
            "branches": branches,
            "repeats": repeats,
            "cache_seconds_median": round(_median(cache_times), 6),
            "cache_seconds_all": [round(value, 6) for value in cache_times],
            "replay_seconds_median": round(_median(replay_times), 6),
            "replay_seconds_all": [round(value, 6) for value in replay_times],
            "speedup": round(_median(replay_times) / _median(cache_times), 3)
            if _median(cache_times) > 0
            else None,
            "max_abs_suffix_score_diff": round(
                float(np.max(np.abs(cache_scores - replay_scores))), 6
            ),
            "max_abs_prior_diff": round(
                float(np.max(np.abs(cache_prior - replay_prior))), 6
            ),
            # Diagnostic d'explication : part de l'écart imputable au lot
            # dans le chemin de rejeu lui-même (hors cache).
            "replay_batch_sensitivity": round(
                _replay_batch_sensitivity(scorer, prefix_ids, candidate_ids, suffix_ids), 6
            ),
            "suffix_score_abs_max": round(float(np.max(np.abs(replay_scores))), 6),
        }
        entries.append(entry)
        del prefix, suffix, candidates
    return entries


def _replay_batch_sensitivity(
    scorer: Qwen3TeacherScorer,
    prefix: tuple[int, ...],
    candidates: tuple[int, ...],
    suffix: tuple[int, ...],
) -> float:
    """Sensibilité du chemin de rejeu à la **taille de lot**.

    Le rejeu est rejoué branche par branche (lot de 1) et comparé au rejeu
    groupé (lot de `n`). L'écart mesuré est **entièrement dû au lot** : il
    isole la part de l'écart cache/rejeu qui ne vient pas du cache.

    Ce diagnostic ne modifie aucun critère pré-enregistré : il explique le
    critère C2, il ne le réécrit pas.
    """
    _, grouped = scorer.replay_job(prefix, candidates, suffix)
    singles = np.empty(len(candidates), dtype=np.float64)
    for index, candidate in enumerate(candidates):
        _, score = scorer.replay_job(prefix, (candidate,), suffix)
        singles[index] = float(score[0])
    return float(np.max(np.abs(grouped - singles)))


def _delta_diff(
    scores: np.ndarray,
    reference_index: int,
) -> np.ndarray:
    """Deltas relatifs `s_i(v) − s_i(x̂_i)` alignés sur les branches."""
    return scores - scores[reference_index]


def _real_states(
    scorer: Qwen3TeacherScorer,
    model: object,
    tokenizer: object,
    prompts: list[dict],
    *,
    k: int,
    positions: int,
    device: torch.device,
    repeats: int,
) -> tuple[list[dict], dict | None]:
    """États réels : prompts J15, bloc de N = 4 tokens, scoring par position.

    Les positions `0 .. N-2` ont un **futur visible** ; la position `N-1`
    n'en a pas et exerce le fallback causal (`no_visible_future`).

    Rend les entrées **et** le premier job réel (préfixe, suffixe, branches),
    réutilisé par le contrôle FP32 : le même job doit être rejoué en FP32
    pour distinguer l'écart de précision de l'écart de logique.
    """
    entries: list[dict] = []
    first_job: dict | None = None
    for prompt in prompts:
        encoded = tokenizer(prompt["text"], return_tensors="pt")
        prompt_ids = encoded["input_ids"].to(device)[0]
        block = _generate_block(model, prompt_ids, BLOCK_SIZE)
        for position in range(min(positions, BLOCK_SIZE)):
            prefix = tuple(int(value) for value in prompt_ids.tolist()) + tuple(
                block[:position]
            )
            suffix = tuple(block[position + 1 :])
            reference_id = int(block[position])
            prior = scorer.causal_prior(prefix)
            selected = teacher_topk(prior, k)
            candidates = tuple(int(value) for value in selected.tolist())
            branches = (
                candidates if reference_id in candidates else candidates + (reference_id,)
            )

            cache_times: list[float] = []
            replay_times: list[float] = []
            cache_scores = np.empty(0, dtype=np.float64)
            replay_scores = np.empty(0, dtype=np.float64)
            cache_prior = np.empty(0, dtype=np.float64)
            replay_prior = np.empty(0, dtype=np.float64)
            _warmup(scorer, prefix, branches, suffix, device)
            for _ in range(repeats):
                elapsed, cache_prior, cache_scores = _time_cache_job(
                    scorer, prefix, branches, suffix, device
                )
                cache_times.append(elapsed)
                elapsed, replay_prior, replay_scores = _time_replay_job(
                    scorer, prefix, branches, suffix, device
                )
                replay_times.append(elapsed)

            request = ScoringRequest(
                prefix_ids=prefix,
                suffix_ids=suffix,
                candidate_ids=candidates,
                reference_id=reference_id,
            )
            result = scorer.score(request)
            target = compute_future_aware_target(
                result.prior,
                candidate_indices=result.candidate_ids,
                candidate_loglikelihoods=result.candidate_loglikelihoods,
                reference_loglikelihood=result.reference_loglikelihood,
                visible_future=bool(suffix),
                arithmetic_dtype=DOPDConfig().arithmetic_dtype,
            )
            # Diagnostics d'explication (ne modifient aucun critère) :
            # l'écart dû au lot dans le **rejeu** lui-même, et l'écart sur
            # les **deltas** `Δs_i(v)`, qui est la grandeur réellement
            # consommée par la cible (eq. 16-17).
            batch_sensitivity = _replay_batch_sensitivity(scorer, prefix, branches, suffix)
            reference_index = branches.index(reference_id)
            cache_deltas = _delta_diff(cache_scores, reference_index)
            replay_deltas = _delta_diff(replay_scores, reference_index)
            entries.append(
                {
                    "prompt_id": prompt["id"],
                    "prompt_category": prompt.get("category"),
                    "position": position,
                    "prefix_length": len(prefix),
                    "suffix_length": len(suffix),
                    "n_candidates": len(candidates),
                    "n_branches": len(branches),
                    "reference_in_candidates": reference_id in candidates,
                    "has_visible_future": bool(suffix),
                    "cache_seconds_median": round(_median(cache_times), 6),
                    "replay_seconds_median": round(_median(replay_times), 6),
                    "max_abs_suffix_score_diff": round(
                        float(np.max(np.abs(cache_scores - replay_scores))), 6
                    ),
                    "max_abs_prior_diff": round(
                        float(np.max(np.abs(cache_prior - replay_prior))), 6
                    ),
                    "max_abs_delta_diff": round(
                        float(np.max(np.abs(cache_deltas - replay_deltas))), 6
                    ),
                    "replay_batch_sensitivity": round(batch_sensitivity, 6),
                    "suffix_score_abs_max": round(
                        float(np.max(np.abs(replay_scores))), 6
                    ),
                    "prior_sum": round(float(np.sum(prior)), 8),
                    "prior_finite": bool(np.all(np.isfinite(prior))),
                    "target": {
                        "applied": bool(target.applied),
                        "reason": target.reason,
                        "sum": round(float(np.sum(target.target)), 8),
                        "finite": bool(np.all(np.isfinite(target.target))),
                        "n_candidates": target.n_candidates,
                    },
                }
            )
            del prior, cache_prior, replay_prior, result, target
            if first_job is None:
                first_job = {
                    "prompt_id": prompt["id"],
                    "position": position,
                    "prefix_ids": prefix,
                    "suffix_ids": suffix,
                    "branch_ids": branches,
                    "reference_id": reference_id,
                }
    return entries, first_job


def _fp32_real_job_control(
    model_id: str,
    job: dict,
    *,
    device: torch.device,
    local_files_only: bool,
) -> dict:
    """Contrôle FP32 du **premier état réel** (proof of the BF16 diagnosis).

    Le même job — même préfixe, même suffixe, mêmes branches — est rejoué avec
    un modèle FP32. Si l'écart cache/rejeu y tombe à l'ordre de `1e-5`, alors
    l'écart BF16 observé vient de la précision, pas de la logique du cache.
    """
    print("[fp32] chargement du modèle FP32 (etat reel) ...", flush=True)
    model, source = load_teacher(
        model_id, device=device, dtype="float32", local_files_only=local_files_only
    )
    try:
        scorer = Qwen3TeacherScorer(model, device=device)
        prefix = tuple(int(value) for value in job["prefix_ids"])
        suffix = tuple(int(value) for value in job["suffix_ids"])
        branches = tuple(int(value) for value in job["branch_ids"])
        prior = scorer.causal_prior(prefix)
        cached = scorer.suffix_loglikelihood(prefix, branches, suffix)
        replay_prior, replayed = scorer.replay_job(prefix, branches, suffix)
        batch_sensitivity = _replay_batch_sensitivity(scorer, prefix, branches, suffix)
        return {
            "dtype": "float32",
            "model_source": source,
            "prompt_id": job["prompt_id"],
            "position": job["position"],
            "prefix_length": len(prefix),
            "suffix_length": len(suffix),
            "branches": len(branches),
            "max_abs_suffix_score_diff": round(
                float(np.max(np.abs(cached - replayed))), 9
            ),
            "max_abs_prior_diff": round(
                float(np.max(np.abs(prior - replay_prior))), 9
            ),
            "replay_batch_sensitivity": round(batch_sensitivity, 9),
            "suffix_score_abs_max": round(float(np.max(np.abs(replayed))), 6),
        }
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _fp32_control(
    model_id: str,
    *,
    seed: int,
    device: torch.device,
    local_files_only: bool,
) -> dict:
    """Contrôle FP32 : le même job de référence, en précision flottante 32."""
    print("[fp32] chargement du modèle FP32 ...", flush=True)
    model, source = load_teacher(
        model_id, device=device, dtype="float32", local_files_only=local_files_only
    )
    try:
        scorer = Qwen3TeacherScorer(model, device=device)
        generator = torch.Generator(device=device).manual_seed(seed)
        branches = REFERENCE_BRANCH_COUNTS[0]
        prefix = torch.randint(
            1000,
            100000,
            (1, REFERENCE_PREFIX_LENGTH),
            dtype=torch.long,
            device=device,
            generator=generator,
        )
        suffix = torch.randint(
            1000,
            100000,
            (1, REFERENCE_SUFFIX_LENGTH),
            dtype=torch.long,
            device=device,
            generator=generator,
        )
        candidates = torch.randint(
            1000,
            100000,
            (branches, 1),
            dtype=torch.long,
            device=device,
            generator=generator,
        )
        prefix_ids = tuple(int(value) for value in prefix[0].tolist())
        suffix_ids = tuple(int(value) for value in suffix[0].tolist())
        candidate_ids = tuple(int(value) for value in candidates[:, 0].tolist())
        _warmup(scorer, prefix_ids, candidate_ids, suffix_ids, device)
        prior = scorer.causal_prior(prefix_ids)
        cache_scores = scorer.suffix_loglikelihood(prefix_ids, candidate_ids, suffix_ids)
        replay_prior, replay_scores = scorer.replay_job(
            prefix_ids, candidate_ids, suffix_ids
        )
        return {
            "dtype": "float32",
            "model_source": source,
            "prefix_length": REFERENCE_PREFIX_LENGTH,
            "suffix_length": REFERENCE_SUFFIX_LENGTH,
            "branches": branches,
            "max_abs_suffix_score_diff": round(
                float(np.max(np.abs(cache_scores - replay_scores))), 9
            ),
            "max_abs_prior_diff": round(
                float(np.max(np.abs(prior - replay_prior))), 9
            ),
        }
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.time()
    producer_commit = _assert_producer_committed()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERREUR : CUDA demande mais indisponible", file=sys.stderr)
        return 2
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    source = resolve_model_source(args.model)
    cache = _cache_environment(source)
    _warn_if_cache_is_outside_repo(cache)

    print(f"[1/5] chargement {args.model} ({args.dtype}, {args.device})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=args.local_files_only
    )
    model, source = load_teacher(
        args.model,
        device=device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    weights_before = _weights_sha256(model)

    scorer = Qwen3TeacherScorer(model, device=device)

    prompt_ids = [
        value.strip() for value in args.prompt_ids.split(",") if value.strip()
    ]
    prompts = _load_prompts(args.prompt_file, prompt_ids)

    print(
        f"[2/5] etats reels : {len(prompts)} prompts x {args.positions} positions",
        flush=True,
    )
    real_states, first_job = _real_states(
        scorer,
        model,
        tokenizer,
        prompts,
        k=args.k,
        positions=args.positions,
        device=device,
        repeats=LATENCY_REPEATS,
    )

    reference = []
    if not args.skip_reference_geometry:
        print("[3/5] cellule de reference J14 (prefixe 1024) ...", flush=True)
        reference = _reference_geometry(
            scorer, seed=args.seed, device=device, repeats=LATENCY_REPEATS
        )

    fp32_control = None
    fp32_real_job = None
    if not args.skip_fp32_control:
        print("[4/5] controle FP32 ...", flush=True)
        fp32_control = _fp32_control(
            args.model,
            seed=args.seed,
            device=device,
            local_files_only=args.local_files_only,
        )
        if first_job is not None:
            fp32_real_job = _fp32_real_job_control(
                args.model,
                first_job,
                device=device,
                local_files_only=args.local_files_only,
            )

    weights_after = _weights_sha256(model)
    print("[5/5] verification des poids et ecriture ...", flush=True)

    config = DOPDConfig()
    command = (
        "python modules/dopd/harness/run_teacher_scorer_smoke.py "
        f"--model {args.model} --device {args.device} --dtype {args.dtype} "
        f"--k {args.k} --prompt-file {args.prompt_file} "
        f"--prompt-ids {args.prompt_ids} --positions {args.positions} "
        f"--seed {args.seed} "
        + ("--local-files-only " if args.local_files_only else "")
        + f"--output {args.output}"
        + (" --skip-reference-geometry" if args.skip_reference_geometry else "")
        + (" --skip-fp32-control" if args.skip_fp32_control else "")
    )

    report = {
        "kind": "dopd_teacher_scorer_smoke",
        "model": args.model,
        "model_source": source,
        "cache": cache,
        "device": args.device,
        "dtype": args.dtype,
        "gpu_name": (
            torch.cuda.get_device_name(0) if device.type == "cuda" else None
        ),
        "gpu_total_mib": (
            round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1)
            if device.type == "cuda"
            else None
        ),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": args.seed,
        "producer_commit": producer_commit,
        "source_sha256": {
            relative: sha256_file(str(_REPO_ROOT / relative))
            for relative in PRODUCER_SOURCES
        },
        "config": {
            "block_size": config.block_size,
            "candidate_k": config.candidate_k,
            "scorer_dtype": config.scorer_dtype,
            "arithmetic_dtype": config.arithmetic_dtype,
            "max_prefix_cache_bytes": MAX_PREFIX_CACHE_BYTES,
            "max_scorer_branches": config.max_scorer_branches,
            "latency_repeats": LATENCY_REPEATS,
            "warmup_repeats": WARMUP_REPEATS,
        },
        "reference_geometry": reference,
        "real_states": real_states,
        "fp32_control": fp32_control,
        "fp32_real_job_control": fp32_real_job,
        "training": {
            "performed": False,
            "optimizer": None,
            "backward": False,
            "weights_updated": weights_before != weights_after,
            "requires_grad_all_false": all(
                not parameter.requires_grad for parameter in model.parameters()
            ),
            "model_training_flag": bool(model.training),
        },
        "weights": {
            "sha256_before": weights_before,
            "sha256_after": weights_after,
            "unchanged": weights_before == weights_after,
        },
        "elapsed_seconds": round(time.time() - started, 2),
        "lineage": _ManifestBuilder(producer_commit)
        .build_manifest(
            artifact=ArtifactRef(path=str(args.output), kind="eval", hashes={}),
            parent=ArtifactRef(path=args.model, kind="checkpoint", hashes={}),
            config=LineageWriter.config_block(
                config,
                teacher_seed=args.seed,
                model_id=args.model,
                dtype=args.dtype,
                device=args.device,
                candidate_k=args.k,
                block_size=BLOCK_SIZE,
                prompt_file=str(args.prompt_file),
                prompt_ids=prompt_ids,
                positions_per_block=args.positions,
                latency_repeats=LATENCY_REPEATS,
                warmup_repeats=WARMUP_REPEATS,
                generation="greedy_argmax",
                rollout_temperature=1.0,
                rollout_top_p=1.0,
                rollout_top_k=0,
            ),
            data={
                "teacher": args.model,
                "prompt_file": str(args.prompt_file),
                "prompt_ids": prompt_ids,
                "scoring_geometry": "prefix_cache_shared_suffix_per_branch",
                "reference_geometry": {
                    "prefix_length": REFERENCE_PREFIX_LENGTH,
                    "suffix_length": REFERENCE_SUFFIX_LENGTH,
                    "branches": list(REFERENCE_BRANCH_COUNTS),
                },
                "cache": cache,
                "source_sha256": {
                    relative: sha256_file(str(_REPO_ROOT / relative))
                    for relative in PRODUCER_SOURCES
                },
            },
            verification={
                "kind": "teacher_scorer_smoke",
                "n_real_states": len(real_states),
                "n_reference_cells": len(reference),
                "fp32_control_executed": fp32_control is not None,
                "fp32_real_job_control_executed": fp32_real_job is not None,
                "weights_unchanged": weights_before == weights_after,
                "requires_grad_all_false": all(
                    not parameter.requires_grad for parameter in model.parameters()
                ),
                "training_performed": False,
            },
            command=command,
        )
        .to_dict(),
    }

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload + "\n", encoding="utf-8")
    print(f"resultat ecrit : {output_path}")
    print(
        f"etats reels : {len(real_states)} ; poids inchangés : "
        f"{weights_before == weights_after}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
