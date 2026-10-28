"""Harnais de decision J18R — mesure reconstruite de l'effet future-aware.

Code producteur de la phase 2, ecrit en phase 1 apres acceptation ecrite du
pre-enregistrement fige `modules/dopd/docs/reports/J18R_PRE.md`. Ce fichier est
**hors paquet** : torch et transformers y sont autorises (ADR-001), mais ils ne
sont importes que dans les fonctions d'execution (`_torch`, `_transformers`),
jamais au niveau module, pour que les tests purs T1-T6 s'exécutent sans torch
(§8, test T7 ; ADR-004).

Aucun run n'est execute par cette mission : la mesure est declenchee par la
commande canonique §7.3, une fois la revue N1/N2 et l'acceptation ecrite de
l'utilisateur obtenues (§11). Aucun push (règle 11 d'AGENTS.md).

Sections implémentées : géométrie §1, cibles §2, criteres et décision §4,
exactitude numerique §5, bras N6 §6, gouvernance §7.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from academiia.core import git_commit, sha256_file
from academiia.dopd import DOPDConfig
from academiia.dopd.diagnostic import (
    RestrictedSupport,
    bootstrap_delta_ci,
    bootstrap_prompt_ci,
    exact_token_posteriors,
    verify_enumeration_agreement,
    kl_exact_to_target,
    per_prompt_marginals,
    sample_compatible_index,
    sign_test,
    summarize_marginals,
)
from academiia.dopd.interfaces import ArtifactRef, LineageWriter
from academiia.dopd.target import (
    REASON_APPLIED,
    REASON_FALLBACK,
    REASON_NO_VISIBLE_FUTURE,
    candidate_union,
    compute_future_aware_target,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

MULTIPROMPT_SCHEMA = "dopd.prompts/1"
DEFAULT_TEACHER_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_PROMPT_FILE = "modules/dopd/eval/annexe_d_qwen3_0.6b_multiprompt/prompts.json"
DEFAULT_OUTPUT_DIR = "modules/dopd/eval/j18r_decision"

# ---------------------------------------------------------------------------
# §1 — géométrie figée
# ---------------------------------------------------------------------------

NOISE_LEVELS: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5)
MASK_RETAIN_PROBABILITY = 0.45

#: configurations par bras, ordre figé (§2.5). `k` n'existe que dans les bras où la
#: règle de production est définie ; `k < V` y est obligatoire (§1.6).
ANCHOR_CONFIGS: tuple[tuple[int, int], ...] = ((4, 6), (4, 12), (6, 6))
SENSITIVITY_CONFIGS: tuple[tuple[int, int], ...] = ANCHOR_CONFIGS
DECISION_CONFIGS: tuple[tuple[int, int, int], ...] = ((4, 12, 4), (4, 24, 8))
CONTROL_CONFIGS: tuple[tuple[int, int], ...] = ((4, 12), (4, 24))

#: §1.5 — échantillon deterministe du controle negatif : 3 prompts x 3 blocs x 2
#: configurations = 18 marginaux.
CONTROL_PROMPTS: tuple[str, ...] = ("p01", "p02", "p03")
CONTROL_BLOCKS: tuple[int, ...] = (0, 1, 2)
CONTROL_MAX_FUTURE_KL = 1e-3
CONTROL_MEDIAN_FUTURE_KL = 1e-5

#: §3.2 — exactement 100 blocs de base par (prompt, configuration).
BLOCKS_PER_PROMPT_CONFIG = 100
MIN_MARGINALS = 2_000

# ---------------------------------------------------------------------------
# §3 — graines figées (N5)
# ---------------------------------------------------------------------------

SEED_ANCHOR = 20261023
BOOTSTRAP_SEED_ANCHOR = 20261024
SEED_DECISION = 20261027
BOOTSTRAP_SEED_DECISION = 20261028
BOOTSTRAP_ITERATIONS = 10_000
GROUP_FIELDS: tuple[str, ...] = ("prompt", "config_index", "block_index")

#: §3.5 — rejeu intégral sur exactement 5 % des jobs.
REPLAY_MODULO = 20
REPLAY_FRACTION = 0.05
REPLAY_TOLERANCE = 1e-3

# ---------------------------------------------------------------------------
# §4 — critères et décision
# ---------------------------------------------------------------------------

C9_MEAN_REFERENCE = 0.0801905853381537
C9_MEAN_TOLERANCE = 5e-3
C9_MIN_COVERAGE = 0.90
C9_MAX_OUTLIER_FRACTION = 0.02
C9_OUTLIER_THRESHOLD = 0.4
SUM_ERROR_TOLERANCE = 1e-5
FLOAT32_APPLICATION_TOLERANCE = 1e-7

CRITERIA_KEYS: tuple[str, ...] = ("C4", "C5", "C6", "C7", "C8", "C9", "C10")
RESERVE_KEYS: tuple[str, ...] = ("R_a", "R_b", "R_c")
DECISION_STATUSES: tuple[str, ...] = ("STOP", "INVALIDE", "GO", "GO conditionnel", "NO-GO")

#: §4.2 R_c — niveaux attendus de la grille figée §1.2 (σ ≥ 1,0) : la réserve est
#: universelle et ne se lève pas sur des données incomplètes.
RESERVE_C_NOISE_LEVELS: tuple[float, ...] = tuple(noise for noise in NOISE_LEVELS if noise >= 1.0)

# ---------------------------------------------------------------------------
# §7 — budgets et gouvernance
# ---------------------------------------------------------------------------

MAX_GPU_SECONDS = 14_400
MAX_GPU_MIB = 16_384
MAX_RSS_GIB = 24
N6_MAX_SECONDS = 1_800
RSS_CHECK_PERIOD_SECONDS = 1.0

#: §9.2 — hashes des sources figées au moment du gel du PRE (vérifiés en phase 1).
FROZEN_HASHES: dict[str, str] = {
    "modules/dopd/dopd/target.py": "e9568bf3329c8c0a17d09019c966ef2f8919ed132ea9240d9cc57005097b4556",
    "modules/dopd/dopd/diagnostic.py": "e936ca5ada3f217ea3c10bcda751e685c60f285585bc59c010b0de9e232aa6a8",
    "modules/dopd/dopd/config.py": "2efeb22cf11f1250d7f09b72ec43d59727c901f2c9b9d23e1fb58cc544de3eed",
    "modules/dopd/dopd/interfaces.py": "1af4b1e441bf54e4b496da1034411b741ec7419ca8022c59c2a0fe4c27f40ce0",
    "modules/dopd/dopd/__init__.py": "8598b8c8bed0f46285ac36f5f3b63662d733c6d492e2d8ac5743393c84eafaf6",
    "modules/dopd/harness/teacher_scorer_qwen3.py": "091db9dc008abf0df2621be8e16e1cc9e79791f28a15392babb90378b783d0be",
    "modules/dopd/harness/diagnostic_annexe_d.py": "8132c775ca455e95b6443e47e3e46cf0de7955227ff06a0f656b2f4faacd1afd",
    "modules/dopd/eval/annexe_d_qwen3_0.6b_multiprompt/prompts.json": "87aa12c9b48a479b0709d7de01912cc00ce77b4ff679fc1f89cfeccb641ba3e5",
    "modules/dopd/eval/annexe_d_qwen3_0.6b_multiprompt/results.json": "ed41ed7e4aee15c66289ae0d3865efcdaf4e0327761d735a62da2f26e742c5f5",
    "modules/dopd/eval/j18_decision/results.json": "4e891a3d9cd54b26f351e629e6dc38b13d46525911db275664b2a425d9fa225e",
    "modules/dopd/docs/reports/AUDIT_J18.md": "0e211490a2bb99650c3f7226f320206c4287c5a02fb6657f82c3cf45d302eaae",
}

#: §7.10 — les deux seules lignes autorisées dans `diagnostic.py`.
CONTROLLED_DIAGNOSTIC_LINES: tuple[str, ...] = (
    '        "mean_causal_kl": summary["mean_causal_kl"],',
    '        "mean_future_kl": summary["mean_future_kl"],',
)

#: §7.2 — shards par bras ; `results.json` n'est jamais écrit si un bras requis est
#: incomplet.
ARM_SHARDS: dict[str, str] = {
    "control": "arm_control.json",
    "anchor": "arm_anchor.json",
    "sensitivity_fp32": "arm_sensitivity_raw.json",
    "decision": "arm_decision.json",
    "n6_chat_template": "arm_n6.json",
}
REQUIRED_ARMS: tuple[str, ...] = ("control", "anchor", "decision")

#: §7.3 — commande canonique rejouable ; écart avec la commande réellement exécutée
#: = violation C6.
CANONICAL_COMMAND = (
    "python modules/dopd/harness/run_j18r_decision.py --model Qwen/Qwen3-0.6B "
    "--device cuda --dtype-anchor bfloat16 --dtype-decision float32 "
    "--dtype-control float32 --dtype-sensitivity float32 "
    f"--prompt-file {DEFAULT_PROMPT_FILE} --blocks 100 --seed 20261027 "
    "--bootstrap-seed 20261028 --bootstrap-iterations 10000 --anchor-seed 20261023 "
    "--anchor-bootstrap-seed 20261024 --replay-fraction 0.05 --replay-tolerance 1e-3 "
    "--n6-chat-template --n6-max-seconds 1800 --max-seconds 14400 --max-gpu-mib 16384 "
    f"--max-rss-gib 24 --output-dir {DEFAULT_OUTPUT_DIR}"
)

#: §7.3b — invocations d'ancrage, chaînes figées (harnais J15 gelé, importé en
#: lecture seule).
J15_HARNESS = "modules/dopd/harness/diagnostic_annexe_d.py"
ANCHOR_INVOCATION = (
    f"python {J15_HARNESS} --model Qwen/Qwen3-0.6B --device cuda --dtype bfloat16 "
    f"--prompt-file {DEFAULT_PROMPT_FILE} --blocks 100 --seed 20261023 "
    "--bootstrap-seed 20261024 --local-files-only "
    f"--output {DEFAULT_OUTPUT_DIR}/arm_anchor_raw.json --top-k 16"
)
SENSITIVITY_INVOCATION = ANCHOR_INVOCATION.replace("--dtype bfloat16", "--dtype float32").replace(
    "arm_anchor_raw.json", "arm_sensitivity_raw.json"
)

#: §9.3 — références d'ancrage et de tolérance (rappel obligatoire, jamais recalculées).
REFERENCE_J15 = {
    "aggregate_delta_kl": 0.0801905853381537,
    "aggregate_ci95": [0.06381534237632878, 0.09617038274240884],
    "median_delta_kl": 0.0,
    "harmful_tail_fraction": 0.1330,
    "degraded_prompts": ["p06", "p10", "p18"],
    "relative_reduction": 0.0858,
    "dtype": "bfloat16",
    "bf16_residual_max": 0.395069,
    "replay_reference_max_delta": 6.2e-05,
}

VOCAB_SIZE_QWEN3 = 151_936

# ---------------------------------------------------------------------------
# §1 — géométrie pure (NumPy)
# ---------------------------------------------------------------------------


def rng_blocks(seed: int, block_size: int, vocab_width: int) -> np.random.Generator:
    """rng des blocs de base et du masquage (dérivation figée §3.3)."""
    return np.random.default_rng(seed + 1000 * block_size + vocab_width)


def rng_perturbation(seed: int, block_size: int, vocab_width: int, noise: float) -> np.random.Generator:
    """rng de la perturbation `Z_X`, tirée une seule fois par (configuration, sigma)."""
    return np.random.default_rng(seed + 100_000 * block_size + 10_000 * vocab_width + int(noise * 10))


def rng_future(seed: int, block_size: int, vocab_width: int, noise: float) -> np.random.Generator:
    """rng du futur compatible, tiré séparément (dérivation figée §3.3)."""
    return np.random.default_rng(seed + 1000 * block_size + vocab_width + int(noise * 10))


def masked_positions(block_size: int, rng: np.random.Generator) -> tuple[int, ...]:
    """Rétention 0,45 par position (convention J14/J15, §1.1)."""
    retained = rng.random(block_size) < MASK_RETAIN_PROBABILITY
    masked = tuple(position for position, keep in enumerate(retained) if not keep)
    if not masked:
        masked = (int(rng.integers(block_size)),)
    return masked


def single_masked_position(block_size: int, rng: np.random.Generator) -> tuple[int, ...]:
    """§1.5 : un seul masque par bloc, toutes les autres positions visibles."""
    return (int(rng.integers(block_size)),)


def visible_state(state: Sequence[int], masked: Sequence[int], block_size: int) -> dict[int, int]:
    """État visible d'un bloc de base : positions conservées -> token."""
    hidden = set(masked)
    return {position: int(state[position]) for position in range(block_size) if position not in hidden}


def has_visible_future(position: int, masked: Sequence[int], block_size: int) -> bool:
    """§1.4 : une position conservée existe strictement à droite de `position`."""
    hidden = set(masked)
    return any(later > position and later not in hidden for later in range(block_size))


def compatible_mask(sequences: np.ndarray, visible: Mapping[int, int]) -> np.ndarray:
    """Séquences énumérées compatibles avec l'état visible (§1.2)."""
    mask = np.ones(sequences.shape[0], dtype=bool)
    for position, token_id in visible.items():
        mask &= sequences[:, position] == int(token_id)
    return mask


def draw_base_block(sequences: np.ndarray, log_joint: np.ndarray, rng: np.random.Generator) -> tuple[int, ...]:
    """Bloc de base tiré proportionnellement au joint du professeur (écart J)."""
    index = sample_compatible_index(log_joint, np.ones(sequences.shape[0], dtype=bool), rng)
    return tuple(int(token) for token in sequences[index])


def topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    """Top-`k` **dans le support de largeur V** (§1.6, N3), tri stable puis identifiant croissant.

    Même convention que `teacher_topk` : tri stable pour départager, puis ordre
    croissant, afin que la règle de production soit déterministe.
    """
    if k < 1:
        raise ValueError(f"k doit être >= 1, reçu {k}")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values doit être un vecteur 1-D non vide")
    return np.sort(np.argsort(-array, kind="stable")[:k].astype(np.int64))


# ---------------------------------------------------------------------------
# §1.1 — construction économe du support (B2) et mémoire chiffrée
# ---------------------------------------------------------------------------

ForwardFn = Callable[[Sequence[tuple[int, ...]]], np.ndarray]


def support_build_peak_bytes(
    block_size: int,
    vocab_width: int,
    vocab_size: int,
    *,
    economical: bool = True,
    batch_size: int = 32,
    itemsize: int = 8,
) -> dict[str, float]:
    """Mémoire du build de support, chiffrée (§1.1, B2).

    Chemin J15 (non économe) : matrice des conditionnelles plein vocabulaire aux
    préfixes `Σ_d V^d × vocab_size × itemsize`, plus la double copie de
    concaténation -> pic ≈ 2 × stockage (`(4,24)` : ≈ 35 Go).

    Chemin économe : seules les colonnes sélectionnées sont conservées
    (`Σ_d V^d × V` entrées) + un tampon de lot transitoire.
    """
    entries = sum(vocab_width**depth for depth in range(block_size))
    if economical:
        stored = entries * vocab_width * itemsize
        transient = min(batch_size, vocab_width ** (block_size - 1)) * vocab_size * itemsize
        return {
            "economical": True,
            "retained_entries": int(entries),
            "stored_bytes": float(stored),
            "peak_bytes": float(stored + transient),
        }
    stored = entries * vocab_size * itemsize
    return {
        "economical": False,
        "retained_entries": int(entries * vocab_size),
        "stored_bytes": float(stored),
        "peak_bytes": float(2.0 * stored),
    }


def assert_support_build_fits(
    block_size: int,
    vocab_width: int,
    vocab_size: int,
    *,
    economical: bool,
    max_rss_gib: float,
    batch_size: int = 32,
) -> dict[str, float]:
    """Garde d'allocation pré-enregistrée : refuser le chemin J15 quand son pic dépasse.

    Le plafond §7.8 (24 GiB) est placé entre les deux régimes : il laisse la marge à
    la construction économe et fait échouer vite toute régression vers le chemin J15
    sur `V = 24`.
    """
    estimate = support_build_peak_bytes(
        block_size, vocab_width, vocab_size, economical=economical, batch_size=batch_size
    )
    limit = max_rss_gib * 1024**3
    if estimate["peak_bytes"] > limit:
        raise MemoryError(
            f"construction de support ({'économe' if economical else 'J15'}) pour "
            f"(N={block_size}, V={vocab_width}) : pic estimé "
            f"{estimate['peak_bytes'] / 1024**3:.2f} GiB > plafond {max_rss_gib} GiB"
        )
    return estimate


def build_economical_support(
    forwards: ForwardFn,
    block_size: int,
    vocab_width: int,
    *,
    option: str = "A",
    arithmetic_dtype: str = "float64",
    batch_size: int = 32,
) -> RestrictedSupport:
    """Support restreint (Annexe D.1), sans matrice plein vocabulaire persistée (B2).

    Option A (défaut, bras décision / contrôle / N6) : deux passes — la première
    accumule le mélange pondéré (eq. 21) et sélectionne les `V` tokens, la seconde
    rejoue **le même jeu de forwards** et ne garde que les colonnes sélectionnées.
    Pour un même jeu de forwards, la sortie est **bit-identique** à
    `_build_restricted_support` (J15) parce que l'ordre des lots, l'accumulation du
    mélange, le tri stable de sélection et la normalisation des poids de préfixe sont
    reproduits ligne à ligne (§8 T8).

    Option B : une seule passe, buffers en `float32` — équivalence tolérée
    (`max |Δ log-prob| ≤ 1e-4`).

    `forwards(prefixes)` rend les log-probabilités (log-softmax) de chaque préfixe sur
    tout le vocabulaire.
    """
    if option not in ("A", "B"):
        raise ValueError(f"option doit être 'A' ou 'B', reçu {option!r}")
    dtype = np.dtype(arithmetic_dtype)
    if dtype.itemsize not in (4, 8):
        raise ValueError(f"arithmetic_dtype doit être float32 ou float64, reçu {arithmetic_dtype!r}")
    if option == "B" and dtype.itemsize != 4:
        raise ValueError("l'option B exige des buffers float32")

    supports: list[np.ndarray] = []
    depth_prefixes: list[tuple[tuple[int, ...], ...]] = []
    restricted_conditionals: list[np.ndarray] = []

    prefix_candidates: list[tuple[int, ...]] = [()]
    prefix_log_weights = np.asarray([0.0], dtype=np.float64)

    for _position in range(block_size):
        depth_prefixes.append(tuple(prefix_candidates))

        # Passe 1 : mélange pondéré et sélection ; rien de plein vocabulaire n'est
        # conservé au-delà du lot courant.
        vocab_size = None
        mixture = None
        for start in range(0, len(prefix_candidates), batch_size):
            batch_prefixes = prefix_candidates[start : start + batch_size]
            batch_log_probs = np.asarray(forwards(batch_prefixes), dtype=dtype)
            if batch_log_probs.ndim != 2:
                raise ValueError("forwards doit rendre une matrice (n_prefixes, vocab_size)")
            if vocab_size is None:
                vocab_size = int(batch_log_probs.shape[1])
                mixture = np.zeros(vocab_size, dtype=np.float64)
            weights = np.exp(prefix_log_weights[start : start + len(batch_prefixes)])
            mixture += np.sum(np.exp(batch_log_probs) * weights[:, None], axis=0)
        if vocab_size is None:
            raise ValueError("aucun forward disponible pour construire le support")
        selected = np.sort(np.argsort(-mixture, kind="stable")[:vocab_width]).astype(np.int64)

        # Passe 2 : extraction des colonnes sélectionnées (mêmes forwards).
        rows: list[np.ndarray] = []
        for start in range(0, len(prefix_candidates), batch_size):
            batch_prefixes = prefix_candidates[start : start + batch_size]
            rows.append(np.asarray(forwards(batch_prefixes), dtype=dtype)[:, selected])
        conditional = np.concatenate(rows, axis=0) if rows else np.empty((0, vocab_width), dtype=dtype)

        supports.append(selected)
        restricted_conditionals.append(conditional)

        next_prefixes: list[tuple[int, ...]] = []
        next_log_weights: list[float] = []
        for prefix, prefix_weight, row in zip(prefix_candidates, prefix_log_weights, conditional, strict=True):
            for candidate_index, token_id in enumerate(selected):
                next_prefixes.append(prefix + (int(token_id),))
                next_log_weights.append(float(prefix_weight + float(row[candidate_index])))
        prefix_candidates = next_prefixes
        prefix_log_weights = np.asarray(next_log_weights, dtype=np.float64)
        prefix_log_weights -= np.logaddexp.reduce(prefix_log_weights)

    return RestrictedSupport(
        supports=tuple(supports),
        prefixes=tuple(depth_prefixes),
        conditional_log_probs=tuple(restricted_conditionals),
    )


# ---------------------------------------------------------------------------
# §2 — cibles
# ---------------------------------------------------------------------------


def inline_full_target(
    causal_prior: np.ndarray,
    scores: Sequence[float],
    reference_score: float,
) -> dict[str, object]:
    """T1 : cible future-aware sur le support complet, arithmétique float64 (§2.2).

    Rend la cible et ses composantes (deltas, facteurs, normaliseur) pour que le test
    T4 puisse comparer étape par étape avec `compute_future_aware_target`. Normaliseur
    non fini ou ≤ 0 : fallback causal déclaré (`applied = False`).
    """
    prior = np.asarray(causal_prior, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    if prior.ndim != 1 or values.ndim != 1 or values.size != prior.size:
        raise ValueError("causal_prior et scores doivent être des vecteurs 1-D alignés")
    deltas = values - float(reference_score)
    factors = np.exp(deltas)
    weights = prior * factors
    normalizer = float(np.sum(weights))
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        return {
            "target": prior.copy(),
            "applied": False,
            "reason": REASON_FALLBACK,
            "deltas": deltas,
            "factors": factors,
            "normalizer": normalizer,
            "n_candidates": int(prior.size),
        }
    return {
        "target": weights / normalizer,
        "applied": True,
        "reason": REASON_APPLIED,
        "deltas": deltas,
        "factors": factors,
        "normalizer": normalizer,
        "n_candidates": int(prior.size),
    }


def production_target(
    causal_prior: np.ndarray,
    candidate_positions: Sequence[int],
    candidate_scores: Sequence[float],
    reference_score: float,
    *,
    visible_future: bool,
    arithmetic_dtype: str = "float32",
) -> object:
    """T2 : règle de production exercée **par la fonction du paquet** (§2.3, N1).

    La reconstruction inline n'est pas autorisée pour T2. Les tokens du support hors
    candidats reçoivent le facteur 1 : `compute_future_aware_target` normalise sur tout
    le prior passé, ici le prior restreint au support de la position (§2.3).
    """
    return compute_future_aware_target(
        causal_prior,
        candidate_indices=list(candidate_positions),
        candidate_loglikelihoods=list(candidate_scores),
        reference_loglikelihood=float(reference_score),
        visible_future=visible_future,
        arithmetic_dtype=arithmetic_dtype,
    )


def float32_reference_target(
    causal_prior: np.ndarray,
    candidate_positions: Sequence[int],
    candidate_scores: Sequence[float],
    reference_score: float,
) -> np.ndarray:
    """Reference indépendante de l'eq. 18 en arithmétique float32 (C5(d), test T6).

    Miroir du paquet : prior normalisé en float64 puis cast float32, deltas et facteurs
    en float32, normaliseur sommé en float32. Sert à prouver que l'arithmétique déclarée
    est bien celle qui a calculé la cible — la sortie du paquet étant rendue en float64.
    """
    prior = np.asarray(causal_prior, dtype=np.float64)
    total = float(prior.sum())
    normalized = prior.copy() if total == 1.0 else prior / total
    positions = [int(index) for index in candidate_positions]
    factors = np.ones(prior.size, dtype=np.float32)
    deltas = np.asarray(
        [np.float32(float(score)) - np.float32(float(reference_score)) for score in candidate_scores],
        dtype=np.float32,
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        factors[positions] = np.exp(deltas)
    weights = prior.astype(np.float32) * factors
    normalizer = weights.sum(dtype=np.float32)
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        return normalized
    return (weights / normalizer).astype(np.float64)


# ---------------------------------------------------------------------------
# §2.4 — records et observations internes
# ---------------------------------------------------------------------------

RECORD_FIELDS: tuple[str, ...] = (
    "arm",
    "block_index",
    "causal_kl",
    "config_index",
    "delta_kl",
    "future_kl",
    "noise",
    "position",
    "prompt",
    "visible_future",
)
DECISION_EXTRA_FIELDS: tuple[str, ...] = ("full_kl", "prod_kl_teacheron", "delta_kl_teacheron")


@dataclass
class MarginalObservation:
    """Marginal d'un bras : record §2.4 plus les extras des publications §4.4.

    Les extras (masse captée, `|candidats|`, motif de fallback) ne figurent **pas**
    dans le schéma figé du record ; ils servent uniquement aux publications.
    """

    arm: str
    prompt: str
    config_index: object
    block_index: int
    position: int
    noise: float
    visible_future: bool
    causal_kl: float
    future_kl: float
    delta_kl: float
    full_kl: float | None = None
    prod_kl_teacheron: float | None = None
    delta_kl_teacheron: float | None = None
    reason: str = REASON_APPLIED
    n_candidates: int = 0
    mass_teacher: float | None = None
    mass_student: float | None = None
    job_index: int | None = None

    def record(self) -> dict[str, object]:
        """Ligne au schéma figé §2.4 ; rien d'autre n'est écrit dans les shards."""
        row: dict[str, object] = {
            "arm": self.arm,
            "block_index": int(self.block_index),
            "causal_kl": float(self.causal_kl),
            "config_index": int(self.config_index) if not isinstance(self.config_index, str) else self.config_index,
            "delta_kl": float(self.delta_kl),
            "future_kl": float(self.future_kl),
            "noise": float(self.noise),
            "position": int(self.position),
            "prompt": self.prompt,
            "visible_future": bool(self.visible_future),
        }
        if self.arm == "decision":
            for name in DECISION_EXTRA_FIELDS:
                value = getattr(self, name)
                row[name] = None if value is None else float(value)
        return row


@dataclass
class IntegrityStats:
    """§5.6 : finitude et erreur de somme sur toutes les distributions."""

    finite: bool = True
    max_sum_error: float = 0.0
    enumeration_agreement_max: float = 0.0
    reason_counts: dict[str, int] = field(default_factory=dict)
    float32_targets_seen: int = 0
    float32_mismatches: int = 0

    def observe_float32_application(
        self,
        arithmetic_dtype: str,
        applied: bool,
        target: np.ndarray,
        independent_reference: np.ndarray,
    ) -> None:
        """C5(d) : la cible est confrontee a une reference float32 independante (T6).

        Les cibles non appliquees (futur invisible, fallback du normaliseur) rendent la
        distribution causale telle quelle : l'arithmetique float32 n'y a rien calcule et ne
        peut donc pas etre verifiee sur ces cas.
        """
        if str(arithmetic_dtype) != "float32" or not bool(applied):
            return
        self.float32_targets_seen += 1
        gap = float(
            np.max(
                np.abs(np.asarray(target, dtype=np.float64) - np.asarray(independent_reference, dtype=np.float64))
            )
        )
        if gap > FLOAT32_APPLICATION_TOLERANCE:
            self.float32_mismatches += 1


    @property
    def float32_applied(self) -> bool:
        return bool(self.float32_targets_seen and not self.float32_mismatches)

    def observe_distribution(self, distribution: np.ndarray) -> None:
        values = np.asarray(distribution, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            self.finite = False
        error = abs(float(np.sum(values)) - 1.0)
        if error > self.max_sum_error:
            self.max_sum_error = error

    def count_reason(self, reason: str) -> None:
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1

    def to_dict(self) -> dict[str, object]:
        return {
            "finite": bool(self.finite),
            "max_sum_error": float(self.max_sum_error),
            "sum_error_tolerance": SUM_ERROR_TOLERANCE,
            "enumeration_agreement_max": float(self.enumeration_agreement_max),
            "float32_applied": bool(self.float32_applied),
            "float32_targets_seen": int(self.float32_targets_seen),
            "float32_mismatches": int(self.float32_mismatches),
            "reason_counts": dict(sorted(self.reason_counts.items())),
        }


@dataclass
class ReplayStats:
    """Publication `numerics.replay` (§5.2)."""

    n_jobs: int = 0
    n_jobs_expected: int = 0
    max_abs_delta: float = 0.0
    min_abs_delta: float = 0.0
    violations: list[int] = field(default_factory=list)
    tolerance: float = REPLAY_TOLERANCE
    modulo: int = REPLAY_MODULO

    def record_job(self, job_index: int, delta: float) -> None:
        delta = float(delta)
        if self.n_jobs == 0 or delta > self.max_abs_delta:
            self.max_abs_delta = delta
        if self.n_jobs == 0 or delta < self.min_abs_delta:
            self.min_abs_delta = delta
        self.n_jobs += 1
        if delta > self.tolerance:
            self.violations.append(int(job_index))

    def holds(self) -> bool:
        """5 % des jobs rejoués sans cache, aucun écart au-dessus de la tolérance."""
        expected = int(round(REPLAY_FRACTION * self.n_jobs_expected)) if self.n_jobs_expected else 0
        return not self.violations and self.n_jobs >= max(expected, 1)


    def to_dict(self) -> dict[str, object]:
        return {
            "n_jobs": int(self.n_jobs),
            "max_abs_delta": float(self.max_abs_delta),
            "min_abs_delta": float(self.min_abs_delta),
            "tolerance": float(self.tolerance),
            "violations": list(self.violations),
        }


# ---------------------------------------------------------------------------
# §5.1 — chemin de mesure (cache de préfixe) et rejeu (§5.2)
# ---------------------------------------------------------------------------


class ScoreProvider(Protocol):
    """Contrat du scoring d'un marginal : prior + scores candidats + référence."""

    def job(
        self,
        prefix: Sequence[int],
        position: int,
        tokens: Sequence[int],
        suffix: Sequence[int],
        reference_token: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Rend `(prior plein vocabulaire, scores alignés sur tokens, s_i(x̂_i))`."""

    def replay_job(
        self,
        prefix: Sequence[int],
        candidates: Sequence[int],
        suffix: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rejeu intégral sans cache : `(prior aux candidats, scores)` (§5.2)."""


# ---------------------------------------------------------------------------
# §1.2 / §2 — marginal et configuration (boucle figée)
# ---------------------------------------------------------------------------


def marginal_for_position(
    *,
    arm: str,
    prompt: str,
    config_index: object,
    block_index: int,
    position: int,
    noise: float,
    state: Sequence[int],
    masked: Sequence[int],
    future: Sequence[int],
    support: RestrictedSupport,
    sequences: np.ndarray,
    log_joint: np.ndarray,
    perturbed_log: np.ndarray,
    exact: Mapping[int, np.ndarray],
    provider: ScoreProvider,
    integrity: IntegrityStats,
    replay_stats: ReplayStats,
    job_counter: int,
    k: int | None,
    arithmetic_dtype: str = "float32",
) -> MarginalObservation:
    """Un marginal (§1.4) : T0, T1 et — quand `k` est défini — T2 + variante S1."""
    block_size = support.block_size
    tokens = support.supports[position]
    prefix = tuple(int(token) for token in future[:position])
    suffix = tuple(int(token) for token in future[position + 1 :])
    reference_token = int(future[position])

    prior_full, scores, reference_score = provider.job(prefix, position, tokens, suffix, reference_token)
    integrity.observe_distribution(prior_full)

    causal = support.causal_prior(prefix, position)
    integrity.observe_distribution(causal)
    exact_posterior = np.asarray(exact[position], dtype=np.float64)
    integrity.observe_distribution(exact_posterior)

    # §1.4 : sans futur visible, la cible future-aware est inopérante — T1 := T0
    # (cas limite eq. 12), comme le gate `visible_future` de T2 (§2.3). Sans ce
    # gate, T1 appliquerait les scores d'un futur échantillonné à une position
    # dont l'état visible ne contient aucun futur : la cible n'est pas la causale
    # et la troncature `full_kl - future_kl` y est fausse (R-J18R-05).
    visible_future = has_visible_future(position, masked, block_size)
    if visible_future:
        full_result = inline_full_target(causal, scores, reference_score)
    else:
        full_result = {
            "target": np.asarray(causal, dtype=np.float64).copy(),
            "applied": False,
            "reason": REASON_NO_VISIBLE_FUTURE,
            "n_candidates": int(causal.size),
        }
    integrity.count_reason(str(full_result["reason"]))
    if not np.all(np.isfinite(np.asarray(full_result["target"], dtype=np.float64))):
        integrity.finite = False

    causal_kl = kl_exact_to_target(exact_posterior, causal)
    full_kl = kl_exact_to_target(exact_posterior, np.asarray(full_result["target"], dtype=np.float64))

    observation = MarginalObservation(
        arm=arm,
        prompt=prompt,
        config_index=config_index,
        block_index=int(block_index),
        position=int(position),
        noise=float(noise),
        visible_future=bool(visible_future),
        causal_kl=float(causal_kl),
        future_kl=float(full_kl),
        delta_kl=float(causal_kl - full_kl),
        full_kl=None if arm == "anchor" else float(full_kl),
        reason=str(full_result["reason"]),
        n_candidates=int(full_result["n_candidates"]),
        job_index=int(job_counter),
    )

    if k is not None:
        student = exact_token_posteriors(sequences, perturbed_log, visible_state(state, masked, block_size), support.supports)
        integrity.observe_distribution(student[position])
        teacher_positions = topk_indices(causal, k)
        student_positions = topk_indices(student[position], k)
        candidates = candidate_union(teacher_positions, student_positions)
        result = production_target(
            causal,
            candidates,
            [scores[int(index)] for index in candidates],
            reference_score,
            visible_future=visible_future,
            arithmetic_dtype=arithmetic_dtype,
        )
        integrity.observe_distribution(np.asarray(result.target, dtype=np.float64))
        integrity.observe_float32_application(
            arithmetic_dtype,
            result.applied,
            result.target,
            float32_reference_target(
                causal, candidates, [scores[int(index)] for index in candidates], reference_score
            ),
        )
        if not np.all(np.isfinite(np.asarray(result.target, dtype=np.float64))):
            integrity.finite = False
        prod_kl = kl_exact_to_target(exact_posterior, np.asarray(result.target, dtype=np.float64))
        observation.future_kl = float(prod_kl)
        observation.delta_kl = float(causal_kl - prod_kl)
        observation.reason = str(result.reason)
        observation.n_candidates = int(result.n_candidates)
        observation.mass_teacher = float(np.sum(causal[candidates]))
        observation.mass_student = float(np.sum(student[position][candidates]))

        teacher_only = production_target(
            causal,
            teacher_positions,
            [scores[int(index)] for index in teacher_positions],
            reference_score,
            visible_future=visible_future,
            arithmetic_dtype=arithmetic_dtype,
        )
        prod_kl_teacheron = kl_exact_to_target(exact_posterior, np.asarray(teacher_only.target, dtype=np.float64))
        observation.prod_kl_teacheron = float(prod_kl_teacheron)
        observation.delta_kl_teacheron = float(causal_kl - prod_kl_teacheron)

    if job_counter % replay_stats.modulo == 0:
        replayed_prior, replayed_scores = provider.replay_job(prefix, [int(token) for token in tokens], suffix)
        ids = [int(token) for token in tokens]
        delta_prior = np.abs(
            np.asarray(replayed_prior, dtype=np.float64) - np.asarray(prior_full, dtype=np.float64)[ids]
        )
        delta_scores = np.abs(np.asarray(replayed_scores, dtype=np.float64) - np.asarray(scores, dtype=np.float64))
        replay_stats.record_job(job_counter, float(max(float(np.max(delta_prior)), float(np.max(delta_scores)))))

    return observation


def process_configuration(
    *,
    arm: str,
    prompt: str,
    config_index: object,
    support: RestrictedSupport,
    sequences: np.ndarray,
    log_joint: np.ndarray,
    provider: ScoreProvider,
    seed: int,
    blocks: int,
    noise_levels: Sequence[float] = NOISE_LEVELS,
    k: int | None = None,
    integrity: IntegrityStats,
    replay_stats: ReplayStats,
    arithmetic_dtype: str = "float32",
    block_sample: Sequence[int] | None = None,
    future_is_state: bool = False,
    budget: BudgetGuard | None = None,
) -> list[MarginalObservation]:
    """Une configuration d'un bras : blocs de base, niveaux de bruit, marginaux.

    Ordre figé (§1.1, §3.3) : bruit en externe, blocs en interne, positions
    ascendantes ; dérivations de rng identiques à J15. `future_is_state` est la
    géométrie de contrôle §1.5 (futur := bloc de base lui-même).
    """
    block_size = support.block_size
    vocab_width = int(support.supports[0].size)
    if k is not None and k >= vocab_width:
        raise ValueError(f"k < V obligatoire (§1.6) : reçu k={k} pour V={vocab_width}")

    rng = rng_blocks(seed, block_size, vocab_width)
    base_blocks: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for block_index in range(blocks):
        block = draw_base_block(sequences, log_joint, rng)
        masked = single_masked_position(block_size, rng) if arm == "control" else masked_positions(block_size, rng)
        base_blocks.append((block, masked))

    observations: list[MarginalObservation] = []
    job_counter = 0
    for noise in noise_levels:
        perturbation = rng_perturbation(seed, block_size, vocab_width, noise).normal(size=sequences.shape[0])
        perturbed_log = log_joint + float(noise) * perturbation
        noise_rng = rng_future(seed, block_size, vocab_width, noise)
        for block_index, (block, masked) in enumerate(base_blocks):
            if block_sample is not None and block_index not in set(block_sample):
                continue
            visible = visible_state(block, masked, block_size)
            exact = exact_token_posteriors(sequences, log_joint, visible, support.supports)
            if block_index == 0:
                agreement = verify_enumeration_agreement(sequences, log_joint, visible, support.supports)
                integrity.enumeration_agreement_max = max(integrity.enumeration_agreement_max, float(agreement))
            compatible = compatible_mask(sequences, visible)
            if future_is_state:
                future_block = tuple(int(token) for token in block)
            else:
                index = sample_compatible_index(perturbed_log, compatible, noise_rng)
                future_block = tuple(int(token) for token in sequences[index])

            for position in sorted(masked):
                job_counter += 1
                observations.append(
                    marginal_for_position(
                        arm=arm,
                        prompt=prompt,
                        config_index=config_index,
                        block_index=int(block_index),
                        position=int(position),
                        noise=float(noise),
                        state=block,
                        masked=masked,
                        future=future_block,
                        support=support,
                        sequences=sequences,
                        log_joint=log_joint,
                        perturbed_log=perturbed_log,
                        exact=exact,
                        provider=provider,
                        integrity=integrity,
                        replay_stats=replay_stats,
                        job_counter=job_counter,
                        k=k,
                        arithmetic_dtype=arithmetic_dtype,
                    )
                )
            if budget is not None:
                budget.check()
    return observations


# ---------------------------------------------------------------------------
# §4.4 — publications
# ---------------------------------------------------------------------------


def exact_binomial_two_tailed_p_value(positive: int, negative: int) -> float:
    """p-value binomiale exacte bilatérale (H0 : p = 0,5), en espace logarithmique.

    `scipy` n'est pas déclaré dans `pyproject.toml` : le calcul utilise `math.lgamma`,
    sans factorielle materialisée. Définition exacte : somme des masses `P(k) ≤ P(obs)`.
    """
    n = int(positive) + int(negative)
    if n <= 0:
        return float("nan")

    def log_pmf(k: int) -> float:
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2.0)

    observed = log_pmf(int(positive))
    total = 0.0
    for k in range(n + 1):
        value = log_pmf(k)
        if value <= observed + 1e-12:
            total += math.exp(value)
    return float(min(total, 1.0))


def _median(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def c1_by_noise(
    records: list[dict],
    *,
    bootstrap_seed: int,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, object]:
    """C1 par niveau de bruit et C1_{σ>0} (PRE-B5), publications obligatoires."""
    result: dict[str, object] = {}
    for noise in sorted({float(row["noise"]) for row in records}):
        subset = [row for row in records if float(row["noise"]) == noise]
        ci = bootstrap_delta_ci(subset, bootstrap_seed, iterations, group_fields=GROUP_FIELDS)
        mean = summarize_marginals(subset)["mean_delta_kl"]
        result[str(noise)] = {
            "n": len(subset),
            "mean_delta_kl": mean,
            "ci95": list(ci),
            "holds": bool(mean is not None and mean > 0.0 and ci[0] > 0.0),
        }
    positive_sigma = [row for row in records if float(row["noise"]) > 0.0]
    ci = bootstrap_delta_ci(positive_sigma, bootstrap_seed, iterations, group_fields=GROUP_FIELDS)
    mean = summarize_marginals(positive_sigma)["mean_delta_kl"]
    result["sigma_positive_only"] = {
        "n": len(positive_sigma),
        "mean_delta_kl": mean,
        "ci95": list(ci),
        "holds": bool(mean is not None and mean > 0.0 and ci[0] > 0.0),
    }
    return result


def arm_publications(
    observations: Sequence[MarginalObservation],
    *,
    arm: str,
    bootstrap_seed: int,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, object]:
    """Publications obligatoires d'un bras (§4.4), sur tous ses marginaux."""
    if not observations:
        return {"arm": arm, "n": 0}
    records = [observation.record() for observation in observations]
    visible_records = [row for row in records if row["visible_future"]]
    deltas = np.asarray([row["delta_kl"] for row in records], dtype=np.float64)
    summary = summarize_marginals(records)
    degraded = [float(row["delta_kl"]) for row in records if float(row["delta_kl"]) < 0.0]

    publications: dict[str, object] = {
        "arm": arm,
        "n": int(summary["n"]),
        "n_visible_future": len(visible_records),
        "n_no_visible_future": len(records) - len(visible_records),
        "mean_causal_kl": summary["mean_causal_kl"],
        "mean_future_kl": summary["mean_future_kl"],
        "mean_delta_kl": summary["mean_delta_kl"],
        "relative_reduction": (
            (summary["mean_causal_kl"] - summary["mean_future_kl"]) / summary["mean_causal_kl"]
            if summary["mean_causal_kl"] not in (None, 0.0)
            else float("nan")
        ),
        "median_delta_kl_all": _median([float(row["delta_kl"]) for row in records]),
        "median_delta_kl_visible_future": _median([float(row["delta_kl"]) for row in visible_records]),
        "degraded_fraction": float(len(degraded) / summary["n"]) if summary["n"] else float("nan"),
        "degraded_mean": float(np.mean(degraded)) if degraded else float("nan"),
        "degraded_worst": float(min(degraded)) if degraded else float("nan"),
        "per_prompt": per_prompt_marginals(records),
        "per_prompt_ci95": {
            prompt: list(ci)
            for prompt, ci in bootstrap_prompt_ci(records, seed=bootstrap_seed, iterations=iterations).items()
        },
        "sign_test": sign_test(deltas),
        "exact_binomial_p_value": exact_binomial_two_tailed_p_value(
            int(sum(1 for row in records if float(row["delta_kl"]) > 0.0)),
            int(sum(1 for row in records if float(row["delta_kl"]) < 0.0)),
        ),
        "per_noise": {},
    }

    for noise in sorted({float(row["noise"]) for row in records}):
        subset = [row for row in records if float(row["noise"]) == noise]
        noise_summary = summarize_marginals(subset)
        publications["per_noise"][str(noise)] = {
            "n": len(subset),
            "mean_causal_kl": noise_summary["mean_causal_kl"],
            "mean_future_kl": noise_summary["mean_future_kl"],
            "mean_delta_kl": noise_summary["mean_delta_kl"],
            "median_delta_kl": _median([float(row["delta_kl"]) for row in subset]),
        }

    if arm == "decision":
        publications["c1_by_noise"] = c1_by_noise(records, bootstrap_seed=bootstrap_seed, iterations=iterations)
        truncation = [
            float(row["full_kl"]) - float(row["future_kl"])
            for row in records
            if row.get("full_kl") is not None
        ]
        # §1.4 (R-J18R-05) : sur les marginaux sans futur visible, T1 := T0 et
        # T2 := T0, donc la troncature y est nulle. L'agrégat « tous » est donc
        # tiré par les seules lignes à futur visible ; le sous-ensemble visible
        # isole la part réellement perdue par la troncature du support.
        truncation_visible = [
            float(row["full_kl"]) - float(row["future_kl"])
            for row in records
            if row.get("full_kl") is not None and row.get("visible_future")
        ]
        publications["truncation_cost"] = {
            "n": len(truncation),
            "min": float(min(truncation)) if truncation else float("nan"),
            "mean": float(np.mean(truncation)) if truncation else float("nan"),
            "max": float(max(truncation)) if truncation else float("nan"),
        }
        publications["truncation_cost_visible_future"] = {
            "n": len(truncation_visible),
            "min": float(min(truncation_visible)) if truncation_visible else float("nan"),
            "mean": float(np.mean(truncation_visible)) if truncation_visible else float("nan"),
            "max": float(max(truncation_visible)) if truncation_visible else float("nan"),
        }
        candidates = [observation.n_candidates for observation in observations]
        publications["candidates_per_marginal"] = {
            "min": int(min(candidates)),
            "mean": float(np.mean(candidates)),
            "max": int(max(candidates)),
        }
        masses_teacher = [float(o.mass_teacher) for o in observations if o.mass_teacher is not None]
        masses_student = [float(o.mass_student) for o in observations if o.mass_student is not None]
        publications["captured_mass"] = {
            "teacher": {
                "min": float(min(masses_teacher)) if masses_teacher else float("nan"),
                "mean": float(np.mean(masses_teacher)) if masses_teacher else float("nan"),
                "max": float(max(masses_teacher)) if masses_teacher else float("nan"),
            },
            "student": {
                "min": float(min(masses_student)) if masses_student else float("nan"),
                "mean": float(np.mean(masses_student)) if masses_student else float("nan"),
                "max": float(max(masses_student)) if masses_student else float("nan"),
            },
        }
        publications["delta_kl_teacheron"] = {
            "mean": float(np.mean([float(o.delta_kl_teacheron) for o in observations if o.delta_kl_teacheron is not None])),
        }

    per_config: dict[str, dict[str, object]] = {}
    for row in records:
        entry = per_config.setdefault(
            str(row["config_index"]), {"blocks": set(), "block_indices": set(), "n_marginals": 0}
        )
        entry["blocks"].add((row["prompt"], int(row["block_index"])))
        entry["block_indices"].add(int(row["block_index"]))
        entry["n_marginals"] += 1
    publications["block_counts"] = {
        config: {
            "n_marginals": int(entry["n_marginals"]),
            "n_blocks_sampled": len(entry["blocks"]),
            "n_distinct_block_indices": len(entry["block_indices"]),
        }
        for config, entry in sorted(per_config.items())
    }
    return publications


# ---------------------------------------------------------------------------
# §4.5 — table de décision (priorité décroissante, figée)
# ---------------------------------------------------------------------------


def decide_j18r_issue(criteria: Mapping[str, object], reserves: Mapping[str, bool]) -> dict[str, object]:
    """Issue figée §4.5 : STOP > INVALIDE > GO > GO conditionnel > NO-GO.

    `criteria` porte les booléens `C4`..`C10`, `arms_complete`, `C1` (tous σ) et
    `C1_sigma_positive`. `STOP` et `INVALIDE` ne sont pas des décisions sur l'effet.
    """
    if not criteria.get("C9"):
        return {
            "status": "STOP",
            "priority": 1,
            "rule": "C9 (ancrage) échoue : dérive du harnais, aucune GO/NO-GO, diagnostic de dérive publié",
            "failed_criteria": ["C9"],
        }
    blockers = [name for name in CRITERIA_KEYS if not criteria.get(name)]
    if not criteria.get("arms_complete"):
        blockers.append("bras requis incomplet")
    if blockers:
        return {
            "status": "INVALIDE",
            "priority": 2,
            "rule": "défaut de validité du run : corriger et relancer, aucune décision sur l'effet",
            "failed_criteria": sorted(set(blockers)),
        }
    triggered = [name for name in RESERVE_KEYS if reserves.get(name)]
    if criteria.get("C1") and criteria.get("C1_sigma_positive") and not triggered:
        return {
            "status": "GO",
            "priority": 3,
            "rule": "C1 (tous σ) et C1_{σ>0} tenus, aucune réserve : ouverture d'un StudentPolicy réduit (≤ 0,6B) en J19",
            "failed_criteria": [],
        }
    if criteria.get("C1"):
        return {
            "status": "GO conditionnel",
            "priority": 4,
            "rule": (
                "C1 (tous σ) tenu mais C1_{σ>0} échoué ou réserve déclenchée : J19 ouvert "
                "uniquement avec TaskVerifier + porte de correction pré-enregistrés"
            ),
            "failed_criteria": ([] if criteria.get("C1_sigma_positive") else ["C1_sigma_positive"]) + triggered,
        }
    return {
        "status": "NO-GO",
        "priority": 5,
        "rule": "C1 (tous σ) non tenu sur run valide : pas d'ouverture de StudentPolicy — résultat légitime",
        "failed_criteria": ["C1"],
    }


def evaluate_reserves(publications: Mapping[str, object]) -> dict[str, bool]:
    """Réserves §4.2 sur le bras de décision (R_a, R_b, R_c).

    R_c est un quantificateur **universel** : elle n'est levée que si chaque niveau
    attendu `σ ≥ 1,0` de la grille figée est présent dans `per_noise` **et** y a une
    moyenne `≤ 0`. Un niveau attendu absent = données incomplètes : pas de réserve.
    """
    per_noise = publications.get("per_noise") or {}
    means = {float(key): value for key, value in per_noise.items()}
    reserve_c = all(
        level in means
        and means[level].get("mean_delta_kl") is not None
        and float(means[level]["mean_delta_kl"]) <= 0.0
        for level in RESERVE_C_NOISE_LEVELS
    )
    return {
        "R_a": bool(publications.get("median_delta_kl_visible_future", float("nan")) <= 0.0),
        "R_b": bool(publications.get("degraded_fraction", 0.0) >= 0.5),
        "R_c": bool(reserve_c),
    }


# ---------------------------------------------------------------------------
# §4.1 — critères falsifiables
# ---------------------------------------------------------------------------


def criterion_c9(anchor_records: list[dict], j15_reference: Mapping[str, object]) -> dict[str, object]:
    """Ancrage C9 : clause (a) écart moyen ≤ 5e-3 ; clause (b) couverture et 2 %."""
    reference_rows = list(j15_reference.get("rows") or [])
    aggregate = summarize_marginals(anchor_records)["mean_delta_kl"]
    reference_by_key = {
        (
            row.get("prompt"),
            row.get("config_index"),
            int(row.get("block_index", -1)),
            int(row.get("position", -1)),
            float(row.get("noise", 0.0)),
        ): float(row["delta_kl"])
        for row in reference_rows
    }
    shared: dict[tuple[object, ...], tuple[float, float]] = {}
    for row in anchor_records:
        key = (row["prompt"], row["config_index"], int(row["block_index"]), int(row["position"]), float(row["noise"]))
        if key in reference_by_key:
            shared[key] = (float(row["delta_kl"]), reference_by_key[key])
    outliers = [key for key, (observed, reference) in shared.items() if abs(observed - reference) > C9_OUTLIER_THRESHOLD]
    coverage = len(shared) / len(reference_rows) if reference_rows else float("nan")
    publications = {
        "holds": bool(
            aggregate is not None and abs(aggregate - C9_MEAN_REFERENCE) <= C9_MEAN_TOLERANCE
            and shared
            and coverage >= C9_MIN_COVERAGE
            and len(outliers) / len(shared) <= C9_MAX_OUTLIER_FRACTION
        ),
        "clause_a": {
            "observed_mean_delta_kl": aggregate,
            "reference": C9_MEAN_REFERENCE,
            "tolerance": C9_MEAN_TOLERANCE,
            "holds": bool(aggregate is not None and abs(aggregate - C9_MEAN_REFERENCE) <= C9_MEAN_TOLERANCE),
        },
        "clause_b": {
            "n_shared_keys": len(shared),
            "coverage_of_reference_lines": coverage,
            "min_coverage": C9_MIN_COVERAGE,
            "outlier_threshold": C9_OUTLIER_THRESHOLD,
            "outliers": len(outliers),
            "outlier_fraction": (len(outliers) / len(shared)) if shared else float("nan"),
            "mean_abs_gap_shared": float(np.mean([abs(a - b) for a, b in shared.values()])) if shared else float("nan"),
            "n_divergent_draws": len(outliers),
            "holds": bool(
                shared and coverage >= C9_MIN_COVERAGE and len(outliers) / len(shared) <= C9_MAX_OUTLIER_FRACTION
            ),
        },
    }
    return publications


def criterion_c10(repo_root: Path = _REPO_ROOT) -> dict[str, object]:
    """C10 : sources gelées intactes ; `diagnostic.py` = exactement les 2 lignes §7.10."""
    frozen_ok: dict[str, bool] = {}
    for relative, expected in FROZEN_HASHES.items():
        if relative == "modules/dopd/dopd/diagnostic.py":
            continue
        frozen_ok[relative] = bool(sha256_file(str(repo_root / relative)) == expected)
    path = repo_root / "modules/dopd/dopd/diagnostic.py"
    lines = path.read_text(encoding="utf-8").splitlines()
    present = all(line in lines for line in CONTROLLED_DIAGNOSTIC_LINES)
    rebuilt = "\n".join(line for line in lines if line not in CONTROLLED_DIAGNOSTIC_LINES) + "\n"
    hash_before_change = hashlib.sha256(rebuilt.encode("utf-8")).hexdigest()
    controlled = bool(present and hash_before_change == FROZEN_HASHES["modules/dopd/dopd/diagnostic.py"])
    return {
        "holds": bool(all(frozen_ok.values()) and controlled),
        "frozen_files": frozen_ok,
        "diagnostic_controlled_change": {
            "controlled_lines_present": present,
            "hash_before_change": hash_before_change,
            "matches_frozen_hash": controlled,
        },
    }


# ---------------------------------------------------------------------------
# §7.8 — budgets (horloge unique : time.monotonic)
# ---------------------------------------------------------------------------


def current_rss_bytes() -> float:
    """RSS courante du process (Linux), en octets."""
    try:
        resident = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
    except (OSError, IndexError):
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    return float(resident * resource.getpagesize())


class BudgetGuard:
    """Garde-budgets §7.8 : temps mural, pic GPU, RSS contrôlée à 1 Hz."""

    def __init__(
        self,
        *,
        max_seconds: float = MAX_GPU_SECONDS,
        max_gpu_mib: float = MAX_GPU_MIB,
        max_rss_gib: float = MAX_RSS_GIB,
        clock: Callable[[], float] = time.monotonic,
        rss: Callable[[], float] = current_rss_bytes,
        gpu_peak_bytes: Callable[[], float] = lambda: 0.0,
    ) -> None:
        self.max_seconds = float(max_seconds)
        self.max_gpu_mib = float(max_gpu_mib)
        self.max_rss_gib = float(max_rss_gib)
        self._clock = clock
        self._rss = rss
        self._gpu_peak_bytes = gpu_peak_bytes
        self.started = clock()
        self.last_check = 0.0
        self.checks = 0

    def check(self, now: float | None = None) -> dict[str, float]:
        now = self._clock() if now is None else float(now)
        if self.checks and now - self.last_check < RSS_CHECK_PERIOD_SECONDS:
            return {"elapsed_seconds": now - self.started}
        self.last_check = now
        self.checks += 1
        elapsed = now - self.started
        rss_gib = self._rss() / 1024**3
        gpu_mib = self._gpu_peak_bytes() / 1024**2
        if elapsed > self.max_seconds:
            raise RuntimeError(f"budget temps dépassé ({elapsed:.1f} s > {self.max_seconds:.1f} s)")
        if rss_gib > self.max_rss_gib:
            raise RuntimeError(f"plafond RSS dépassé ({rss_gib:.2f} GiB > {self.max_rss_gib} GiB)")
        if gpu_mib > self.max_gpu_mib:
            raise RuntimeError(f"pic GPU dépassé ({gpu_mib:.1f} MiB > {self.max_gpu_mib} MiB)")
        return {"elapsed_seconds": elapsed, "rss_gib": rss_gib, "gpu_peak_mib": gpu_mib}


class BackwardCounter:
    """§7.5 : `training.*` dérivé d'un compteur mesuré au runtime, jamais d'un littéral."""

    def __init__(self) -> None:
        self.count = 0

    def record(self, n: int = 1) -> None:
        self.count += int(n)

    def training_block(self, requires_grad_all_false: bool) -> dict[str, object]:
        performed = self.count > 0
        return {
            "performed": performed,
            "backward": performed,
            "weights_updated": performed,
            "backward_count": int(self.count),
            "requires_grad_all_false": bool(requires_grad_all_false),
        }


# ---------------------------------------------------------------------------
# §7.7 — cache HF redirigé et garde-fou « modèle dans le dépôt » AVANT chargement
# ---------------------------------------------------------------------------

HF_CACHE_DIR = "_artefact/hf-cache"


def redirect_hf_cache(repo_root: Path = _REPO_ROOT) -> dict[str, str]:
    """Impose la redirection du cache HF **par le harnais** (règle 9)."""
    target = str(repo_root / HF_CACHE_DIR)
    os.makedirs(target, exist_ok=True)
    os.environ["HF_HOME"] = target
    os.environ["HUGGINGFACE_HUB_CACHE"] = target
    return {"HF_HOME": target, "HUGGINGFACE_HUB_CACHE": target}


def verify_model_source_in_repo(model: str, repo_root: Path = _REPO_ROOT) -> dict[str, object]:
    """Résout la source et vérifie son appartenance au dépôt **avant** tout chargement.

    Violation → refus avant tout chargement (sortie 2, aucun forward), règle 9.
    """
    environment = redirect_hf_cache(repo_root)
    direct = Path(model)
    if direct.is_dir():
        snapshot = direct.resolve()
    else:
        org, name = model.split("/", 1)
        hub = Path(environment["HF_HOME"]) / "hub" / f"models--{org}--{name}" / "snapshots"
        if not hub.is_dir():
            raise SystemExit(2)
        candidates = sorted((p for p in hub.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise SystemExit(2)
        snapshot = candidates[-1]
    resolved = snapshot.resolve()
    root = repo_root.resolve()
    if root not in resolved.parents and resolved != root:
        raise SystemExit(2)
    weights = resolved / "model.safetensors"
    return {
        "model_source_in_repo": True,
        "snapshot": str(resolved),
        "model_revision": resolved.name,
        "sha256_model_safetensors": sha256_file(str(weights)) if weights.is_file() else None,
    }


# ---------------------------------------------------------------------------
# §7.2 — shards et sidecar de lignée
# ---------------------------------------------------------------------------


class _ManifestBuilder(LineageWriter):
    """Implémentation d'écriture du manifeste ADR-002 (le paquet reste pur)."""

    def write(self, manifest) -> str:
        return json.dumps(manifest.to_dict(), ensure_ascii=False, sort_keys=True)


def write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return sha256_file(str(path))


def run_arm_subprocess(invocation: str, repo_root: Path = _REPO_ROOT) -> int:
    """§7.3b : invocation du harnais J15 gelé, chaîne exactement figée."""
    return subprocess.run(shlex.split(invocation), cwd=repo_root, check=False).returncode


def ingest_arm_shard(path: Path, arm: str) -> list[dict]:
    """Ingère un shard brut produit par l'invocation §7.3b et étiquette les lignes."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows") or []
    labeled = []
    for row in rows:
        labeled.append({"arm": arm, **row})
    return labeled


# ---------------------------------------------------------------------------
# Phase 2 — exécution réelle (torch / transformers importés paresseusement)
# ---------------------------------------------------------------------------


def _torch():
    import torch

    return torch


def _transformers():
    import transformers

    return transformers


def model_forward(
    model,
    prompt_ids,
    device: object,
    batch_size: int = 32,
    arithmetic_dtype: str = "float64",
) -> ForwardFn:
    """Forward produisant les log-probabilités plein vocabulaire d'un lot de préfixes."""
    torch = _torch()

    def forwards(prefixes: Sequence[tuple[int, ...]]) -> np.ndarray:
        rows = [
            torch.cat((prompt_ids, torch.tensor(list(prefix), dtype=torch.long, device=device)))
            for prefix in prefixes
        ]
        with torch.no_grad():
            logits = model(input_ids=torch.stack(rows)).logits[:, -1, :]
            return torch.log_softmax(logits, dim=-1).cpu().numpy()

    return forwards


class ProductionScoreProvider:
    """Adapter du `TeacherScorer` de production sur le contrat §5.1."""

    def __init__(self, scorer) -> None:
        self._scorer = scorer

    def job(
        self,
        prefix: Sequence[int],
        position: int,
        tokens: Sequence[int],
        suffix: Sequence[int],
        reference_token: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        from academiia.dopd.interfaces import ScoringRequest

        request = ScoringRequest(
            prefix_ids=tuple(int(token) for token in prefix),
            suffix_ids=tuple(int(token) for token in suffix),
            candidate_ids=tuple(int(token) for token in tokens),
            reference_id=int(reference_token),
        )
        result = self._scorer.score(request)
        scores = np.asarray(result.candidate_loglikelihoods, dtype=np.float64)
        return np.asarray(result.prior, dtype=np.float64), scores, float(result.reference_loglikelihood)

    def replay_job(
        self,
        prefix: Sequence[int],
        candidates: Sequence[int],
        suffix: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        prior, scores = self._scorer.replay_job(
            [int(token) for token in prefix], [int(token) for token in candidates], [int(token) for token in suffix]
        )
        return np.asarray(prior, dtype=np.float64), np.asarray(scores, dtype=np.float64)


def load_prompt_set(path: Path) -> list[tuple[str, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != MULTIPROMPT_SCHEMA:
        raise ValueError(f"schema de fichier de prompts inattendu : {data.get('schema')!r}")
    prompts = [(entry["id"], entry["text"]) for entry in data.get("prompts", [])]
    if not prompts:
        raise ValueError("aucun prompt valide dans le fichier")
    return prompts

# ---------------------------------------------------------------------------
# Exécution des bras (phase 2)
# ---------------------------------------------------------------------------

ARM_CONFIGS: dict[str, tuple[tuple[int, ...], ...]] = {
    "anchor": ANCHOR_CONFIGS,
    "sensitivity_fp32": SENSITIVITY_CONFIGS,
    "decision": DECISION_CONFIGS,
    "control": CONTROL_CONFIGS,
    "n6_chat_template": DECISION_CONFIGS,
}

#: §6 : bras exécutés par `execute` — N6 (non décisionnel) est conditionné par le
#: drapeau `--n6-chat-template`, dans le plan comme dans l'exécution.
DELEGATED_ARMS: tuple[str, ...] = ("anchor", "sensitivity_fp32")
EXECUTED_ARMS: tuple[str, ...] = ("control", "decision", "n6_chat_template")


def executed_arms(args: argparse.Namespace) -> tuple[str, ...]:
    """Bras réellement exécutés : N6 uniquement si `--n6-chat-template` est présent (§6)."""
    return tuple(arm for arm in EXECUTED_ARMS if arm != "n6_chat_template" or args.n6_chat_template)


def planned_arms(args: argparse.Namespace) -> tuple[str, ...]:
    """Bras du plan : les délégués toujours (§7.3b), les exécutés selon le drapeau N6 (§6)."""
    return tuple(DELEGATED_ARMS) + executed_arms(args)


def arm_dtype(args: argparse.Namespace, arm: str) -> str:
    """dtype effectif d'un bras (§7.3) : choisi par bras, pas par modèle."""
    return {
        "anchor": args.dtype_anchor,
        "sensitivity_fp32": args.dtype_sensitivity,
        "decision": args.dtype_decision,
        "control": args.dtype_control,
        "n6_chat_template": args.dtype_decision,
    }[arm]


def load_scorer_module():
    """Charge le module de scoring par chemin : les harnais vivent hors du paquet."""
    spec = importlib.util.spec_from_file_location(
        "dopd_teacher_scorer_qwen3", _REPO_ROOT / "modules/dopd/harness/teacher_scorer_qwen3.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("module de scoring introuvable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prompt_token_ids(tokenizer, text: str, device: object, *, chat_template: bool):
    """Encodage d'un prompt : base (§1.2) ou variante N6 avec gabarit de conversation (§6)."""
    torch = _torch()
    if chat_template:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False
        )
        return torch.tensor([int(token) for token in encoded], dtype=torch.long, device=device)
    return tokenizer(text, return_tensors="pt").to(device)["input_ids"][0]


def run_arm(
    *,
    arm: str,
    args: argparse.Namespace,
    prompts: list[tuple[str, str]],
    model: object,
    provider: ScoreProvider,
    tokenizer: object,
    device: object,
    budget: BudgetGuard,
    integrity: IntegrityStats,
    replay_stats: ReplayStats,
) -> dict[str, object]:
    """Bras exécuté par J18R : construction économe, arithmétique FP32, boucle figée."""
    observations: list[MarginalObservation] = []
    for prompt_id, text in prompts:
        if arm == "control" and prompt_id not in CONTROL_PROMPTS:
            continue
        prompt_ids = prompt_token_ids(tokenizer, text, device, chat_template=(arm == "n6_chat_template"))
        for config in ARM_CONFIGS[arm]:
            block_size, vocab_width = int(config[0]), int(config[1])
            k = int(config[2]) if len(config) > 2 else None
            assert_support_build_fits(
                block_size, vocab_width, VOCAB_SIZE_QWEN3, economical=True, max_rss_gib=args.max_rss_gib
            )
            support = build_economical_support(
                model_forward(model, prompt_ids, device, arithmetic_dtype="float64"),
                block_size,
                vocab_width,
                option="A",
                arithmetic_dtype="float64",
            )
            sequences, log_joint = support.enumerate_joint()
            observations.extend(
                process_configuration(
                    arm=arm,
                    prompt=prompt_id,
                    config_index=f"N{block_size}_V{vocab_width}",
                    support=support,
                    sequences=sequences,
                    log_joint=log_joint,
                    provider=provider,
                    seed=args.seed,
                    blocks=len(CONTROL_BLOCKS) if arm == "control" else args.blocks,
                    k=k,
                    integrity=integrity,
                    replay_stats=replay_stats,
                    arithmetic_dtype=arm_dtype(args, arm),
                    block_sample=CONTROL_BLOCKS if arm == "control" else None,
                    future_is_state=(arm == "control"),
                    budget=budget,
                )
            )
    replay_stats.n_jobs_expected += len(observations)
    records = [observation.record() for observation in observations]
    return {
        "arm": arm,
        "records": records,
        "publications": arm_publications(observations, arm=arm, bootstrap_seed=args.bootstrap_seed),
        "complete": bool(records),
    }


def control_metrics(records: list[dict]) -> dict[str, object]:
    """§1.5 : les 18 valeurs publiées du contrôle négatif (3 prompts x 3 blocs x 2 configs)."""
    future = [float(row["future_kl"]) for row in records]
    expected = len(CONTROL_PROMPTS) * len(CONTROL_BLOCKS) * len(CONTROL_CONFIGS)
    return {
        "n": len(records),
        "expected": expected,
        "max_future_kl": float(max(future)) if future else float("nan"),
        "median_future_kl": _median(future),
        "threshold_max": CONTROL_MAX_FUTURE_KL,
        "threshold_median": CONTROL_MEDIAN_FUTURE_KL,
        "holds": bool(
            len(records) == expected
            and future
            and max(future) <= CONTROL_MAX_FUTURE_KL
            and _median(future) <= CONTROL_MEDIAN_FUTURE_KL
        ),
    }


def expected_blocks_per_config(arm: str, n_prompts: int, blocks: int) -> int:
    """Effectif attendu d'un bras (§3.2) : le contrôle est l'échantillon déterministe
    figé §1.5 (3 prompts x 3 blocs), il ne dépend pas de la liste complète des prompts."""
    if arm == "control":
        return len(CONTROL_PROMPTS) * len(CONTROL_BLOCKS)
    return int(n_prompts) * int(blocks)


def criterion_c4(publications: Mapping[str, Mapping[str, object]], blocks: int, n_prompts: int) -> dict[str, object]:
    """C4 : effectifs par configuration et blocs de base (§3.2, §4.4)."""
    per_config: dict[str, object] = {}
    holds = True
    for arm, values in publications.items():
        for config, counts in (values.get("block_counts") or {}).items():
            expected_blocks = expected_blocks_per_config(arm, n_prompts, blocks)
            ok = int(counts["n_blocks_sampled"]) == expected_blocks and int(counts["n_distinct_block_indices"]) <= (
                len(CONTROL_BLOCKS) if arm == "control" else blocks)
            if arm in ("decision", "n6_chat_template"):
                ok = ok and int(counts["n_marginals"]) >= MIN_MARGINALS
            key = f"{arm}/{config}"
            per_config[key] = {
                "n_marginals": int(counts["n_marginals"]),
                "n_blocks_sampled": int(counts["n_blocks_sampled"]),
                "n_distinct_block_indices": int(counts["n_distinct_block_indices"]),
                "expected_blocks_sampled": expected_blocks,
                "holds": bool(ok),
            }
            holds = holds and ok
    return {
        "holds": bool(holds),
        "min_marginals_per_config": MIN_MARGINALS,
        "blocks_per_config": blocks,
        "n_prompts": int(n_prompts),
        "per_config": per_config,
    }


def criterion_c5(
    integrity: IntegrityStats,
    replay_stats: ReplayStats,
    control: Mapping[str, object],
) -> dict[str, object]:
    """C5 : intégrité numérique (finitude, sommes, rejeu, float32 appliqué, contrôle)."""
    return {
        "holds": bool(
            integrity.finite
            and integrity.max_sum_error <= SUM_ERROR_TOLERANCE
            and replay_stats.holds()
            and integrity.float32_applied
            and control.get("holds")
            and integrity.enumeration_agreement_max <= SUM_ERROR_TOLERANCE
        ),
        "clause_a_finite": bool(integrity.finite),
        "clause_b_sum_error": {
            "max": float(integrity.max_sum_error),
            "tolerance": SUM_ERROR_TOLERANCE,
            "holds": bool(integrity.max_sum_error <= SUM_ERROR_TOLERANCE),
        },
        "clause_c_replay": dict(replay_stats.to_dict(), holds=bool(replay_stats.holds())),
        "clause_d_float32_applied": {
            "targets_seen": int(integrity.float32_targets_seen),
            "mismatches": int(integrity.float32_mismatches),
            "holds": bool(integrity.float32_applied),
        },
        "clause_e_control_and_enumeration": {
            "control": dict(control),
            "enumeration_agreement_max": float(integrity.enumeration_agreement_max),
            "holds": bool(bool(control.get("holds")) and integrity.enumeration_agreement_max <= SUM_ERROR_TOLERANCE),
        },
    }


def criterion_c7(model: object, counter: BackwardCounter) -> dict[str, object]:
    """C7 : professeur gelé — aucun backward, aucun poids modifié, `requires_grad` faux."""
    requires_grad_all_false = all(not bool(parameter.requires_grad) for parameter in model.parameters())
    training_mode = bool(getattr(model, "training", False))
    return {
        "holds": bool(counter.count == 0 and requires_grad_all_false and not training_mode),
        "backward_count": int(counter.count),
        "requires_grad_all_false": bool(requires_grad_all_false),
        "model_training_mode": training_mode,
    }


def versions_block() -> dict[str, object]:
    """§7.4 : versions consignées ; `torch_version` et `transformers_version` distinctes."""
    torch = _torch()
    transformers = _transformers()
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "git_commit": git_commit(),
    }

def observations_from_records(records: list[dict]) -> list[MarginalObservation]:
    """Réhydrate des observations à partir d'un shard brut (bras délégués §7.3b)."""
    observations = []
    for row in records:
        observations.append(
            MarginalObservation(
                arm=str(row["arm"]),
                prompt=str(row["prompt"]),
                config_index=row["config_index"],
                block_index=int(row["block_index"]),
                position=int(row["position"]),
                noise=float(row["noise"]),
                visible_future=bool(row["visible_future"]),
                causal_kl=float(row["causal_kl"]),
                future_kl=float(row["future_kl"]),
                delta_kl=float(row["delta_kl"]),
            )
        )
    return observations


def criterion_c1(
    records: list[dict],
    publications: Mapping[str, object],
    *,
    bootstrap_seed: int,
    iterations: int,
) -> dict[str, object]:
    """C1 : agrégat (tous σ) > 0 avec IC 95 % excluant zéro ; par σ = publication (§4.2)."""
    aggregate = bootstrap_delta_ci(records, bootstrap_seed, iterations, group_fields=GROUP_FIELDS)
    mean = summarize_marginals(records)["mean_delta_kl"]
    by_noise = {key: value for key, value in (publications.get("c1_by_noise") or {}).items() if key != "sigma_positive_only"}
    sigma_positive = (publications.get("c1_by_noise") or {}).get("sigma_positive_only") or {}
    return {
        # §4.2 et PRE-B5 : C1 est le critère agrégé ; `per_noise` est une publication
        # obligatoire et `C1_sigma_positive` un critère distinct. Aucun durcissement.
        "holds": bool(mean is not None and mean > 0.0 and aggregate[0] > 0.0),
        "aggregate": {"mean_delta_kl": mean, "ci95": [float(aggregate[0]), float(aggregate[1])]},
        "per_noise": {key: dict(value) for key, value in sorted(by_noise.items())},
        "sigma_positive_only": dict(sigma_positive),
    }


def execute(args: argparse.Namespace, argv: Sequence[str]) -> dict[str, object]:
    """Run complet : garde-fou avant chargement, bras, critères, décision, shards, sidecar."""
    torch = _torch()
    transformers = _transformers()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # §7.7 : vérification d'appartenance au dépôt AVANT tout chargement (C8).
    guard = verify_model_source_in_repo(args.model, _REPO_ROOT)
    scorer_module = load_scorer_module()
    model, source = scorer_module.load_teacher(args.model, device=args.device, dtype=args.dtype_decision)
    tokenizer = transformers.AutoTokenizer.from_pretrained(source)
    provider = ProductionScoreProvider(scorer_module.Qwen3TeacherScorer(model, tokenizer))
    counter = BackwardCounter()
    integrity = IntegrityStats()
    fraction = float(args.replay_fraction)
    replay_stats = ReplayStats(
        tolerance=float(args.replay_tolerance),
        modulo=int(round(1.0 / fraction)) if fraction > 0 else REPLAY_MODULO,
    )

    def gpu_peak_bytes() -> float:
        return float(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0.0

    budget = BudgetGuard(
        max_seconds=args.max_seconds,
        max_gpu_mib=args.max_gpu_mib,
        max_rss_gib=args.max_rss_gib,
        gpu_peak_bytes=gpu_peak_bytes,
    )
    n6_budget = BudgetGuard(
        max_seconds=args.n6_max_seconds,
        max_gpu_mib=args.max_gpu_mib,
        max_rss_gib=args.max_rss_gib,
        gpu_peak_bytes=gpu_peak_bytes,
    )
    prompts = load_prompt_set(_REPO_ROOT / args.prompt_file)
    arms: dict[str, dict[str, object]] = {}
    produced: dict[str, str] = {}

    # §7.3b : ancrage et sensibilité rendus par le harnais J15 gelé, ingérés tels quels.
    for arm, invocation in (("anchor", ANCHOR_INVOCATION), ("sensitivity_fp32", SENSITIVITY_INVOCATION)):
        returncode = run_arm_subprocess(invocation, _REPO_ROOT)
        raw_path = output_dir / ARM_SHARDS[arm].replace(".json", "_raw.json")
        records = ingest_arm_shard(raw_path, arm)
        publications = arm_publications(observations_from_records(records), arm=arm, bootstrap_seed=args.anchor_bootstrap_seed)
        arms[arm] = {
            "arm": arm,
            "invocation": invocation,
            "returncode": int(returncode),
            "records": records,
            "publications": publications,
            "complete": bool(records) and returncode == 0,
        }
        produced[ARM_SHARDS[arm]] = write_json(output_dir / ARM_SHARDS[arm], arms[arm])

    for arm in executed_arms(args):
        result = run_arm(
            arm=arm,
            args=args,
            prompts=prompts,
            model=model,
            provider=provider,
            tokenizer=tokenizer,
            device=args.device,
            budget=n6_budget if arm == "n6_chat_template" else budget,
            integrity=integrity,
            replay_stats=replay_stats,
        )
        arms[arm] = result
        produced[ARM_SHARDS[arm]] = write_json(output_dir / ARM_SHARDS[arm], result)

    control = control_metrics(arms["control"]["records"])
    j15_reference = json.loads((_REPO_ROOT / "modules/dopd/eval/annexe_d_qwen3_0.6b_multiprompt/results.json").read_text(encoding="utf-8"))
    c1 = criterion_c1(arms["decision"]["records"], arms["decision"]["publications"], bootstrap_seed=args.bootstrap_seed, iterations=args.bootstrap_iterations)
    reserves = evaluate_reserves(arms["decision"]["publications"])
    criteria_shell = {
        "C4": criterion_c4({arm: arms[arm]["publications"] for arm in arms}, args.blocks, len(prompts)),
        "C5": criterion_c5(integrity, replay_stats, control),
        "C6": {"holds": reconstruct_command([token for token in argv if token != ACCEPTANCE_FLAG]) == CANONICAL_COMMAND},
        "C7": criterion_c7(model, counter),
        "C8": {"holds": bool(guard.get("model_source_in_repo"))},
        "C9": criterion_c9(arms["anchor"]["records"], j15_reference),
        "C10": criterion_c10(_REPO_ROOT),
    }
    flags = {key: bool(value["holds"]) for key, value in criteria_shell.items()}
    flags["C1"] = bool(c1["holds"])
    flags["C1_sigma_positive"] = bool((c1.get("sigma_positive_only") or {}).get("holds"))
    flags["arms_complete"] = all(bool(arms[arm]["complete"]) for arm in REQUIRED_ARMS)
    decision = decide_j18r_issue(flags, reserves)

    runtime = {
        "elapsed_seconds": float(budget.check()["elapsed_seconds"]),
        "budget_checks": int(budget.checks),
        "rss_gib_at_end": float(current_rss_bytes() / 1024**3),
        "gpu_peak_mib": float(gpu_peak_bytes() / 1024**2),
        "device": args.device,
        "dtypes": {
            "anchor": args.dtype_anchor,
            "decision": args.dtype_decision,
            "control": args.dtype_control,
            "sensitivity": args.dtype_sensitivity,
        },
        "versions": versions_block(),
    }
    results = {
        "kind": "j18r_decision",
        "protocol": {
            "seed": args.seed,
            "bootstrap_seed": args.bootstrap_seed,
            "anchor_seed": args.anchor_seed,
            "anchor_bootstrap_seed": args.anchor_bootstrap_seed,
            "iterations": args.bootstrap_iterations,
            "group_fields": list(GROUP_FIELDS),
            "noise_levels": list(NOISE_LEVELS),
            "mask_retention": MASK_RETAIN_PROBABILITY,
            "configs": {arm: [list(config) for config in configs] for arm, configs in ARM_CONFIGS.items()},
            "replay_fraction": fraction,
            "replay_modulo": replay_stats.modulo,
            "replay_tolerance": replay_stats.tolerance,
            "canonical_command": CANONICAL_COMMAND,
        },
        "runtime": runtime,
        "numerical_integrity": dict(integrity.to_dict(), replay=replay_stats.to_dict()),
        "criteria": {
            "C1": c1,
            **{key: value for key, value in criteria_shell.items()},
            "arms_complete": flags["arms_complete"],
        },
        "reserves": reserves,
        "decision": decision,
        "arms": {arm: arms[arm]["publications"] for arm in arms},
        "training": counter.training_block(criterion_c7(model, counter)["requires_grad_all_false"]),
        "reference_j15": REFERENCE_J15,
    }
    if not flags["arms_complete"]:
        # §7.2 : `results.json` n'est jamais écrit si un bras requis est incomplet.
        return {"shards": produced, "results_written": False, "decision": decision}
    results_hash = write_json(output_dir / "results.json", results)
    produced["results.json"] = results_hash
    manifest = _ManifestBuilder().build_manifest(
        artifact=ArtifactRef(
            path=str(Path(args.output_dir) / "results.json"), kind="eval", hashes=dict(produced)
        ),
        parent=ArtifactRef(
            path="modules/dopd/eval/annexe_d_qwen3_0.6b_multiprompt/results.json",
            kind="eval",
            hashes={"results.json": FROZEN_HASHES["modules/dopd/eval/annexe_d_qwen3_0.6b_multiprompt/results.json"]},
        ),
        config=LineageWriter.config_block(DOPDConfig(), seed=args.seed, bootstrap_seed=args.bootstrap_seed),
        data={"prompts_file": args.prompt_file, "n_prompts": len(prompts), "model": args.model},
        verification={"criteria": results["criteria"], "decision": decision},
        command=reconstruct_command([token for token in argv if token != ACCEPTANCE_FLAG]),
    )
    sidecar = output_dir / "lineage.json"
    sidecar.write_text(_ManifestBuilder().write(manifest), encoding="utf-8")
    produced["lineage.json"] = sha256_file(str(sidecar))
    return {"shards": produced, "results_written": True, "decision": decision, "manifest": manifest.to_dict()}


ACCEPTANCE_FLAG = "--confirm-accepted"



# ---------------------------------------------------------------------------
# CLI et plan (§7.3, rejouabilité testée en phase 1)
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Harnais de décision J18R (PRE figé) — mesure reconstruite de l'effet "
            "future-aware sur Qwen3-0.6B."
        )
    )
    parser.add_argument("--model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype-anchor", default="bfloat16")
    parser.add_argument("--dtype-decision", default="float32")
    parser.add_argument("--dtype-control", default="float32")
    parser.add_argument("--dtype-sensitivity", default="float32")
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--blocks", type=int, default=BLOCKS_PER_PROMPT_CONFIG)
    parser.add_argument("--seed", type=int, default=SEED_DECISION)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED_DECISION)
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    parser.add_argument("--anchor-seed", type=int, default=SEED_ANCHOR)
    parser.add_argument("--anchor-bootstrap-seed", type=int, default=BOOTSTRAP_SEED_ANCHOR)
    parser.add_argument("--replay-fraction", type=float, default=REPLAY_FRACTION)
    parser.add_argument("--replay-tolerance", type=float, default=REPLAY_TOLERANCE)
    parser.add_argument("--n6-chat-template", action="store_true")
    parser.add_argument("--n6-max-seconds", type=float, default=N6_MAX_SECONDS)
    parser.add_argument("--max-seconds", type=float, default=MAX_GPU_SECONDS)
    parser.add_argument("--max-gpu-mib", type=float, default=MAX_GPU_MIB)
    parser.add_argument("--max-rss-gib", type=float, default=MAX_RSS_GIB)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="affiche le plan (bras, configurations, mémoire estimée) sans torch ni GPU",
    )
    parser.add_argument(
        "--confirm-accepted",
        action="store_true",
        help="acceptation écrite du PRE (§11) ; sans elle le run est refusé",
    )
    return parser


def reconstruct_command(argv: Sequence[str]) -> str:
    """Commande réellement exécutée, reconstruite pour la comparaison C6."""
    return " ".join(["python modules/dopd/harness/run_j18r_decision.py", *argv])


def dry_run_plan(argv: Sequence[str]) -> dict[str, object]:
    """Plan exécutable sans torch : bras, configurations, mémoire estimée (§1.1, §7.8)."""
    args = build_parser().parse_args(argv)
    arms: dict[str, object] = {}
    for arm, configs in ((arm, ARM_CONFIGS[arm]) for arm in planned_arms(args)):
        estimates = {}
        for config in configs:
            block_size, vocab_width = config[0], config[1]
            economical = arm in ("decision", "control", "n6_chat_template")
            estimates[f"N{block_size}_V{vocab_width}"] = support_build_peak_bytes(
                block_size, vocab_width, VOCAB_SIZE_QWEN3, economical=economical
            )
        arms[arm] = {
            "configs": [list(config) for config in configs],
            "noise_levels": list(NOISE_LEVELS),
            "shard": ARM_SHARDS[arm],
            "memory_estimates": estimates,
        }
    return {
        "command_executed": reconstruct_command(list(argv)),
        "canonical_command": CANONICAL_COMMAND,
        "canonical_match": reconstruct_command(list(argv)) == CANONICAL_COMMAND,
        "arms": arms,
        "budgets": {
            "max_seconds": args.max_seconds,
            "n6_max_seconds": args.n6_max_seconds,
            "max_gpu_mib": args.max_gpu_mib,
            "max_rss_gib": args.max_rss_gib,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run_plan(argv), ensure_ascii=False, indent=2))
        return 0
    # §11 du PRE : la phase 2 (run GPU) n'est déclenchée qu'après acceptation écrite.
    if not args.confirm_accepted:
        raise SystemExit(
            "J18R phase 2 non autorisée sans acceptation écrite du PRE (§11) : la commande "
            "canonique §7.3 est rejouable, --dry-run affiche le plan sans torch ni GPU."
        )
    outcome = execute(args, argv)
    print(json.dumps(outcome, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
