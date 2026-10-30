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
import contextlib
import errno
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import resource
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
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
#: configurations x 1 niveau de perturbation = 18 marginaux.
CONTROL_PROMPTS: tuple[str, ...] = ("p01", "p02", "p03")
CONTROL_BLOCKS: tuple[int, ...] = (0, 1, 2)
# R-J18R-17 (constat D7) : la grille §2.5 donne `σ = 0` **au seul** bras de controle
# (« controle | (4,12), (4,24) | 0 »). Les seuils C5(e) sont des seuils d'**identite**
# (~ 4 ordres au-dessus du residu d'identite, ~ 5 ordres sous le plus petit effet reel
# attende, ~ 0,08) : faire tourner le controle sur `NOISE_LEVELS` mesurerait une
# geometrie perturbee (σ = 0,5/1,0/1,5) contre des seuils declares pour σ = 0.
CONTROL_NOISE_LEVELS: tuple[float, ...] = (0.0,)
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

#: J18R-final — seuil **dur** du watchdog VRAM, exprimé en **VRAM totale utilisée**
#: (`nvidia-smi memory.used`, toutes applications confondues).
#:
#: Dérivation (chiffrée, J18R-final §2) :
#:
#: * capacité physique mesurée : 32 607 MiB totaux, baseline non-mission mesurée
#:   1 090 MiB (Xwayland) ⇒ **31 517 MiB** utilisables ;
#: * pic J18R observé au vol d'essai §12 : **11 691 MiB** utilisés (total GPU) ;
#: * plus grand pic **alloué** mesuré dans `BUDGET_RTX5090.md` §1.4 pour un job de
#:   scoring professeur : **18 835 MiB** (k = 64, contexte 2 048) ;
#: * seuil retenu : **28 672 MiB** (28 GiB) utilisés au total, soit 91 % des
#:   31 517 MiB utilisables — **2 845 MiB** sous l'épuisement physique — et
#:   2,45 × le pic J18R observé, 1,52 × le plus grand pic alloué mesuré.
#:
#: Portée : le seuil porte sur la **VRAM totale** (mission + applications tierces).
#: Il est distinct du plafond PRE `--max-gpu-mib` (16 384 MiB), qui porte sur
#: l'**allocation torch** du processus du harnais (`torch.cuda.max_memory_allocated`).
VRAM_WATCHDOG_MAX_USED_MIB = 28_672
VRAM_WATCHDOG_SAMPLE_SECONDS = 1.0
VRAM_WATCHDOG_INSTRUMENT = "nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits"
VRAM_WATCHDOG_DERIVATION = (
    "seuil dur = 28 672 MiB de VRAM totale utilisée ; capacité mesurée 32 607 MiB "
    "moins baseline non-mission 1 090 MiB = 31 517 MiB utilisables ; seuil = 91 % "
    "des utilisables (2 845 MiB sous l'épuisement) ; pic J18R observé 11 691 MiB ; "
    "plus grand pic alloué mesuré (BUDGET_RTX5090.md §1.4) 18 835 MiB"
)

#: J18R-final — plafond **RLIMIT_AS** des sous-processus délégués §7.3b.
#:
#: RLIMIT_AS porte sur l'**espace d'adressage virtuel** (CUDA réserve de grandes
#: plages virtuelles) : il n'est pas un plafond RSS et ne peut pas être calé sur
#: `--max-rss-gib`. La valeur est **mesurée** (J18R-final §3) :
#:
#: * initialisation CUDA + chargement Qwen3-0.6B BF16 + forward trivial : échec à
#:   24 GiB (`AcceleratorError` CUDA), échec à 32 GiB (`mmap` ENOMEM du shard),
#:   **succès à partir de 40 GiB** (`forward_ok`, VmPeak 40 287 348 kB) ;
#: * exécution réelle du harnais J15 délégué (`--blocks 2`, BF16) : **VmPeak
#:   mesuré 109,73 GiB** (RSS de groupe 21,06 GiB) sous une limite de 160 GiB ;
#:   la taille virtuelle franchit déjà 86 GiB à t = 6,9 s (pendant le
#:   chargement) : toute limite sous ~112 GiB tue le child **avant** la mesure.
#:
#: Valeur retenue : **128 GiB**, plus petit multiple de 32 GiB au-dessus du pic
#: virtuel mesuré majoré de 15 % : ceil(109,73 × 1,15 / 32) × 32 = 128 GiB
#: (marge 16,6 % sur le pic BF16 mesuré).
#:
#: **Plage 48–64 GiB écartée** : elle est incompatible avec le `VmPeak` mesuré
#: (109,73 GiB) — le child J15 serait tué par `MemoryError` pendant le
#: chargement, pas protégé. RLIMIT_AS ne protège d'ailleurs **pas** la RAM
#: résidente : la protection RAM est portée par le **watchdog de RSS de groupe**
#: (`DELEGATED_GROUP_RSS_MAX_GIB`) et par la **sérialisation** des mesures.
#:
#: Portée : plafond virtuel du **groupe** délégué (child + descendants). Il ne
#: couvre pas la VRAM (mesurée par le watchdog VRAM) ni la RSS (mesurée par
#: `group_rss_bytes` / `RUSAGE_CHILDREN` / cgroup).
DELEGATED_RLIMIT_AS_GIB = 128
DELEGATED_RLIMIT_AS_DERIVATION = (
    "RLIMIT_AS = 128 GiB, mesuré : VmPeak réel du harnais J15 délégué (--blocks 2, "
    "BF16) = 109,73 GiB (86,14 GiB déjà franchis à t=6,9 s, pendant le chargement) ; "
    "initialisation CUDA + chargement modèle BF16 + forward : échec à 24 GiB "
    "(AcceleratorError) et à 32 GiB (mmap ENOMEM), succès à 40 GiB ; limite = "
    "ceil(109,73 × 1,15 / 32) × 32 = 128 GiB (marge 16,6 %) ; plafond virtuel, "
    "distinct du plafond RSS 24 GiB et du watchdog VRAM ; la plage 48–64 GiB est "
    "écartée (incompatible avec le VmPeak mesuré)"
)

#: J18R-final — plafond **RSS du groupe délégué** (child + descendants), en octets.
#:
#: C'est le garde qui protège réellement la RAM du guest WSL : `RLIMIT_AS` ne
#: borne pas la RSS. La valeur est le plafond RSS **du PRE** (§7.8, 24 GiB),
#: appliqué au **groupe** delegates : le pic mesuré au vol d'essai §12 est de
#: 22 385,8 MiB (FP32) / 21 584,8 MiB (BF16), soit 21,86 / 21,08 GiB — sous le
#: plafond avec 2,14 GiB de marge. Le groupe est tué (SIGTERM puis SIGKILL) au
#: dépassement : `BudgetExceeded.kind = "memory"`.
DELEGATED_GROUP_RSS_MAX_GIB = MAX_RSS_GIB

#: J18R-final — VRAM libre minimale exigée **avant** tout lancement GPU
#: (PRE §7.8 : « si `memory.free` < 17 GiB : STOPPER »). 17 GiB = 17 408 MiB.
VRAM_PREFLIGHT_MIN_FREE_MIB = 17 * 1024

#: J18R-final — fichier de verrou d'**exclusion mutuelle** des mesures GPU.
#: Sous `_artefact/` (règle 9) : jamais versionné, jamais hors du dépôt.
MEASUREMENT_LOCK_RELATIVE_PATH = "_artefact/locks/j18r_measurement.lock"

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

#: §7.2 — shards **étiquetés** par bras ; `results.json` n'est jamais écrit si un bras requis est
#: incomplet. Le bras de sensibilité n'a pas de shard étiqueté : son brut est ingéré tel quel et
#: ses publications vivent dans `results.json` (§4.4) — R-J18R-08.
ARM_SHARDS: dict[str, str] = {
    "control": "arm_control.json",
    "anchor": "arm_anchor.json",
    "decision": "arm_decision.json",
    "n6_chat_template": "arm_n6.json",
}
#: §7.3b — shards **bruts** écrits par les sous-invocations J15. J18R les lit et ne les écrit
#: jamais : les réécrire casserait le hash et la chaîne figée (D2, R-J18R-08).
RAW_SHARDS: dict[str, str] = {
    "anchor": "arm_anchor_raw.json",
    "sensitivity_fp32": "arm_sensitivity_raw.json",
}
REQUIRED_ARMS: tuple[str, ...] = ("control", "anchor", "decision")


def arm_shard_names(arm: str) -> list[str]:
    """Shards §7.2 d'un bras : le brut produit par J15 (bras délégués), puis le shard étiqueté."""
    names = []
    if arm in RAW_SHARDS:
        names.append(RAW_SHARDS[arm])
    if arm in ARM_SHARDS:
        names.append(ARM_SHARDS[arm])
    return names


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

#: §7.3b — invocations d'ancrage, chaînes figées (harnais J15 gelé, importé en lecture seule).
#: Le chemin de sortie suit `--output-dir` (D3, R-J18R-09) : au répertoire canonique, la chaîne
#: recopiée dans le PRE est reproduite **à l'identique**, `--blocks` canonique compris.
J15_HARNESS = "modules/dopd/harness/diagnostic_annexe_d.py"
CANONICAL_BLOCKS = 100


def _j15_invocation(dtype: str, raw_name: str, output_dir: str, blocks: int) -> str:
    tokens = [
        "python",
        J15_HARNESS,
        "--model",
        DEFAULT_TEACHER_MODEL,
        "--device",
        "cuda",
        "--dtype",
        dtype,
        "--prompt-file",
        DEFAULT_PROMPT_FILE,
        "--blocks",
        str(blocks),
        "--seed",
        str(SEED_ANCHOR),
        "--bootstrap-seed",
        str(BOOTSTRAP_SEED_ANCHOR),
        "--local-files-only",
        "--output",
        f"{output_dir}/{raw_name}",
        "--top-k",
        "16",
    ]
    return " ".join(tokens)


def anchor_invocation(output_dir: str = DEFAULT_OUTPUT_DIR, blocks: int = CANONICAL_BLOCKS) -> str:
    """§7.3b — ancrage BF16 : harnais J15 gelé, `--output` propagé à `--output-dir`."""
    return _j15_invocation("bfloat16", RAW_SHARDS["anchor"], output_dir, blocks)


def sensitivity_invocation(output_dir: str = DEFAULT_OUTPUT_DIR, blocks: int = CANONICAL_BLOCKS) -> str:
    """§7.3b — sensibilité FP32 : même chaîne, `--dtype` et nom de sortie seuls écartés."""
    return _j15_invocation("float32", RAW_SHARDS["sensitivity_fp32"], output_dir, blocks)


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


# ---------------------------------------------------------------------------
# §3.5 / §5.2 — compteur de **jobs** d'un bras (R-J18R-17, constat D6)
# ---------------------------------------------------------------------------


@dataclass
class JobCounter:
    """Compteur des **jobs** d'un bras (§3.5), monotone, jamais remis à zéro.

    §3.5 définit le job comme « une demande de scoring d'un marginal » et §5.2
    sélectionne le rejeu sur `job_index mod 20 == 0` — exactement **5 %** des jobs
    du **bras**. Compté par configuration (le comportement d'avant R-J18R-17), le
    compteur repart à 1 à chaque configuration : un seul job est éligible par
    tranche de 20, et sur un bras à effectif faible (le contrôle : 18 jobs par
    configuration, 36 sur ses deux configurations) le taux réel tombe à 2/36 = 5,6 %
    la première fois, **0 %** ensuite — C5(c) « ≥ 5 % des jobs » devient
    inatteignable sur un bras réel. Un compteur créé **au niveau du bras** donne le
    taux réel `floor(n_jobs / 20) / n_jobs ≈ 5 %` sur l'ensemble des jobs émis.
    """

    value: int = 0

    def next_index(self) -> int:
        """Index 1-based du job suivant ; le compteur ne décroît jamais."""
        self.value += 1
        return self.value


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
    job_index: int,
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
        job_index=int(job_index),
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

    if job_index % replay_stats.modulo == 0:
        replayed_prior, replayed_scores = provider.replay_job(prefix, [int(token) for token in tokens], suffix)
        ids = [int(token) for token in tokens]
        delta_prior = np.abs(
            np.asarray(replayed_prior, dtype=np.float64) - np.asarray(prior_full, dtype=np.float64)[ids]
        )
        delta_scores = np.abs(np.asarray(replayed_scores, dtype=np.float64) - np.asarray(scores, dtype=np.float64))
        replay_stats.record_job(job_index, float(max(float(np.max(delta_prior)), float(np.max(delta_scores)))))

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
    # R-J18R-17 : compteur de **jobs du bras** (§3.5) — créé par `run_arm`, partagé
    # entre les configurations, jamais remis à zéro (`None` = compteur local, tests).
    job_counter: JobCounter | None = None,
    arithmetic_dtype: str = "float32",
    block_sample: Sequence[int] | None = None,
    future_is_state: bool = False,
    budget: BudgetGuard | None = None,
    # R-J18R-19 (A1/D5) : engagement **durable** au fil de l'eau, à la frontière
    # des blocs **complets** uniquement ; un bloc inachevé n'est jamais publié.
    observations_out: list[MarginalObservation] | None = None,
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
    # R-J18R-17 : le compteur est **celui du bras**, passé par `run_arm` ; il passe
    # d'une configuration à l'autre sans remise à zéro (§3.5, §5.2).
    job_counter = job_counter if job_counter is not None else JobCounter()
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

            block_observations: list[MarginalObservation] = []
            for position in sorted(masked):
                job_index = job_counter.next_index()
                block_observations.append(
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
                        job_index=job_index,
                        k=k,
                        arithmetic_dtype=arithmetic_dtype,
                    )
                )
            # R-J18R-19 (A1/D5) : le bloc **complet** est engagé sur la liste
            # partagée **avant** le contrôle du budget ; l'arrêt d'un bras laisse
            # les blocs complets déjà validés, jamais un bloc inachevé.
            observations.extend(block_observations)
            if observations_out is not None and observations_out is not observations:
                observations_out.extend(block_observations)
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

    `criteria` porte les booléens `C4`..`C10`, `arms_complete`, `C1` (tous σ),
    `C1_sigma_positive` et `C9_evaluable` : cette dernière clé est `False` quand **aucune**
    ligne n'est partagée avec l'artefact J15 (bras d'ancrage absent). L'écart n'est alors pas
    mesuré : la priorité 1 (`STOP`, dérive du harnais) est réservée à un ancrage mesuré en
    échec ; l'absence de données tombe en `INVALIDE` (R-J18R-10). `STOP` et `INVALIDE` ne sont
    pas des décisions sur l'effet.
    """
    if not criteria.get("C9") and criteria.get("C9_evaluable", True):
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


def canonical_j18r_command(args: argparse.Namespace) -> str:
    """Reconstruction canonique §7.3 à partir des champs comparés (R-J18R-19, A8).

    Le drapeau d'acceptation `--confirm-accepted` n'est **jamais** ajouté,
    **jamais** retiré d'une trace existante : la commande rejouable est la
    configuration canonique, la trace fidèle est `command_executed`.
    """
    parts = [
        "python modules/dopd/harness/run_j18r_decision.py",
        "--model", args.model,
        "--device", args.device,
        "--dtype-anchor", args.dtype_anchor,
        "--dtype-decision", args.dtype_decision,
        "--dtype-control", args.dtype_control,
        "--dtype-sensitivity", args.dtype_sensitivity,
        "--prompt-file", args.prompt_file,
        "--blocks", str(args.blocks),
        "--seed", str(args.seed),
        "--bootstrap-seed", str(args.bootstrap_seed),
        "--bootstrap-iterations", str(args.bootstrap_iterations),
        "--anchor-seed", str(args.anchor_seed),
        "--anchor-bootstrap-seed", str(args.anchor_bootstrap_seed),
        "--replay-fraction", str(args.replay_fraction),
        "--replay-tolerance", str(args.replay_tolerance),
    ]
    if args.n6_chat_template:
        parts.append("--n6-chat-template")
    parts += [
        "--n6-max-seconds", str(args.n6_max_seconds),
        "--max-seconds", str(args.max_seconds),
        "--max-gpu-mib", str(args.max_gpu_mib),
        "--max-rss-gib", str(args.max_rss_gib),
        "--output-dir", args.output_dir,
    ]
    return " ".join(parts)

CANONICAL_FIELDS: tuple[tuple[str, str], ...] = (
    ("model", "--model"),
    ("device", "--device"),
    ("dtype_anchor", "--dtype-anchor"),
    ("dtype_decision", "--dtype-decision"),
    ("dtype_control", "--dtype-control"),
    ("dtype_sensitivity", "--dtype-sensitivity"),
    ("prompt_file", "--prompt-file"),
    ("blocks", "--blocks"),
    ("seed", "--seed"),
    ("bootstrap_seed", "--bootstrap-seed"),
    ("bootstrap_iterations", "--bootstrap-iterations"),
    ("anchor_seed", "--anchor-seed"),
    ("anchor_bootstrap_seed", "--anchor-bootstrap-seed"),
    ("replay_fraction", "--replay-fraction"),
    ("replay_tolerance", "--replay-tolerance"),
    ("n6_chat_template", "--n6-chat-template"),
    ("n6_max_seconds", "--n6-max-seconds"),
    ("max_seconds", "--max-seconds"),
    ("max_gpu_mib", "--max-gpu-mib"),
    ("max_rss_gib", "--max-rss-gib"),
    ("output_dir", "--output-dir"),
)


def criterion_c6(argv: Sequence[str], args: argparse.Namespace) -> dict[str, object]:
    """C6 (R-J18R-19, A8) : **configuration complète** champ à champ, pas la chaîne seule.

    Comparés : modèle, device, dtype par bras, fichier de prompts, effectifs,
    graines, rejeu, plafonds (global, N6, GPU, RSS), drapeau N6, sortie. Puis la
    chaîne **sans** le drapeau d'acceptation. Propriété double **déclarée** :

    - commande **rejouable** : `canonical_j18r_command(args)` — la reconstruction
      canonique à partir des champs comparés, drapeau **jamais** ajouté ;
    - trace **fidèle** : `command_executed` — la commande réellement exécutée,
      **avec** `--confirm-accepted` (§11). Le drapeau n'est **jamais** supprimé
      silencieusement d'une trace.

    Écart consigné au rapport (J18R.md §15, R-J18R-19) : la commande canonique
    figée §7.3 **ne porte pas** `--confirm-accepted` ; la commande exécutée le
    porte. `holds` **n'est pas** acquis par convention : tout champ en écart
    le rend faux.
    """
    canonical_args = build_parser().parse_args(shlex.split(CANONICAL_COMMAND)[2:])
    fields = {}
    for name, flag in CANONICAL_FIELDS:
        executed, canonical = getattr(args, name), getattr(canonical_args, name)
        fields[name] = {"flag": flag, "executed": executed, "canonical": canonical, "holds": bool(executed == canonical)}
    stripped = [token for token in argv if token != ACCEPTANCE_FLAG]
    return {
        "holds": all(bool(field["holds"]) for field in fields.values())
        and reconstruct_command(stripped) == CANONICAL_COMMAND,
        "fields": fields,
        "command_executed": reconstruct_command(list(argv)),
        "canonical_reconstruction": canonical_j18r_command(args),
        "canonical_command": CANONICAL_COMMAND,
        "string_match_without_acceptance_flag": bool(reconstruct_command(stripped) == CANONICAL_COMMAND),
        "deviation_declared": {
            "flag": ACCEPTANCE_FLAG,
            "note": (
                "la commande exécutée porte l'acceptation écrite §11 ; la commande "
                "canonique figée §7.3 ne la porte pas ; écart consigné au rapport "
                "(J18R.md §15, R-J18R-19), non supprimé de la trace"
            ),
        },
    }


def criterion_c9(
    anchor_records: list[dict],
    j15_reference: Mapping[str, object],
    *,
    blocks: int = CANONICAL_BLOCKS,
    anchor_executed: bool = True,
    n_prompts: int | None = None,
) -> dict[str, object]:
    """Ancrage C9 : clause (a) écart moyen ≤ 5e-3 ; clause (b) couverture et 2 %.

    `evaluable` distingue les états (R-J18R-14, R-J18R-19 A4, PRE §4.5 / §4.1) :

    - **écart mesuré** (ancrage exécuté, **effectif réellement présent** —
      prompts × configs × indices de blocs distincts `0..blocks-1` — et lignes
      partagées avec J15) : l'échec des clauses donne **STOP** (dérive) ;
    - **défaillance d'exécution** de l'ancrage (bras absent, sous-processus en
      échec) : l'écart n'est **pas mesuré** → **INVALIDE** (R-J18R-10) ;
    - **effectif insuffisant** : le drapeau `--blocks` **n'est pas** l'effectif
      réellement présent (R-J18R-19 A4). Un shard d'ancrage **tronqué** rend C9
      **non évaluable** : la clause (b) n'interprète **pas** une troncature comme
      une dérive. C4 échoue sur les effectifs, la priorité INVALIDE s'applique.

    Ordre de vérification figé : (1) exécution ; (2) effectifs réellement
    présents — completeness, indices distincts, clés attendues ; (3) échantillon
    conforme → clauses C9 ; (4) STOP **uniquement** sur échantillon conforme.
    `STOP` > `INVALIDE` ne vaut que si **les deux anomalies sont établies**
    (C9 mesuré en échec **et** C4 en échec).
    """
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
    # R-J18R-19 (A4) : l'effectif qui sert à la comparaison est celui des
    # **enregistrements présents**, pas le drapeau `--blocks`.
    observed_pairs: dict[tuple[str, str], set[int]] = {}
    for row in anchor_records:
        observed_pairs.setdefault((str(row["prompt"]), str(row["config_index"])), set()).add(int(row["block_index"]))
    expected_configs = [f"N{block_size}_V{vocab_width}" for block_size, vocab_width in ANCHOR_CONFIGS]
    if n_prompts is None:
        # comparaison d'unité sur un jeu minimal de lignes : aucun inventaire complet exigé
        required_pairs: set[tuple[str, str]] = set()
    else:
        required_pairs = {(f"p{index:02d}", config) for index in range(1, int(n_prompts) + 1) for config in expected_configs}
    missing_pairs = sorted(required_pairs - set(observed_pairs))
    incomplete_pairs = sorted(
        key for key in required_pairs if key in observed_pairs and observed_pairs[key] != set(range(int(blocks)))
    )
    sample_complete = not missing_pairs and not incomplete_pairs
    sample_conform = int(blocks) >= CANONICAL_BLOCKS and sample_complete
    if not anchor_executed:
        evaluable_reason = "anchor_not_executed"
    elif not shared:
        evaluable_reason = "no_shared_keys"
    elif int(blocks) < CANONICAL_BLOCKS:
        evaluable_reason = "insufficient_blocks"
    elif not sample_complete:
        evaluable_reason = "incomplete_anchor_sample"
    else:
        evaluable_reason = "measured"
    publications = {
        "evaluable": bool(shared and anchor_executed and sample_conform),
        "evaluable_reason": evaluable_reason,
        "blocks_measured": int(blocks),
        "blocks_required": CANONICAL_BLOCKS,
        # R-J18R-19 (A4) : l'inventaire réellement présent est publié, pas le drapeau.
        "n_prompts_expected": (None if n_prompts is None else int(n_prompts)),
        "n_pairs_present": len(observed_pairs),
        "sample_complete": bool(sample_complete),
        "missing_pairs": [list(pair) for pair in missing_pairs],
        "incomplete_pairs": [list(pair) for pair in incomplete_pairs],
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


def child_rss_peak_bytes() -> int:
    """Pic RSS des **enfants attendus** par ce processus (`RUSAGE_CHILDREN`), en octets.

    Le plafond `--max-rss-gib` est appliqué à `current_rss_bytes()` = RSS du
    **processus du harnais**. Les invocations d'ancrage §7.3b sont des
    **sous-processus** : leur mémoire (≈ 35 Go pour le chemin J15 non économe à
    `V = 24`, chiffré §1.1) n'entre **pas** dans cette mesure.
    `RUSAGE_CHILDREN.ru_maxrss` est le pic du **plus gros enfant déjà attendu**
    (`subprocess.run` attend) ; il ne couvre ni les descendants de ces enfants, ni
    la mémoire GPU de ces enfants. 0 = aucun enfant terminé mesuré.
    """
    return int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) * 1024


def cgroup_memory_peak_bytes() -> int | None:
    """Pic mémoire du **cgroup** (`memory.peak` v2, `max_usage_in_bytes` v1) ; None si absent.

    Seule vue **transverse** (harnais + sous-processus du même cgroup). Absente sous
    cgroup v1 sans le fichier, absente sur macOS.
    """
    for relative in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            text = Path(relative).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text.isdigit():
            return int(text)
    return None


def delegated_memory_observation(group: Mapping[str, object] | None = None) -> dict[str, object]:
    """Mesure **réellement disponible** après une invocation déléguée §7.3b (constat D9,
    R-J18R-18, R-J18R-19).

    R-J18R-19 (D9) : le sous-processus vit dans le **groupe** du harnais
    (`start_new_session=True`) et la limite RLIMIT_AS est posée **avant** `exec`.
    L'observation porte la limite appliquée, le plafond de **temps** de ce
    sous-processus, le signal de stop envoyé et la vérification du groupe **après**
    le stop. Le **pic v1** (`memory.max_usage_in_bytes`) est mesuré **pendant** que
    le sous-processus est vivant ; `memory.peak` v2 n'est **pas** présenté comme v1.
    Les descendants de ces sous-processus et leur mémoire GPU restent explicitement
    non couverts.
    """
    group = group or {}
    children = child_rss_peak_bytes()
    recorded = group.get("cgroup_memory_peak_bytes")
    cgroup = float(recorded) if recorded is not None else cgroup_memory_peak_bytes()
    observed = bool(children > 0 or cgroup is not None)
    instruments: list[str] = []
    if children > 0:
        instruments.append("RUSAGE_CHILDREN")
    if cgroup is not None:
        instruments.append("cgroup memory.peak")
    if bool(group.get("limit_applied", False)):
        instruments.append("RLIMIT_AS")
    return {
        "max_rss_children_bytes": int(children),
        "cgroup_memory_peak_bytes": (None if cgroup is None else int(cgroup)),
        "observed": bool(observed),
        "measured_by": instruments,
        "group_members_covered": bool(group.get("limit_applied", False)),
        "limit_mib": (None if group.get("limit_mib") is None else int(group["limit_mib"])),
        "stop_after_seconds": (
            None if group.get("stop_after_seconds") is None else float(group["stop_after_seconds"])
        ),
        "group_stop_signal": group.get("stop_signal"),
        # J18R-final : **raison** de l'arrêt (temps / VRAM / RSS de groupe) et
        # **fait mesuré** associé.
        "group_stop_reason": group.get("stop_reason"),
        "group_stop_event": group.get("stop_event"),
        "group_verified_after_stop": bool(group.get("verified_after_stop", False)),
        # J18R-final : pic RSS **du groupe délégué** (child + descendants), mesuré
        # à 1 Hz pendant la vie du sous-processus ; c'est le garde qui protège la
        # RAM du guest (RLIMIT_AS ne borne pas la RSS).
        "group_rss_peak_bytes": (
            None if group.get("group_rss_peak_bytes") is None else int(group["group_rss_peak_bytes"])
        ),
        "group_rss_peak_gib": (
            None if group.get("group_rss_peak_gib") is None else float(group["group_rss_peak_gib"])
        ),
        "group_rss_limit_bytes": (
            None if group.get("group_rss_max_bytes") is None else int(group["group_rss_max_bytes"])
        ),
        "vram_watchdog": group.get("vram_watchdog"),
        "note": (
            "pic des enfants **attendus** par le harnais (`subprocess.run`) ; les "
            "descendants de ces sous-processus et leur mémoire GPU ne sont **pas** "
            "couverts par cette mesure"
            if observed
            else "non observé : RUSAGE_CHILDREN = 0 et aucun fichier memory.peak "
            "(cgroup v1 sans max_usage_in_bytes, macOS)"
        ),
    }



class BudgetExceeded(RuntimeError):
    """Dépassement de budget **typé** (R-J18R-14).

    `kind` distingue le plafond **propre à N6** (`"time"`, 1800 s, arrêt du seul
    bras N6 avec publication partielle explicite) des dépassements **globaux**
    (`"rss"`, `"gpu"`, `"gpu_vram"`, `"memory"`), qui restent des arrêts du run
    entier.

    `event` porte le **fait mesuré** du dépassement (watchdog VRAM, RSS de
    groupe) : il est publié dans le manifeste ou dans l'artefact d'abandon, et
    jamais reconstruit après coup.
    """

    def __init__(self, kind: str, message: str, event: Mapping[str, object] | None = None) -> None:
        super().__init__(message)
        self.kind = str(kind)
        self.event: dict[str, object] | None = (
            None if event is None else {str(key): value for key, value in event.items()}
        )


class VramWatchdogUnavailable(RuntimeError):
    """Le watchdog VRAM n'a pas pu mesurer la VRAM (`nvidia-smi` indisponible ou
    illisible). Le harnais **refuse** d'exécuter : pas de supposition de budget."""


class VramPreflightRefused(RuntimeError):
    """VRAM libre insuffisante **avant** le lancement (PRE §7.8) : refus, aucun run."""


class MeasurementLockBusy(RuntimeError):
    """Une autre mesure GPU detient le verrou d'exclusion mutuelle (J18R-final).

    Le second lancement est **refusé** (jamais mis en file) : le protocole
    post-OOM impose une seule exécution lourde à la fois.
    """


class VramWatchdog:
    """Watchdog VRAM J18R-final : VRAM **totale** utilisée, seuil **dur**, arrêt net.

    Instrument : `nvidia-smi --query-gpu=memory.used,memory.free,memory.total`
    (une requête par échantillon, période 1 s). Le seuil porte sur la mémoire
    **utilisée totale** (mission + applications tierces) ; il est **distinct** du
    plafond PRE `--max-gpu-mib`, qui porte sur l'allocation torch du harnais.

    Portée et limites (consignées dans le manifeste) :

    * **observé** : `memory.used` / `memory.free` / `memory.total` du GPU 0 au
      niveau du pilote, échantillonné à 1 s pendant la vie du sous-processus ;
    * **limité** : l'échantillonnage à 1 s peut laisser passer un pic **plus court**
      que la période d'échantillonnage ; l'instrument mesure le **total** GPU et
      n'attribue pas la mémoire à un processus ; un pilote qui cesse de répondre
      fait **échouer** la mesure (refus), jamais passer pour un dépassement.
    """

    def __init__(
        self,
        *,
        max_used_mib: int = VRAM_WATCHDOG_MAX_USED_MIB,
        sample_seconds: float = VRAM_WATCHDOG_SAMPLE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_used_mib = int(max_used_mib)
        self.sample_seconds = float(sample_seconds)
        self._clock = clock
        self._last_sample_at = -1e18
        self.samples = 0
        self.peak_used_mib: int | None = None
        self.min_free_mib: int | None = None
        self.total_mib: int | None = None
        self.last_used_mib: int | None = None
        self.exceeded = False
        self.event: dict[str, object] | None = None

    @staticmethod
    def query() -> dict[str, int]:
        """Lit la VRAM du GPU 0 ; lève `VramWatchdogUnavailable` si la mesure échoue."""
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise VramWatchdogUnavailable(f"mesure VRAM indisponible : {exc}") from exc
        if proc.returncode != 0:
            raise VramWatchdogUnavailable(
                f"mesure VRAM indisponible (nvidia-smi rc={proc.returncode}) : "
                f"{proc.stderr.strip()[:200]}"
            )
        try:
            used_s, free_s, total_s = proc.stdout.strip().splitlines()[0].split(",")
            return {
                "used_mib": int(used_s.strip()),
                "free_mib": int(free_s.strip()),
                "total_mib": int(total_s.strip()),
            }
        except (ValueError, IndexError, AttributeError) as exc:
            raise VramWatchdogUnavailable(
                f"mesure VRAM illisible : {proc.stdout.strip()[:200]}"
            ) from exc

    def sample(self, *, force: bool = False) -> dict[str, int]:
        """Échantillonne la VRAM ; ne l'échantillonne qu'une fois par période."""
        now = self._clock()
        if not force and self.samples and now - self._last_sample_at < self.sample_seconds:
            return {
                "used_mib": int(self.last_used_mib or 0),
                "free_mib": int(self.min_free_mib or 0),
                "total_mib": int(self.total_mib or 0),
            }
        reading = self.query()
        self._last_sample_at = now
        self.samples += 1
        self.last_used_mib = reading["used_mib"]
        self.total_mib = reading["total_mib"]
        self.peak_used_mib = max(int(self.peak_used_mib or 0), reading["used_mib"])
        self.min_free_mib = min(int(self.min_free_mib or reading["free_mib"]), reading["free_mib"])
        if reading["used_mib"] > self.max_used_mib and not self.exceeded:
            self.exceeded = True
            self.event = {
                "reason": "vram_used_above_hard_threshold",
                "threshold_used_mib": int(self.max_used_mib),
                "observed_used_mib": int(reading["used_mib"]),
                "observed_free_mib": int(reading["free_mib"]),
                "observed_total_mib": int(reading["total_mib"]),
                "sample_index": int(self.samples),
                "instrument": VRAM_WATCHDOG_INSTRUMENT,
            }
        return reading

    def to_dict(self) -> dict[str, object]:
        """Trace du watchdog, publiée dans le manifeste."""
        return {
            "instrument": VRAM_WATCHDOG_INSTRUMENT,
            "threshold_used_mib": int(self.max_used_mib),
            "threshold_derivation": VRAM_WATCHDOG_DERIVATION,
            "sample_seconds": float(self.sample_seconds),
            "samples": int(self.samples),
            "peak_used_mib": (None if self.peak_used_mib is None else int(self.peak_used_mib)),
            "min_free_mib": (None if self.min_free_mib is None else int(self.min_free_mib)),
            "total_mib": (None if self.total_mib is None else int(self.total_mib)),
            "exceeded": bool(self.exceeded),
            "event": self.event,
            "scope": (
                "VRAM totale du GPU 0 (mission + applications tierces), échantillonnée "
                "à 1 s via nvidia-smi ; ne couvre pas les pics plus courts que la période "
                "d'échantillonnage et n'attribue pas la mémoire à un processus"
            ),
        }


class BudgetGuard:
    """Garde-budgets §7.8 : temps mural, pic GPU, RSS contrôlée à 1 Hz.

    R-J18R-14 : le plafond **propre de N6** (1800 s) produit une exception typée
    `"time"` — arrêt du seul bras N6, publication partielle explicite. Les
    plafonds **globaux** (RSS, VRAM) produisent `"rss"` / `"gpu"` : ils restent
    des arrêts du run entier. Le plafond global de temps (4 h, PRE §7.8) est
    « hors N6 » : il est porté par le garde global, jamais par celui de N6.
    """

    def __init__(
        self,
        *,
        max_seconds: float = MAX_GPU_SECONDS,
        max_gpu_mib: float = MAX_GPU_MIB,
        max_rss_gib: float = MAX_RSS_GIB,
        clock: Callable[[], float] = time.monotonic,
        rss: Callable[[], float] = current_rss_bytes,
        gpu_peak_bytes: Callable[[], float] = lambda: 0.0,
        time_kind: str = "time",
        vram_watchdog: VramWatchdog | None = None,
    ) -> None:
        self.time_kind = str(time_kind)
        self.max_seconds = float(max_seconds)
        self.max_gpu_mib = float(max_gpu_mib)
        self.max_rss_gib = float(max_rss_gib)
        self._clock = clock
        self._rss = rss
        self._gpu_peak_bytes = gpu_peak_bytes
        self.vram_watchdog = vram_watchdog
        self.started = clock()
        self.last_check = 0.0
        self.checks = 0
        self.rss_peak_bytes = 0.0
        self.paused = False
        self.paused_at = 0.0

    def pause(self) -> None:
        """Fige le temps écoulé (le budget global « hors N6 » ne court pas pendant N6)."""
        if not self.paused:
            self.paused_at = float(self._clock() - self.started)
            self.paused = True

    def resume(self) -> None:
        """Reprend le décompte du temps écoulé au point de la pause."""
        if self.paused:
            self.started = self._clock() - self.paused_at
            self.paused = False

    def elapsed_at(self, now: float) -> float:
        """Temps écoulé du garde à l'instant `now` (gelé pendant une pause)."""
        if self.paused:
            return float(self.paused_at)
        return float(now - self.started)

    @property
    def elapsed_seconds(self) -> float:
        """Temps écoulé du garde (gelé pendant une pause)."""
        return self.elapsed_at(self._clock())

    def note_rss(self) -> float:
        """Relève la RSS courante et met à jour le pic observé par le garde."""
        current = float(self._rss())
        if current > self.rss_peak_bytes:
            self.rss_peak_bytes = current
        return current

    def check(self, now: float | None = None) -> dict[str, float]:
        now = self._clock() if now is None else float(now)
        self.check_vram_watchdog()
        if self.checks and now - self.last_check < RSS_CHECK_PERIOD_SECONDS:
            # R-J18R-19 (A5) : le throttle de 1 Hz porte sur l'échantillonnage
            # RSS / GPU, **pas** sur le plafond de temps global. Toute `check()`
            # compare d'abord le temps global ; la marge au plafond est de 0 s.
            elapsed = self.elapsed_at(now)
            if elapsed > self.max_seconds:
                raise BudgetExceeded(
                    self.time_kind,
                    f"budget temps dépassé ({elapsed:.1f} s > {self.max_seconds:.1f} s)",
                )
            return {"elapsed_seconds": elapsed}
        self.last_check = now
        self.checks += 1
        elapsed = self.elapsed_at(now)
        rss_bytes = self.note_rss()
        rss_gib = rss_bytes / 1024**3
        gpu_mib = self._gpu_peak_bytes() / 1024**2
        if elapsed > self.max_seconds:
            raise BudgetExceeded(
                self.time_kind, f"budget temps dépassé ({elapsed:.1f} s > {self.max_seconds:.1f} s)"
            )
        if rss_gib > self.max_rss_gib:
            raise BudgetExceeded(
                "rss", f"plafond RSS dépassé ({rss_gib:.2f} GiB > {self.max_rss_gib} GiB)"
            )
        if gpu_mib > self.max_gpu_mib:
            raise BudgetExceeded(
                "gpu", f"pic GPU dépassé ({gpu_mib:.1f} MiB > {self.max_gpu_mib} MiB)"
            )
        return {"elapsed_seconds": elapsed, "rss_gib": rss_gib, "gpu_peak_mib": gpu_mib}

    def check_vram_watchdog(self) -> None:
        """Watchdog VRAM : échantillonnage et arrêt **dur** au seuil.

        Le dépassement lève `BudgetExceeded("gpu_vram", …)` avec le fait mesuré
        ; un instrument indisponible lève `VramWatchdogUnavailable` (refus).
        """
        if self.vram_watchdog is None:
            return
        self.vram_watchdog.sample()
        if self.vram_watchdog.exceeded:
            event = dict(self.vram_watchdog.event or {})
            raise BudgetExceeded(
                "gpu_vram",
                f"plafond VRAM du watchdog dépassé ({event})",
                event=event,
            )

    def guard_time(self, label: str) -> float:
        """R-J18R-19 (A5) : vérification du plafond global par l'horloge injectée.

        Placée **avant** l'écriture de `results.json` et **après** les
        sous-processus, **avant** les traitements finaux : le dépassement du
        plafond ne se découvre **pas** après la persistance. Le temps en pause
        (N6) est exclu ; le temps des autres bras n'est **pas** remis à zéro.
        """
        elapsed = self.elapsed_seconds
        if elapsed > self.max_seconds:
            raise BudgetExceeded(
                self.time_kind,
                f"dépassement du plafond global avant {label} "
                f"({elapsed:.1f} s > {self.max_seconds:.1f} s)",
            )
        return elapsed



class MemoryLimitEnforced(RuntimeError):
    """R-J18R-19 (D9) : le plafond RLIMIT_AS posé en `preexec_fn` **échoue** ;
    le harnais **refuse** la phase 2 (sortie 2), `results.json` non écrit."""


class GroupStopResisted(RuntimeError):
    """R-J18R-19 (D9) : le groupe du sous-processus a **résisté** au signal de
    stop ; le bras n'est **pas** consigné comme complet."""

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


def memory_limit_mib(args: argparse.Namespace) -> int:
    """Plafond RLIMIT_AS des sous-processus §7.3b (J18R-final).

    Le plafond est **mesuré**, pas déduit du plafond RSS : RLIMIT_AS porte sur
    l'espace d'adressage virtuel (réservations CUDA), sans rapport avec
    `--max-rss-gib`. Valeur : `DELEGATED_RLIMIT_AS_GIB` (voir la dérivation).
    """
    return int(DELEGATED_RLIMIT_AS_GIB * 1024)


def memory_limit_available() -> int:
    """Mesure du mécanisme RLIMIT_AS : un child pose la limite en `preexec_fn` **avant**
    `exec` et la **lit** ; 0 = mécanisme absent. Ce harnais **refuse** la phase 2 si 0."""
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import resource, json;"
            " resource.setrlimit(resource.RLIMIT_AS, (64 * 1024**2, 64 * 1024**2));"
            " print(json.dumps(list(resource.getrlimit(resource.RLIMIT_AS))))",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    try:
        soft = int(json.loads(probe.stdout.strip())[0])
    except (OSError, ValueError, IndexError, json.JSONDecodeError):
        return 0
    return int(soft // (1024**2))


def _group_survivors(pgid: int) -> int:
    """Processus **encore vivants** dans le groupe `pgid` après le stop (vue `/proc`)."""
    count = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        try:
            if os.getpgid(pid) == pgid:
                count += 1
        except (ProcessLookupError, PermissionError, OSError):
            continue
    return count


def vram_preflight(
    *,
    min_free_mib: int = VRAM_PREFLIGHT_MIN_FREE_MIB,
    watchdog: VramWatchdog | None = None,
) -> dict[str, int]:
    """Vérifie la VRAM **libre** avant tout lancement GPU (PRE §7.8).

    Refus dur (`VramPreflightRefused`) si la mémoire libre est inférieure au
    seuil : aucun run n'est lancé. Le contrôle est **actif** (aucun processus
    LM Studio n'est déchargé, aucun processus n'est tué).
    """
    reading = (watchdog.query() if watchdog is not None else VramWatchdog.query())
    if reading["free_mib"] < int(min_free_mib):
        raise VramPreflightRefused(
            f"VRAM libre insuffisante avant lancement GPU : {reading['free_mib']} MiB libres "
            f"< seuil {int(min_free_mib)} MiB (total {reading['total_mib']} MiB, "
            f"utilisé {reading['used_mib']} MiB) : STOP, aucun run lancé"
        )
    return reading


def group_rss_bytes(pgid: int) -> int:
    """RSS **cumulée** des processus du groupe `pgid` (en octets), vue `/proc`.

    C'est la mesure qui protège réellement la RAM du guest WSL : `RLIMIT_AS`
    borne l'espace virtuel, pas la RSS. Lecture best-effort : un processus qui
    disparaît pendant la lecture est ignoré (`/proc` non lisible).
    """
    total = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        try:
            if os.getpgid(pid) != pgid:
                continue
            statm = (entry / "statm").read_text(encoding="utf-8").split()
            total += int(statm[1]) * os.sysconf("SC_PAGE_SIZE")
        except (ProcessLookupError, PermissionError, OSError, IndexError, ValueError):
            continue
    return int(total)


@contextlib.contextmanager
def measurement_lock(repo_root: Path = _REPO_ROOT):
    """Exclusion mutuelle des mesures GPU (J18R-final, post-OOM 2026-10-29).

    Verrou `flock` **exclusif et non bloquant** sur un fichier sous `_artefact/`.
    Le second lancement est **refusé** (`MeasurementLockBusy`), jamais mis en
    file : le protocole impose une seule exécution lourde à la fois. Le verrou
    est relâché à la sortie du bloc, y compris en cas d'exception.

    Le descripteur est **conservé ouvert** pendant toute la durée du bloc :
    `flock` est lié au descripteur, pas au processus.
    """
    lock_path = repo_root / MEASUREMENT_LOCK_RELATIVE_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            raise MeasurementLockBusy(
                "une autre mesure GPU detient le verrou d'exclusion mutuelle "
                f"({lock_path}) : lancement refusé (une seule exécution lourde à la fois)"
            ) from exc
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()} {time.monotonic():.3f}\n".encode("utf-8"))
        try:
            yield {"path": str(lock_path), "pid": os.getpid()}
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def run_arm_subprocess(
    invocation: str,
    repo_root: Path = _REPO_ROOT,
    *,
    limit_mib: int,
    stop_after_seconds: float,
    clock: Callable[[], float] = time.monotonic,
    watchdog: VramWatchdog | None = None,
    group_rss_max_bytes: int = int(DELEGATED_GROUP_RSS_MAX_GIB) * 1024**3,
) -> dict[str, object]:
    """§7.3b : invocation du harnais J15 gelé, chaîne exactement figée, **sous
    surveillance mémoire + VRAM + groupe** (R-J18R-19 D9 + J18R-final).

    - `start_new_session=True` : le child **et** ses descendants sont dans le
      **groupe** du harnais ;
    - plafond RLIMIT_AS posé en `preexec_fn` **avant** `exec` (limite dure). Un
      `preexec_fn` qui échoue lève `MemoryLimitEnforced` : refus **sortie 2** du
      harnais **avant** le run complet, `results.json` **non écrit** ;
    - **watchdog VRAM** (J18R-final) : la VRAM **totale** est échantillonnée à
      1 Hz pendant la vie du child ; au-delà du **seuil dur**, le **groupe** est
      tué (SIGTERM puis SIGKILL) et l'arrêt est consigné (`kind = "gpu_vram"`) ;
    - **plafond RSS de groupe** (J18R-final) : la RSS cumulée du groupe est
      échantillonnée à 1 Hz ; au-delà du plafond, le groupe est tué et l'arrêt
      est consigné (`kind = "memory"`) ;
    - au-delà du plafond de **temps** de ce sous-processus (`stop_after_seconds`),
      `SIGTERM` au groupe puis `SIGKILL` 2 s après ; si des membres **résistent**
      (signal refusé, survivants après `SIGKILL`), `GroupStopResisted` — le bras
      n'est **pas** consigné comme complet.
    """
    argv = shlex.split(invocation)
    limit_bytes = int(limit_mib) * 1024**2

    def preexec() -> None:
        # Canal EPIPE du parent fermé : l'exception est relue côté parent.
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

    started = clock()
    try:
        proc = subprocess.Popen(argv, cwd=repo_root, start_new_session=True, preexec_fn=preexec)
    except Exception as exc:  # noqa: BLE001 — pré-exec refusé : pas de run
        raise MemoryLimitEnforced(
            f"plafond RLIMIT_AS ({limit_mib} MiB) non appliqué au sous-processus §7.3b : {exc}"
        ) from exc
    pgid = int(proc.pid)
    stop_signal: str | None = None
    resisted = False
    stop_reason: str | None = None
    stop_event: dict[str, object] | None = None
    group_rss_peak = 0
    deadline = started + float(stop_after_seconds)

    def kill_group(sig: signal.Signals) -> bool:
        """Envoie `sig` au **groupe** ; faux si le signal n'a pas pu être délivré."""
        try:
            os.killpg(pgid, sig)
        except OSError:
            return False
        return True

    def stop_group(reason: str, event: dict[str, object] | None) -> bool:
        """Arrêt **dur** du groupe : SIGTERM puis SIGKILL 2 s après.

        Renvoie faux si un signal n'a pas pu être délivré (groupe résistant).
        """
        nonlocal stop_reason, stop_event, stop_signal
        stop_reason = reason
        stop_event = event
        stop_signal = "SIGTERM"
        if not kill_group(signal.SIGTERM):
            return False
        term_deadline = clock() + 2.0
        while proc.poll() is None and clock() < term_deadline:
            time.sleep(0.05)
        if proc.poll() is None:
            stop_signal = "SIGKILL"
            if not kill_group(signal.SIGKILL):
                return False
        return True

    while proc.poll() is None:
        now = clock()
        if now >= deadline:
            if not stop_group("time", None):
                resisted = True
            break
        # --- watchdog VRAM (seuil dur) -------------------------------------
        if watchdog is not None:
            try:
                watchdog.sample()
            except VramWatchdogUnavailable as exc:
                # Instrument indisponible : refus (pas de supposition de budget).
                if not stop_group("vram_watchdog_unavailable", {"reason": "vram_watchdog_unavailable", "error": str(exc)}):
                    resisted = True
                break
            if watchdog.exceeded:
                if not stop_group("vram", dict(watchdog.event or {})):
                    resisted = True
                break
        # --- plafond RSS de groupe -----------------------------------------
        rss = group_rss_bytes(pgid)
        if rss > group_rss_peak:
            group_rss_peak = rss
        if group_rss_max_bytes and rss > group_rss_max_bytes:
            event = {
                "reason": "group_rss_above_hard_threshold",
                "threshold_bytes": int(group_rss_max_bytes),
                "observed_bytes": int(rss),
                "observed_gib": float(rss / 1024**3),
                "pgid": int(pgid),
            }
            if not stop_group("memory", event):
                resisted = True
            break
        time.sleep(0.05)
    proc.wait()
    survivors = _group_survivors(pgid)
    resisted = bool(resisted or survivors > 0)
    group: dict[str, object] = {
        "pgid": pgid,
        "limit_mib": int(limit_mib),
        "limit_applied": True,
        "stop_after_seconds": float(stop_after_seconds),
        "stop_signal": stop_signal,
        "stop_reason": stop_reason,
        "stop_event": stop_event,
        "survivors_after_stop": int(survivors),
        "verified_after_stop": bool(not resisted),
        "elapsed_seconds": float(clock() - started),
        "group_rss_peak_bytes": int(group_rss_peak),
        "group_rss_peak_gib": float(group_rss_peak / 1024**3),
        "group_rss_max_bytes": int(group_rss_max_bytes),
        "group_rss_limit_applied": bool(group_rss_max_bytes),
    }
    if watchdog is not None:
        group["vram_watchdog"] = watchdog.to_dict()
    if stop_reason == "vram":
        raise BudgetExceeded(
            "gpu_vram",
            f"plafond VRAM du watchdog dépassé ({stop_event}) : groupe {pgid} arrêté",
            event=stop_event,
        )
    if stop_reason == "memory":
        raise BudgetExceeded(
            "memory",
            f"plafond RSS du groupe dépassé ({stop_event}) : groupe {pgid} arrêté",
            event=stop_event,
        )
    if stop_reason == "vram_watchdog_unavailable":
        raise VramWatchdogUnavailable(
            f"watchdog VRAM indisponible pendant le sous-processus §7.3b : {stop_event}"
        )
    return {
        "returncode": int(proc.returncode),
        "group": group,
        "resisted": bool(resisted),
    }


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


#: plafond du cache de préfixe du scorer de production ; miroir de
#: `teacher_scorer_qwen3.py:122` (`MAX_PREFIX_CACHE_BYTES = 16 * 1024**3`).
MAX_PREFIX_CACHE_BYTES = 16 * 1024**3


class ProductionScoreProvider:
    """Adapter du `TeacherScorer` de production sur le contrat §5.1.

    Le contrat §5.1 s'exprime en préfixe **relatif** du bloc (`future[:position]`) ; le scorer
    de production conditionne ses probabilités sur la séquence complète `q + z` (eq. 14), comme
    le support J15 (`model_forward` forward `prompt_ids + prefix`). À `position = 0` le préfixe
    relatif est vide et `Qwen3TeacherScorer` le refuse (`teacher_scorer_qwen3.py:288-289` et
    `:380-381`) : le provider porte donc les identifiants du prompt (D4, R-J18R-12).
    """

    def __init__(self, scorer, prompt_ids: Sequence[int]) -> None:
        self._scorer = scorer
        self._prompt_ids = tuple(int(token) for token in prompt_ids)

    def _conditioned(self, prefix: Sequence[int]) -> tuple[int, ...]:
        """`q + z_i` : les identifiants du prompt précèdent le préfixe relatif du bloc."""
        return self._prompt_ids + tuple(int(token) for token in prefix)

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
            prefix_ids=self._conditioned(prefix),
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
            list(self._conditioned(prefix)),
            [int(token) for token in candidates],
            [int(token) for token in suffix],
        )
        # Le contrat §5.2 attend le prior **aux candidats** ; le scorer de production rend le
        # prior plein vocabulaire (eq. 14) : il faut donc le restreindre (R-J18R-12).
        prior_array = np.asarray(prior, dtype=np.float64)
        ids = np.asarray([int(token) for token in candidates], dtype=np.int64)
        restricted = prior_array[ids] if prior_array.ndim == 1 and prior_array.size > ids.max(initial=-1) else prior_array
        return restricted, np.asarray(scores, dtype=np.float64)


def production_provider_factory(
    scorer_class: Callable[..., object],
    model: object,
    device: str,
    max_prefix_cache_bytes: int = MAX_PREFIX_CACHE_BYTES,
) -> Callable[[Sequence[int]], ScoreProvider]:
    """Fabrique **un provider par prompt** (un scorer, un cache de préfixe par séquence).

    Constructeur de production : `Qwen3TeacherScorer(model, *, device=..., ...)` — le tokenizer
    est hors du contrat `TeacherScorer` et `device` est lié par mot-cle (D1, R-J18R-07).
    """

    def factory(prompt_ids: Sequence[int]) -> ScoreProvider:
        return ProductionScoreProvider(
            scorer_class(model, device=device, max_prefix_cache_bytes=max_prefix_cache_bytes),
            prompt_ids,
        )

    return factory


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


def extract_input_ids(encoded: object) -> list[int]:
    """Extrait la **liste plate** des identifiants d'un encodage de tokenizer.

    Formes acceptées (R-J18R-13) :

    - `Mapping` (y compris `BatchEncoding`, qui dérive de `UserDict` et **n'est pas**
      un `dict`) : seul le champ `input_ids` est extrait ; les autres champs
      (`attention_mask`, ...) sont ignorés ;
    - liste plate d'entiers ;
    - liste contenant **une seule** séquence d'entiers (forme `[1, longueur]`) :
      elle est aplatie ; plusieurs séquences sont refusées.

    Toute autre forme — vide, non-entier, ambiguë — est refusée par `ValueError`.
    """
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise ValueError("encodage Mapping sans champ 'input_ids'")
        raw = encoded["input_ids"]
    else:
        raw = encoded
    if isinstance(raw, (str, bytes)):
        raise ValueError("encodage invalide : texte reçu là où des identifiants sont attendus")
    if hasattr(raw, "tolist") and not isinstance(raw, (list, tuple)):
        raw = raw.tolist()
    if isinstance(raw, Integral) or not isinstance(raw, Sequence):
        raise ValueError(f"forme d'encodage refusée : {type(raw).__name__}")
    values = list(raw)
    if values and all(
        isinstance(item, Sequence) and not isinstance(item, (str, bytes)) for item in values
    ):
        if len(values) != 1:
            raise ValueError("encodage ambigu : plusieurs séquences refusées")
        values = list(values[0])
    ids: list[int] = []
    for token in values:
        if isinstance(token, bool) or not isinstance(token, Integral):
            raise ValueError(f"identifiant non entier : {token!r}")
        ids.append(int(token))
    if not ids:
        raise ValueError("encodage vide : aucun identifiant")
    return ids


def prompt_token_ids(tokenizer, text: str, device: object, *, chat_template: bool):
    """Encodage d'un prompt : base (§1.2) ou variante N6 avec gabarit de conversation (§6).

    N6 (R-J18R-13) : le gabarit est rendu **en texte** (`tokenize=False`), puis le
    rendu est encodé par le tokenizer (`add_special_tokens=False`). Cette voie
    évite la dérive d'API `transformers` (le retour `tokenize=True` est un
    `BatchEncoding`, pas une liste) et garde `enable_thinking=False` effectif.
    """
    torch = _torch()
    if chat_template:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(rendered, str):
            raise ValueError(
                f"apply_chat_template(tokenize=False) doit rendre un texte, reçu {type(rendered).__name__}"
            )
        encoded = tokenizer(rendered, add_special_tokens=False)
        ids = extract_input_ids(encoded)
        return torch.tensor(ids, dtype=torch.long, device=device)
    return tokenizer(text, return_tensors="pt").to(device)["input_ids"][0]


def run_arm(
    *,
    arm: str,
    args: argparse.Namespace,
    prompts: list[tuple[str, str]],
    model: object,
    provider_factory: Callable[[Sequence[int]], ScoreProvider],
    tokenizer: object,
    device: object,
    budget: BudgetGuard,
    integrity: IntegrityStats,
    replay_stats: ReplayStats,
    clock: Callable[[], float] = time.monotonic,
    observations_out: list[MarginalObservation] | None = None,
) -> dict[str, object]:
    """Bras exécuté par J18R : construction économe, arithmétique FP32, boucle figée.

    Le provider est construit **par prompt** (D4, R-J18R-12) : ses scores sont conditionnés
    sur `q + z_i`, et `position = 0` donne un préfixe relatif vide, que le scorer refuse.

    `observations_out` reçoit les observations **au fil de l'eau** : un arrêt en cours
    de bras (plafond N6) laisse ainsi les observations déjà produites (R-J18R-14).

    Deux états **du bras** sont créés ici, pas par configuration : le compteur de
    **jobs** (§3.5, R-J18R-17) et la liste des observations. Le bras de **contrôle**
    ne tourne **que** sur `σ = 0` (grille figée §2.5, R-J18R-17).
    """
    observations: list[MarginalObservation] = (
        observations_out if observations_out is not None else []
    )
    started = clock()
    # R-J18R-17 : compteur de **jobs du bras** — traverse toutes les configurations
    # sans remise à zéro ; c'est ce qui donne le taux réel de 5 % du rejeu §5.2.
    job_counter = JobCounter()
    for prompt_id, text in prompts:
        if arm == "control" and prompt_id not in CONTROL_PROMPTS:
            continue
        prompt_ids = prompt_token_ids(tokenizer, text, device, chat_template=(arm == "n6_chat_template"))
        provider = provider_factory(prompt_ids)
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
                # R-J18R-17 : le contrôle ne tourne que sur σ = 0 (§2.5) ; les
                # seuils C5(e) sont des seuils d'identité, pas de perturbation.
                noise_levels=CONTROL_NOISE_LEVELS if arm == "control" else NOISE_LEVELS,
                job_counter=job_counter,
                budget=budget,
                # R-J18R-19 (A1/D5) : les blocs **complets** sont engagés sur la
                # liste partagée au fil de l'eau ; l'arrêt du bras (plafond N6)
                # ne laisse **pas** la publication vide.
                observations_out=observations,
            )
    replay_stats.n_jobs_expected += len(observations)
    records = [observation.record() for observation in observations]
    elapsed = float(clock() - started)
    return {
        "arm": arm,
        "records": records,
        "publications": arm_publications(observations, arm=arm, bootstrap_seed=args.bootstrap_seed),
        "complete": bool(records),
        "status": "completed",
        "elapsed_seconds": elapsed,
        "n_observations": len(observations),
        # nombre de **jobs** émis par le bras (§3.5) — base du taux de rejeu §5.2.
        "n_jobs": int(job_counter.value),
        # R-J18R-19 (A1) : métadonnées de **couverture** (blocs complets engagés).
        "coverage": {
            "n_jobs_emitted": int(job_counter.value),
            "n_blocks_completed": int(
                len({(str(o.prompt), str(o.config_index), int(o.block_index)) for o in observations})
            ),
        },
    }


#: Types d'**infrastructure** jamais absorbés par le garde N6 (R-J18R-16) :
#: épuisement d'allocation (`MemoryError`, `torch.cuda.OutOfMemoryError`).
_INFRA_FAILURE_TYPE_NAMES: frozenset[str] = frozenset({"MemoryError", "OutOfMemoryError"})

#: Motifs de message d'un échec mémoire / GPU (torch, CUDA, allocator) — R-J18R-16.
_INFRA_FAILURE_MESSAGE_PATTERNS: tuple[str, ...] = (
    "out of memory",
    "cannot allocate",
    "can't allocate",
    "allocation failed",
    "insufficient memory",
    "failed to allocate",
)


def is_infra_failure(error: BaseException) -> bool:
    """L'échec est d'**infrastructure** (mémoire / GPU) : arrêt du run, pas un incident N6.

    La liste est **étroite et déclarée** (R-J18R-16) : l'interception ajoutée en
    R-J18R-14 ne vaut **pas** « toute exception ⇒ `partial` valide ». Elle couvre les
    **échecs métier** du bras non décisionnel (gate T1, troncature, gabarit de
    conversation refusé, provider refusé, grammaire du prompt). Un épuisement de
    mémoire n'est pas une divergence d'implémentation que N6 peut absorber : la VRAM
    de N6 est **la même** que celle des bras décisionnels et le PRE §7.8 dit « arrêt
    immédiat » pour les plafonds.
    """
    if type(error).__name__ in _INFRA_FAILURE_TYPE_NAMES:
        return True
    if getattr(error, "errno", None) == errno.ENOMEM:
        return True
    message = str(error).casefold()
    return any(pattern in message for pattern in _INFRA_FAILURE_MESSAGE_PATTERNS)


def run_arm_guarded(arm: str, **kwargs) -> dict[str, object]:
    """Exécute un bras en **interceptant les exceptions du seul bras N6** (R-J18R-14, R-J18R-16).

    Interception **limitée à N6**, et la règle est **à trois étages** (R-J18R-16),
    pas « toute exception ⇒ `partial` » :

    - `BudgetExceeded.kind == "time"` (plafond **propre de N6**, `--n6-max-seconds`) :
      arrêt du bras N6 seul, publication partielle explicite, le **run continue** ;
    - `BudgetExceeded.kind` **global** (`"rss"`, `"gpu"`, `"global_time"`) : **propagé**.
      Le budget N6 (`n6_budget`) porte les **mêmes** seuils RSS / VRAM que le budget
      global : un dépassement de mémoire pendant N6 n'est pas absorbable ;
    - échec d'**infrastructure** mémoire / GPU (`is_infra_failure`) : **propagé** ;
    - échec **métier** N6 (gate T1, troncature, gabarit, provider) : publication
      partielle sur les **seules observations émises**, sans statistiques inventées
      si le bras n'a rien émis.
    """
    clock = kwargs.get("clock", time.monotonic)
    started = clock()
    if arm != "n6_chat_template":
        return run_arm(arm=arm, **kwargs)
    observations: list[MarginalObservation] = []
    try:
        return run_arm(arm=arm, observations_out=observations, **kwargs)
    except BudgetExceeded as exc:
        if exc.kind != "time":
            raise
        return _n6_partial_result(
            arm, observations, exc, started, clock, kwargs["args"].bootstrap_seed
        )
    except Exception as exc:  # noqa: BLE001 — interception **limitée** au bras N6
        if is_infra_failure(exc):
            # mémoire / GPU : arrêt du run, pas un incident du bras non décisionnel.
            raise
        return _n6_partial_result(
            arm, observations, exc, started, clock, kwargs["args"].bootstrap_seed
        )


def _n6_partial_result(
    arm: str,
    observations: list[MarginalObservation],
    error: BaseException,
    started: float,
    clock: Callable[[], float],
    bootstrap_seed: int = BOOTSTRAP_SEED_DECISION,
) -> dict[str, object]:
    """Publication **partielle** du bras N6 (PRE §6, R-J18R-14).

    - plafond N6 atteint (`BudgetExceeded.kind == "time"`) : publication partielle
      explicite, décision principale conservée ;
    - observations puis exception : échec consigné, données **diagnostiques** seulement,
      jamais un « résultat N6 valide » ;
    - échec avant toute observation : aucune statistique inventée.
    """
    records = [observation.record() for observation in observations]
    publications = arm_publications(observations, arm=arm, bootstrap_seed=bootstrap_seed)
    publications["partial"] = True
    publications["diagnostic_only"] = bool(records)
    status = "budget_exceeded" if isinstance(error, BudgetExceeded) else "failed"
    return {
        "arm": arm,
        "records": records,
        "publications": publications,
        "complete": False,
        "status": status,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "kind": getattr(error, "kind", None),
        },
        "elapsed_seconds": float(clock() - started),
        "n_observations": len(observations),
        # R-J18R-17 : nombre de **jobs émis avant l'arrêt** (index maximal du compteur
        # du bras) — base de la couverture de rejeu publiée par le bloc N6.
        "n_jobs": int(max([int(obs.job_index) for obs in observations if obs.job_index is not None], default=0)),
        # R-J18R-19 (A1) : couverture **réellement engagée** au moment de l'arrêt
        # (jobs émis, blocs complets sur la liste partagée).
        "coverage": {
            "n_jobs_emitted": int(max([int(obs.job_index) for obs in observations if obs.job_index is not None], default=0)),
            "n_blocks_completed": int(
                len({(str(o.prompt), str(o.config_index), int(o.block_index)) for o in observations})
            ),
        },
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


def criterion_c4(
    publications: Mapping[str, Mapping[str, object]], blocks: int, n_prompts: int
) -> dict[str, object]:
    """C4 : effectifs par configuration et blocs de base (§3.2, §4.4).

    R-J18R-14 : le bras **N6 est non décisionnel** (§6, §4.5) — il ne participe
    **pas** à C4 ; sa partialité (plafond 1800 s, exception) n'invalide donc pas
    le run. Ses effectifs restent publiés dans `arms.n6_chat_template`.
    """
    per_config: dict[str, object] = {}
    holds = True
    for arm, values in publications.items():
        if arm == "n6_chat_template":
            continue
        for config, counts in (values.get("block_counts") or {}).items():
            expected_blocks = expected_blocks_per_config(arm, n_prompts, blocks)
            ok = int(counts["n_blocks_sampled"]) == expected_blocks and int(
                counts["n_distinct_block_indices"]
            ) <= (len(CONTROL_BLOCKS) if arm == "control" else blocks)
            if arm == "decision":
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
    *,
    observed_arms: Sequence[str] = ("control", "decision"),
    excluded_arms: Sequence[str] = ("anchor", "sensitivity_fp32", "n6_chat_template"),
    arm_jobs: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """C5 : intégrité numérique (finitude, sommes, rejeu, float32 appliqué, contrôle).

    R-J18R-15 (constat D5 / audit A2) : C5 est évalué **sur les bras de ses
    instances**. Le bras N6 a ses **propres** instances (`n6_integrity`,
    `n6_replay`) et publie son intégrité dans le **bloc N6** : une anomalie N6
    **n'entre pas dans C5** et ne déclenche **pas** `INVALIDE`. Les bras délégués
    §7.3b (`anchor`, `sensitivity_fp32`) sont mesurés par leurs propres harnais.

    R-J18R-17 (constat D6) : la couverture de rejeu §5.2 est calculée sur le nombre
    de **jobs du bras de décision** (`arm_jobs["decision"]`), produit par un
    compteur **monotone par bras** (pas par configuration). Le bras de contrôle
    (18 jobs, `σ = 0`, §1.5/§2.5) n'atteint **pas** le multiple 20 : sa validité est
    la clause d'**identité** C5(e), pas une couverture de 5 %. La couverture est
    **publiée**, jamais présupposée.
    """
    jobs = dict(arm_jobs) if arm_jobs is not None else {}
    sampled = int(jobs.get("decision", 0))
    if sampled == 0 and observed_arms:
        # bras évalué sur ses **propres** instances (N6) : dénominateur = ses jobs.
        sampled = int(jobs.get(observed_arms[0], 0))
    replayed = int(replay_stats.n_jobs)
    expected = int(math.ceil(REPLAY_FRACTION * sampled)) if sampled > 0 else 1
    coverage = float(replayed / sampled) if sampled > 0 else 0.0
    clause_c_holds = bool(
        not replay_stats.violations
        and replayed >= max(expected, 1)
        and (sampled == 0 or coverage >= REPLAY_FRACTION - 1e-12)
    )
    return {
        "holds": bool(
            integrity.finite
            and integrity.max_sum_error <= SUM_ERROR_TOLERANCE
            and clause_c_holds
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
        "clause_c_replay": dict(
            replay_stats.to_dict(),
            holds=bool(clause_c_holds),
            sampled_jobs=int(sampled),
            expected_replays=int(max(expected, 1)),
            coverage=coverage,
            coverage_note=(
                "dénominateur = jobs du **bras de décision** (compteur monotone par "
                "bras, §3.5) ; le bras de contrôle (18 jobs, σ = 0) n'atteint pas le "
                "multiple 20 et est validé par la clause d'identité C5(e), pas par "
                "une couverture de 5 %"
            ),
        ),
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
        # R-J18R-15 : portée **déclarée** de C5 — les bras observés par ces instances.
        "observed_arms": list(observed_arms),
        "excluded_arms": list(excluded_arms),
        "arm_jobs": {arm: int(count) for arm, count in jobs.items()},
        "scope_note": (
            "C5 est évalué sur les bras mesurés par ce harnais ; le bras N6 a ses "
            "propres instances d'intégrité/rejeu et publie son intégrité dans le bloc "
            "N6 (non décisionnel) ; les bras délégués §7.3b sont mesurés par leurs "
            "propres harnais"
        ),
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


def _n6_status_block(
    arm_result: Mapping[str, object] | None,
    integrity_c5: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Bloc d'état du bras N6 (PRE §6, R-J18R-14, R-J18R-15) : statut, cause, intégrité propre.

    N6 est **non décisionnel** : son état **et son intégrité** (instances propres)
    sont publiés ici, et ne modifient **jamais** l'issue principale (C1–C10).
    """
    if arm_result is None:
        return {"present": False, "status": "not_run", "decisionnal": False, "partial": False}
    error = arm_result.get("error") or {}
    publications = arm_result.get("publications") or {}
    return {
        "present": True,
        "decisionnal": False,
        "status": str(arm_result.get("status", "completed")),
        "partial": bool(publications.get("partial", False)),
        "diagnostic_only": bool(publications.get("diagnostic_only", False)),
        "n_observations": int(arm_result.get("n_observations", 0)),
        "elapsed_seconds": float(arm_result.get("elapsed_seconds", 0.0)),
        # R-J18R-19 (A1) : couverture **réellement engagée** au moment de l'arrêt
        # (jobs émis, blocs complets sur la liste partagée).
        "coverage": dict(arm_result.get("coverage") or {"n_jobs_emitted": 0, "n_blocks_completed": 0}),
        "error": {
            "type": error.get("type"),
            "message": error.get("message"),
            "kind": error.get("kind"),
        }
        if error
        else None,
        # R-J18R-15 : intégrité **du bras N6**, évaluée sur ses propres instances.
        "integrity_c5": (dict(integrity_c5) if integrity_c5 is not None else None),
        "integrity_note": (
            "intégrité évaluée sur les instances propres de N6 ; jamais dans `criteria`, "
            "jamais dans le C5 du run (R-J18R-15)"
        ),
    }


def execute(args: argparse.Namespace, argv: Sequence[str]) -> dict[str, object]:
    """Run complet : garde-fou avant chargement, bras, critères, décision, shards, sidecar.

    J18R-final : le run est exécuté **sous** le verrou d'exclusion mutuelle des
    mesures GPU (une seule exécution lourde à la fois) et **après** le preflight
    VRAM. Un verrou déjà pris ou une VRAM libre insuffisante ⇒ **refus**
    (sortie 2), aucun chargement, aucun artefact de résultat.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with measurement_lock(_REPO_ROOT) as lock:
            # J18R-final — **preflight VRAM** (PRE §7.8) : VRAM libre mesurée
            # **avant** tout chargement ; refus si la mémoire libre est sous le
            # seuil. Aucun processus n'est déchargé, aucun processus n'est tué.
            watchdog = VramWatchdog()
            preflight = vram_preflight(watchdog=watchdog)
            torch = _torch()
            transformers = _transformers()
            return _execute_locked(
                args, argv, torch, transformers, output_dir, watchdog, preflight, lock
            )
    except (MeasurementLockBusy, VramPreflightRefused, VramWatchdogUnavailable) as exc:
        print(f"J18R : {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _execute_locked(
    args: argparse.Namespace,
    argv: Sequence[str],
    torch,
    transformers,
    output_dir: Path,
    watchdog: VramWatchdog,
    preflight: Mapping[str, int],
    lock: Mapping[str, object],
) -> dict[str, object]:
    """Corps du run, exécuté **sous** le verrou d'exclusion mutuelle (J18R-final)."""
    # §7.7 : vérification d'appartenance au dépôt AVANT tout chargement (C8).
    guard = verify_model_source_in_repo(args.model, _REPO_ROOT)
    scorer_module = load_scorer_module()
    model, source = scorer_module.load_teacher(args.model, device=args.device, dtype=args.dtype_decision)
    tokenizer = transformers.AutoTokenizer.from_pretrained(source)
    # D1 + D4 (R-J18R-07, R-J18R-12) : le constructeur de production prend `model` et le
    # `device` par mot-cle, jamais le tokenizer ; le provider est construit par prompt, son
    # préfixe est conditionné sur `q + z_i` (eq. 14).
    provider_factory = production_provider_factory(scorer_module.Qwen3TeacherScorer, model, device=args.device)
    counter = BackwardCounter()
    integrity = IntegrityStats()
    fraction = float(args.replay_fraction)
    replay_stats = ReplayStats(
        tolerance=float(args.replay_tolerance),
        modulo=int(round(1.0 / fraction)) if fraction > 0 else REPLAY_MODULO,
    )
    # R-J18R-15 (constat D5 / audit A2) : le bras **N6** a ses **propres** instances
    # d'intégrité et de rejeu. `C5` est évalué sur les bras **qu'il observe**
    # (contrôle + décision) ; une anomalie N6 est publiée dans le **bloc N6** et ne
    # contamine **ni** `C5`, **ni** `criteria`, **ni** l'issue de la table §4.5 :
    # l'issue n'est pas fonction du statut N6.
    n6_integrity = IntegrityStats()
    n6_replay = ReplayStats(tolerance=float(args.replay_tolerance), modulo=replay_stats.modulo)

    def gpu_peak_bytes() -> float:
        return float(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0.0

    budget = BudgetGuard(
        max_seconds=args.max_seconds,
        max_gpu_mib=args.max_gpu_mib,
        max_rss_gib=args.max_rss_gib,
        gpu_peak_bytes=gpu_peak_bytes,
        time_kind="global_time",
        vram_watchdog=watchdog,
    )
    n6_budget = BudgetGuard(
        max_seconds=args.n6_max_seconds,
        max_gpu_mib=args.max_gpu_mib,
        max_rss_gib=args.max_rss_gib,
        gpu_peak_bytes=gpu_peak_bytes,
        time_kind="time",
        vram_watchdog=watchdog,
    )
    prompts = load_prompt_set(_REPO_ROOT / args.prompt_file)
    arms: dict[str, dict[str, object]] = {}
    produced: dict[str, str] = {}

    # §7.3b : ancrage et sensibilité rendus par le harnais J15 gelé, ingérés tels quels.
    # `--output` suit `--output-dir` (D3, R-J18R-09) ; les bruts ne sont **jamais** réécrits
    # et leur hash est relevé avant ingestion (D2, R-J18R-08). Un sous-processus défaillant
    # rend le bras incomplet : `arms_complete` est faux, `results.json` n'est pas écrit (§7.2).
    for arm in ("anchor", "sensitivity_fp32"):
        invocation = (
            anchor_invocation(str(args.output_dir), args.blocks)
            if arm == "anchor"
            else sensitivity_invocation(str(args.output_dir), args.blocks)
        )
        try:
            # R-J18R-19 (D9) + J18R-final : sous-processus **dans le groupe** du
            # harnais, plafond RLIMIT_AS posé **avant** `exec`, stop
            # SIGTERM→SIGKILL à son plafond de **temps** (global), **watchdog VRAM**
            # branché (seuil dur, kill du groupe) et **plafond RSS de groupe**.
            delegated = run_arm_subprocess(
                invocation,
                _REPO_ROOT,
                limit_mib=memory_limit_mib(args),
                stop_after_seconds=float(args.max_seconds),
                watchdog=watchdog,
                group_rss_max_bytes=int(DELEGATED_GROUP_RSS_MAX_GIB) * 1024**3,
            )
        except MemoryLimitEnforced as exc:
            # Mécanisme du plafond **non appliqué** : refus **avant** le run complet,
            # `results.json` **non écrit** (sortie 2).
            print(f"J18R : {exc}", file=sys.stderr)
            raise SystemExit(2)
        except BudgetExceeded as exc:
            # J18R-final : dépassement VRAM / RSS du groupe — arrêt **dur** du run.
            # Le fait mesuré est écrit dans un artefact d'abandon (hors `results.json`,
            # qui ne doit porter que des données de run valides).
            write_json(
                output_dir / "budget_abort.json",
                {
                    "kind": exc.kind,
                    "message": str(exc),
                    "event": exc.event,
                    "arm": arm,
                    "invocation": invocation,
                    "vram_watchdog": watchdog.to_dict(),
                },
            )
            print(f"J18R : {exc}", file=sys.stderr)
            raise
        group = dict(delegated.get("group") or {})
        returncode = int(delegated["returncode"])
        resisted = bool(delegated.get("resisted"))
        delegated_elapsed = float(group.get("elapsed_seconds", 0.0))
        raw_path = output_dir / RAW_SHARDS[arm]
        # R-J18R-10 : un sous-processus J15 défaillant ne produit pas son brut ; le bras est
        # incomplet (`arms_complete` faux, `results.json` non écrit), cela ne fait pas-crasher le run.
        raw_hash = sha256_file(str(raw_path)) if raw_path.exists() else None
        records = ingest_arm_shard(raw_path, arm) if raw_path.exists() else []
        publications = arm_publications(
            observations_from_records(records), arm=arm, bootstrap_seed=args.anchor_bootstrap_seed
        )
        arms[arm] = {
            "arm": arm,
            "invocation": invocation,
            "returncode": int(returncode),
            "records": records,
            "publications": publications,
            # R-J18R-19 (D9) : un groupe qui **résiste** au stop rend le bras
            # **non complet** ; il n'est **jamais** consigné comme complet.
            "complete": bool(records) and returncode == 0 and not resisted,
            "status": (
                "completed" if (records and returncode == 0 and not resisted) else "failed"
            ),
            "elapsed_seconds": delegated_elapsed,
            "n_observations": len(records),
            "group": group,
            # R-J18R-18 (constat D9) : le plafond RLIMIT_AS est posé **avant** `exec`
            # sur le child **et ses descendants** (groupe) ; le **pic v1** du groupe est
            # mesuré **pendant** la vie du sous-processus. Les descendants et la mémoire
            # GPU ne sont **pas** couverts par la mesure du harnais seul.
            "delegated_memory": delegated_memory_observation(group),
        }
        if raw_hash is not None:
            produced[RAW_SHARDS[arm]] = raw_hash
        if arm in ARM_SHARDS:
            # §7.2 : seul l'ancrage a un shard étiqueté ; la sensibilité publie dans `results.json`.
            arms[arm]["raw_sha256_before_ingestion"] = raw_hash
            produced[ARM_SHARDS[arm]] = write_json(output_dir / ARM_SHARDS[arm], arms[arm])

    for arm in executed_arms(args):
        # R-J18R-14 : N6 est **non décisionnel** et s'exécute **en dernier**, après les
        # shards contrôle / décision. Le budget global « hors N6 » (PRE §7.8) est donc
        # **en pause** pendant N6 ; les plafonds RSS / VRAM globaux restent surveillés
        # par le garde N6 (mêmes seuils) et restent des arrêts du run.
        if arm == "n6_chat_template":
            budget.pause()
        try:
            result = run_arm_guarded(
                arm,
                args=args,
                prompts=prompts,
                model=model,
                provider_factory=provider_factory,
                tokenizer=tokenizer,
                device=args.device,
                budget=n6_budget if arm == "n6_chat_template" else budget,
                # R-J18R-15 : N6 alimente ses **propres** instances ; ses marginaux
                # n'entrent **jamais** dans les instances de C5.
                integrity=n6_integrity if arm == "n6_chat_template" else integrity,
                replay_stats=n6_replay if arm == "n6_chat_template" else replay_stats,
            )
        finally:
            if arm == "n6_chat_template":
                budget.resume()
        arms[arm] = result
        produced[ARM_SHARDS[arm]] = write_json(output_dir / ARM_SHARDS[arm], result)

    final_started = time.monotonic()
    control = control_metrics(arms["control"]["records"])
    j15_reference = json.loads((_REPO_ROOT / "modules/dopd/eval/annexe_d_qwen3_0.6b_multiprompt/results.json").read_text(encoding="utf-8"))
    c1 = criterion_c1(arms["decision"]["records"], arms["decision"]["publications"], bootstrap_seed=args.bootstrap_seed, iterations=args.bootstrap_iterations)
    reserves = evaluate_reserves(arms["decision"]["publications"])
    criteria_shell = {
        "C4": criterion_c4({arm: arms[arm]["publications"] for arm in arms}, args.blocks, len(prompts)),
        # R-J18R-17 : le dénominateur de la couverture de rejeu est le nombre de
        # **jobs** du bras de décision, compté par le compteur monotone du bras.
        "C5": criterion_c5(
            integrity,
            replay_stats,
            control,
            arm_jobs={arm: int(arms[arm].get("n_jobs", 0)) for arm in ("control", "decision")},
        ),
        "C6": criterion_c6(argv, args),
        "C7": criterion_c7(model, counter),
        "C8": {"holds": bool(guard.get("model_source_in_repo"))},
        "C9": criterion_c9(
            arms["anchor"]["records"],
            j15_reference,
            blocks=args.blocks,
            anchor_executed=bool(arms["anchor"]["complete"]),
            # R-J18R-19 (A4) : l'inventaire attendu est celui des prompts chargés.
            n_prompts=len(prompts),
        ),
        "C10": criterion_c10(_REPO_ROOT),
    }
    flags = {key: bool(value["holds"]) for key, value in criteria_shell.items()}
    flags["C1"] = bool(c1["holds"])
    flags["C1_sigma_positive"] = bool((c1.get("sigma_positive_only") or {}).get("holds"))
    flags["arms_complete"] = all(bool(arms[arm]["complete"]) for arm in REQUIRED_ARMS)
    # R-J18R-10 : sans ligne partagée avec l'artefact J15, l'écart n'est pas mesuré ;
    # la priorité 1 (`STOP`) est réservée à un ancrage mesuré en échec.
    flags["C9_evaluable"] = bool(criteria_shell["C9"]["evaluable"])
    decision = decide_j18r_issue(flags, reserves)
    # R-J18R-15 : l'intégrité du bras N6 est évaluée sur ses **propres** instances et
    # publiée **dans le bloc N6** ; elle n'entre pas dans `criteria`, et une anomalie
    # N6 ne déclenche pas `INVALIDE`.
    n6_arm = arms.get("n6_chat_template")
    n6_c5: dict[str, object] | None = None
    if n6_arm is not None:
        # Le bras N6 est évalué sur ses propres jobs : dénominateur = ses jobs.
        n6_c5 = criterion_c5(
            n6_integrity,
            n6_replay,
            control,
            observed_arms=("n6_chat_template",),
            excluded_arms=("control", "decision", "anchor", "sensitivity_fp32"),
            arm_jobs={"n6_chat_template": int(n6_arm.get("n_jobs", 0))},
        )


    # R-J18R-18 (constat D9) : portée **réelle** du plafond RSS, et mesure des enfants.
    children_peak = child_rss_peak_bytes()
    cgroup_peak = cgroup_memory_peak_bytes()
    runtime = {
        "elapsed_seconds": float(budget.check()["elapsed_seconds"]),
        "budget_checks": int(budget.checks),
        "rss_gib_at_end": float(current_rss_bytes() / 1024**3),
        "rss_peak_gib": float(max(budget.rss_peak_bytes, n6_budget.rss_peak_bytes) / 1024**3),
        "gpu_peak_mib": float(gpu_peak_bytes() / 1024**2),
        "device": args.device,
        "dtypes": {
            "anchor": args.dtype_anchor,
            "decision": args.dtype_decision,
            "control": args.dtype_control,
            "sensitivity": args.dtype_sensitivity,
        },
        "versions": versions_block(),
        # R-J18R-18 : le plafond `--max-rss-gib` est appliqué au **processus du harnais**.
        # Les invocations §7.3b sont des **sous-processus** non couverts par cette mesure ;
        # la mesure disponible (pic des enfants **attendus**, pic cgroup) est consignée,
        # avec un « non observé » explicite quand elle n'existe pas.
        "rss_budget_scope": (
            "plafond --max-rss-gib appliqué à la RSS du processus du harnais "
            "(/proc/self/statm) ; les invocations d'ancrage §7.3b (ancrage BF16, "
            "sensibilité FP32) sont des sous-processus dont la mémoire n'est pas "
            "couverte par ce plafond"
        ),
        "rss_peak_children_gib": float(children_peak / 1024**3),
        "cgroup_memory_peak_gib": (None if cgroup_peak is None else float(cgroup_peak / 1024**3)),
        "delegated_memory": {arm: arms[arm].get("delegated_memory") for arm in ("anchor", "sensitivity_fp32")},
        # J18R-final — trace du **watchdog VRAM** (branché), du **preflight** et du
        # **verrou d'exclusion mutuelle** : faits mesurés, jamais reconstruits.
        "vram_watchdog": watchdog.to_dict(),
        "vram_preflight": {
            "used_mib": int(preflight["used_mib"]),
            "free_mib": int(preflight["free_mib"]),
            "total_mib": int(preflight["total_mib"]),
            "min_free_mib_required": int(VRAM_PREFLIGHT_MIN_FREE_MIB),
        },
        "measurement_lock": {
            "path": str(lock.get("path")),
            "pid": int(lock.get("pid", os.getpid())),
            "exclusive": True,
            "scope": (
                "une seule exécution lourde à la fois (post-OOM 2026-10-29) ; le "
                "second lancement est refusé, jamais mis en file"
            ),
        },
        "delegated_rlimit_as": {
            "limit_mib": int(memory_limit_mib(args)),
            "limit_gib": int(DELEGATED_RLIMIT_AS_GIB),
            "derivation": DELEGATED_RLIMIT_AS_DERIVATION,
            "scope": (
                "plafond d'espace virtuel du groupe délégué (child + descendants), "
                "posé avant exec ; ne borne PAS la RSS résidente"
            ),
        },
        "delegated_group_rss_limit": {
            "limit_gib": int(DELEGATED_GROUP_RSS_MAX_GIB),
            "scope": (
                "plafond RSS cumulée du groupe délégué (child + descendants), "
                "échantillonnée à 1 Hz ; dépassement => kill du groupe, "
                "BudgetExceeded.kind = 'memory'"
            ),
        },
        # R-J18R-19 (A5) : le plafond global est vérifié par l'horloge injectée
        # **avant** l'écriture de `results.json` ; la garde est consignée.
        "budget_time_guarded_before_write": True,
    }
    # R-J18R-14 : temps **par bras** publiés (bras requis vs N6 vs traitements finaux).
    arm_timings = {
        arm: {
            "elapsed_seconds": float(arms[arm].get("elapsed_seconds", 0.0)),
            "status": str(arms[arm].get("status", "completed")),
            "n_observations": int(arms[arm].get("n_observations", len(arms[arm]["records"]))),
        }
        for arm in arms
    }
    arm_timings["final_treatments"] = {
        "elapsed_seconds": float(time.monotonic() - final_started),
        "status": "completed",
        "n_observations": 0,
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
            # R-J18R-17 : le bras de contrôle ne tourne **que** sur σ = 0 (grille figée §2.5).
            "control_noise_levels": list(CONTROL_NOISE_LEVELS),
            "mask_retention": MASK_RETAIN_PROBABILITY,
            "configs": {arm: [list(config) for config in configs] for arm, configs in ARM_CONFIGS.items()},
            "replay_fraction": fraction,
            "replay_modulo": replay_stats.modulo,
            "replay_tolerance": replay_stats.tolerance,
            "canonical_command": CANONICAL_COMMAND,
        },
        "runtime": runtime,
        "numerical_integrity": dict(
            integrity.to_dict(),
            replay=replay_stats.to_dict(),
            # R-J18R-15 : portée déclarée — C5 n'observe **ni** N6 **ni** les bras délégués.
            observed_arms=list(criteria_shell["C5"]["observed_arms"]),
            excluded_arms=list(criteria_shell["C5"]["excluded_arms"]),
        ),
        "criteria": {
            "C1": c1,
            **{key: value for key, value in criteria_shell.items()},
            "arms_complete": flags["arms_complete"],
        },
        "reserves": reserves,
        "decision": decision,
        "arms": {arm: arms[arm]["publications"] for arm in arms},
        "arm_timings": arm_timings,
        "n6": _n6_status_block(arms.get("n6_chat_template"), n6_c5),
        "training": counter.training_block(criterion_c7(model, counter)["requires_grad_all_false"]),
        "reference_j15": REFERENCE_J15,
    }
    if not flags["arms_complete"]:
        # §7.2 : `results.json` n'est jamais écrit si un bras requis est incomplet.
        return {"shards": produced, "results_written": False, "decision": decision}
    # R-J18R-19 (A5) : le plafond global est vérifié par l'horloge injectée
    # **avant** l'écriture de `results.json`, **après** les sous-processus, **avant**
    # la persistance : le dépassement ne se découvre **pas** après la publication.
    budget.guard_time("écriture de results.json")
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
        verification={
            "criteria": results["criteria"],
            "decision": decision,
            # R-J18R-18 : la portée du plafond RSS est consignée au manifeste (constat D9).
            "budget_scope": runtime["rss_budget_scope"],
            # R-J18R-19 (A8) : le bloc `training` **mesuré** au runtime est consigné
            # au manifeste, avec la garde de temps **avant** écriture.
            "training": results["training"],
            "time_guarded_before_write": bool(runtime["budget_time_guarded_before_write"]),
            # J18R-final : événement du watchdog VRAM (branché), préflight, verrou
            # d'exclusion mutuelle et plafonds délégués — faits mesurés.
            "vram_watchdog": runtime["vram_watchdog"],
            "vram_preflight": runtime["vram_preflight"],
            "measurement_lock": runtime["measurement_lock"],
            "delegated_rlimit_as": runtime["delegated_rlimit_as"],
            "delegated_group_rss_limit": runtime["delegated_group_rss_limit"],
        },
        command=reconstruct_command(list(argv)),
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
            # R-J18R-17 : le bras de contrôle ne tourne **que** sur σ = 0 (§2.5).
            "control_noise_levels": (list(CONTROL_NOISE_LEVELS) if arm == "control" else None),
            "shards": arm_shard_names(arm),
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
            # R-J18R-18 / R-J18R-19 (D9) : portée réelle, consignée dans le plan.
            "rss_budget_scope": (
                "appliqué à la RSS du processus du harnais ; les invocations d'ancrage "
                "§7.3b sont des sous-processus **limités par RLIMIT_AS** (posé avant "
                "exec, child et descendants dans le groupe du harnais) et par un "
                "**plafond RSS de groupe** ; leur mémoire GPU n'est pas couverte par "
                "cette mesure mais l'est par le **watchdog VRAM**"
            ),
            # R-J18R-19 (D9) + J18R-final : plafond RLIMIT_AS **virtuel** du groupe
            # délégué — mesuré, distinct du plafond RSS du processus du harnais.
            "budget_rss_mib": int(args.max_rss_gib * 1024),
            "delegated_memory_limit_mib": memory_limit_mib(args),
            "delegated_memory_limit_derivation": DELEGATED_RLIMIT_AS_DERIVATION,
            "delegated_group_rss_limit_mib": int(DELEGATED_GROUP_RSS_MAX_GIB * 1024),
            "delegated_stop_after_seconds": args.max_seconds,
            # J18R-final : watchdog VRAM branché (seuil dur) + exclusion mutuelle.
            "vram_watchdog_max_used_mib": int(VRAM_WATCHDOG_MAX_USED_MIB),
            "vram_watchdog_derivation": VRAM_WATCHDOG_DERIVATION,
            "vram_preflight_min_free_mib": int(VRAM_PREFLIGHT_MIN_FREE_MIB),
            "measurement_lock": MEASUREMENT_LOCK_RELATIVE_PATH,
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
