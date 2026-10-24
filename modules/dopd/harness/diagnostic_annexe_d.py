#!/usr/bin/env python3
"""Diagnostic Annexe D de d-OPD sur Qwen3-0.6B — inférence seule.

Le protocole est pré-enregistré dans ``modules/dopd/docs/DOPD_SPEC.md`` §6.9.
Ce script construit un support teacher fini, énumère exactement le posterior
conditionné, puis compare la cible causale et la cible future-aware construite
avec **un seul** futur échantillonné.

Ce que le script ne fait jamais : aucun optimizer, aucun backward, aucun
``requires_grad``, aucune mise à jour de poids, aucun rollout de modèle, aucun
benchmark. Le checkpoint teacher est gelé et lu seulement.

Commande GPU prévue :

    HF_HOME="$PWD/_artefact/hf-cache" python \
        modules/dopd/harness/diagnostic_annexe_d.py \
        --model Qwen/Qwen3-0.6B --device cuda --local-files-only \
        --output modules/dopd/eval/annexe_d_qwen3_0.6b/results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_HF_CACHE = _REPO_ROOT / "_artefact" / "hf-cache"
if not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = str(_DEFAULT_HF_CACHE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from academiia.core import sha256_file  # noqa: E402
from academiia.dopd import DOPDConfig  # noqa: E402
from academiia.dopd.diagnostic import (  # noqa: E402
    RestrictedSupport,
    bootstrap_delta_ci,
    evaluate_gate,
    evaluate_multiprompt,
    exact_token_posteriors,
    kl_exact_to_target,
    sample_compatible_index,
    summarize_marginals,
    truncate_causal_gap,
    verify_enumeration_agreement,
)
from academiia.dopd.interfaces import (  # noqa: E402
    ArtifactRef,
    LineageManifest,
    LineageWriter,
)

DEFAULT_PROMPT = "Question: What is the capital of France?\nAnswer:"
NOISE_LEVELS = (0.0, 0.5, 1.0, 1.5)
MASK_RETAIN_PROBABILITY = 0.45
DIAGNOSTIC_SEED = 20261023
BOOTSTRAP_SEED = 20261024
BOOTSTRAP_ITERATIONS = 10_000
PRIMARY_CONFIGURATIONS = ((4, 6), (4, 12), (6, 6))
PRODUCER_SOURCES = (
    "modules/dopd/harness/diagnostic_annexe_d.py",
    "modules/dopd/dopd/diagnostic.py",
)

# J15 : diagnostic multi-prompt (variabilité inter-prompt) et mesure de la
# troncature top-k de la cible d'entraînement.
MULTIPROMPT_SCHEMA = "dopd.prompts/1"
DEFAULT_TOP_K = 16
DEFAULT_TOPK_MAX_MARGINALS = 2_048


class _ManifestBuilder(LineageWriter):
    """Construit le manifeste ADR-002 sans écrire un fichier séparé.

    Le commit consigné est celui vérifié par ``_assert_producer_committed`` :
    il contient effectivement le code producteur.
    """

    def __init__(self, commit: str) -> None:
        self._commit = commit

    def write(self, manifest):
        raise NotImplementedError("le résultat et son manifeste sont écrits ensemble")

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


def _hf_cache_dirs() -> list[Path]:
    candidates = []
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


def _resolve_model_source(model: str) -> str:
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostic Annexe D d-OPD (enumeration exacte, aucun entrainement)."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--blocks", type=int, default=100)
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DIAGNOSTIC_SEED)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output",
        default="modules/dopd/eval/annexe_d_qwen3_0.6b/results.json",
        help="chemin JSON relatif a la racine du depot",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=7200.0,
        help="budget GPU mural maximal (defaut : 7200 s, protocole J14)",
    )
    # J15 : diagnostic multi-prompt (variabilité inter-prompt) et troncature top-k.
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="jeu de prompts JSON (schema dopd.prompts/1) pour un diagnostic "
        "multi-prompt ; absent = mode single-prompt J14",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="mesurer l'ecart de la cible tronquee a k (defaut : 0 = desactive) ; "
        f"valeur usuelle {DEFAULT_TOP_K} (boucle d'entraînement d-OPD)",
    )
    parser.add_argument(
        "--topk-max",
        type=int,
        default=DEFAULT_TOPK_MAX_MARGINALS,
        help="nombre maximal de marginaux mesures pour la troncature top-k ; "
        "borne du cout, independant du nombre total de marginaux",
    )
    return parser.parse_args(argv)


def _suffix_loglikelihoods(
    model,
    prompt_ids: torch.Tensor,
    prefix: tuple[int, ...],
    candidate_ids: np.ndarray,
    suffix: tuple[int, ...],
    device: torch.device,
) -> np.ndarray:
    """Log-vraisemblances du suffixe pour chaque candidat (eq. 26).

    Les branches sont evaluees dans un seul forward : elles ont la meme
    longueur et ne different que par le token candidat a la position ``i``.
    """
    rows = []
    for token_id in candidate_ids:
        branch = torch.tensor(
            prefix + (int(token_id),) + suffix,
            dtype=torch.long,
            device=device,
        )
        rows.append(torch.cat((prompt_ids, branch)))
    input_ids = torch.stack(rows)
    candidate_index = prompt_ids.shape[0] + len(prefix)
    # Le logit a la position du candidat predit le premier token du suffixe.
    positions = list(range(candidate_index, candidate_index + len(suffix)))
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[:, positions, :].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        targets = input_ids[:, candidate_index + 1 : candidate_index + 1 + len(suffix)]
        token_log_probs = log_probs.gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        scores = token_log_probs.sum(dim=-1).cpu().numpy().astype(np.float64)
    del logits, log_probs
    return scores


def _build_restricted_support(
    model,
    prompt_ids: torch.Tensor,
    block_size: int,
    vocab_width: int,
    device: torch.device,
    batch_size: int = 32,
) -> RestrictedSupport:
    """Construit le support fini position par position (Annexe D.1).

    À chaque position, la distribution teacher est la mixture pondérée par la
    probabilité des préfixes restreints. Les ``V`` meilleurs tokens de cette
    mixture définissent le support de la position suivante. Les préfixes sont
    interrogés par lots afin de limiter le nombre de forwards.
    """
    supports: list[np.ndarray] = []
    depth_prefixes: list[tuple[tuple[int, ...], ...]] = []
    restricted_conditionals: list[np.ndarray] = []

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
            # `prefix_log_weights` est deja normalise globalement : la mixture
            # de l'equation 21 ne renormalise pas chaque lot separement.
            weights = np.exp(prefix_log_weights[start : start + len(batch_prefixes)])
            mixture += np.sum(np.exp(batch_log_probs) * weights[:, None], axis=0)
            full_conditionals.append(batch_log_probs)
        conditional = np.concatenate(full_conditionals, axis=0)
        selected = np.sort(np.argsort(-mixture, kind="stable")[:vocab_width]).astype(np.int64)
        restricted_conditional = conditional[:, selected]
        supports.append(selected)
        restricted_conditionals.append(restricted_conditional)

        next_prefixes: list[tuple[int, ...]] = []
        next_log_weights: list[float] = []
        for prefix, prefix_weight, row in zip(
            prefix_candidates,
            prefix_log_weights,
            restricted_conditional,
            strict=True,
        ):
            for candidate_index, token_id in enumerate(selected):
                next_prefixes.append(prefix + (int(token_id),))
                next_log_weights.append(
                    float(prefix_weight + row[candidate_index])
                )
        prefix_candidates = next_prefixes
        prefix_log_weights = np.asarray(next_log_weights, dtype=np.float64)
        prefix_log_weights -= np.logaddexp.reduce(prefix_log_weights)

    return RestrictedSupport(
        supports=tuple(supports),
        prefixes=tuple(depth_prefixes),
        conditional_log_probs=tuple(restricted_conditionals),
    )


def _sample_block(
    support: RestrictedSupport,
    sequences: np.ndarray,
    log_joint: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    index = sample_compatible_index(log_joint, np.ones(sequences.shape[0], dtype=bool), rng)
    return tuple(int(token) for token in sequences[index])


def _masked_positions(
    block_size: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    retained = rng.random(block_size) < MASK_RETAIN_PROBABILITY
    masked = tuple(position for position, keep in enumerate(retained) if not keep)
    if not masked:
        masked = (int(rng.integers(block_size)),)
    return masked


def _causal_and_future_targets(
    support: RestrictedSupport,
    masked_positions: tuple[int, ...],
    future_block: tuple[int, ...],
    model,
    prompt_ids: torch.Tensor,
    device: torch.device,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Calcule les deux cibles de l'Annexe D.3 pour chaque position masquée.

    Le préfixe et le suffixe proviennent tous deux du **bloc compatible
    échantillonné** ``x̂`` : le préfixe ``Â_i`` définit le prior causal (eq. 25)
    et le suffixe ``Ĉ_i`` les scores de compatibilité (eq. 26). La correction
    n'est appliquée que si l'état masqué expose au moins un token visible à
    droite de la position ; sinon les deux cibles coïncident.
    """
    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for position in masked_positions:
        prefix = future_block[:position]
        causal = support.causal_prior(prefix, position)
        position_support = support.supports[position]
        suffix = future_block[position + 1 :]
        has_visible_future = any(
            later > position and later not in masked_positions
            for later in range(support.block_size)
        )
        if not suffix or not has_visible_future:
            results[position] = (causal, causal.copy())
            continue
        candidate_scores_array = _suffix_loglikelihoods(
            model,
            prompt_ids,
            prefix,
            position_support,
            suffix,
            device,
        )
        reference_index = int(np.flatnonzero(position_support == future_block[position])[0])
        deltas = candidate_scores_array - candidate_scores_array[reference_index]
        # Le facteur commun exp(-max(delta)) est neutre apres normalisation ;
        # il evite un debordement sans modifier la cible.
        shifted = deltas - np.max(deltas)
        weights = causal * np.exp(shifted)
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            future = causal.copy()
        else:
            future = weights / total
        results[position] = (causal, future)
    return results


def _rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _assert_producer_committed() -> str:
    """Exige que le code producteur soit présent et intact dans ``HEAD``.

    Le résultat consigne le commit courant : un tiers doit pouvoir retrouver
    exactement le producteur en y faisant un checkout de ce commit. Refuser de
    produire un résultat depuis une version de travail non versionnée — ou
    modifiée après le commit — évite un manifeste qui désigne un commit sans
    le code correspondant.

    Retourne le commit court consigné.
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
                "versionner le code avant d'executer le diagnostic"
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
                "committer les modifications avant d'executer le diagnostic"
            )
    return commit



# Seuil de la proportion de prompts significatifs (J15, critere 6.9.7-3).
# Au moins la moitie des prompts doivent montrer un effet positif significatif.
PROMPT_SIGNIFICANT_THRESHOLD = 0.5


def _load_prompt_set(path: str | os.PathLike[str]) -> list[tuple[str, str]]:
    """Charge un jeu de prompts (schema dopd.prompts/1) : liste de (id, texte)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != MULTIPROMPT_SCHEMA:
        raise ValueError(
            f"schema de fichier de prompts inattendu : {data.get('schema')!r} "
            f"(attendu {MULTIPROMPT_SCHEMA})"
        )
    prompts = data.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("le jeu de prompts doit contenir une liste non vide de prompts")
    jeux: list[tuple[str, str]] = []
    for entry in prompts:
        prompt_id = entry.get("id")
        text = entry.get("text")
        if not isinstance(prompt_id, str) or not isinstance(text, str) or not text:
            raise ValueError(f"prompt {entry!r} invalide : 'id' et 'text' obligatoires")
        jeux.append((prompt_id, text))
    if not jeux:
        raise ValueError("aucun prompt valide dans le fichier")
    return jeux


def _full_causal_at_position(
    model,
    prompt_ids: torch.Tensor,
    prefix: tuple[int, ...],
    device: torch.device,
) -> np.ndarray:
    """Distribution causale complète (vocab entier) a la position suivant ``prefix``.

    Utilise un seul forward : le prior causal a la position suivante est le
    softmax des logits en dernière position du contexte ``prompt_ids + prefix``.
    Cette distribution sert à quantifier l'ecart de la cible tronquee a k.
    """
    context = torch.tensor(list(prefix), dtype=torch.long, device=device)
    input_ids = torch.cat((prompt_ids, context)).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[:, -1, :].float()
        full = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)[0]
    return full


def _process_configuration(
    model,
    prompt_ids: torch.Tensor,
    block_size: int,
    vocab_width: int,
    device: torch.device,
    *,
    seed: int,
    blocks: int,
    started: float,
    max_seconds: float,
    prompt_label: str | None = None,
    topk: dict | None = None,
) -> tuple[list[dict], float, float]:
    """Traite une configuration (N, V) : blocs de base, niveaux de bruit, marginaux.

    Retourne les lignes de marginaux, l'ecart d'enumeration maximal et l'ecart
    de somme de distribution maxima observes. La graine, le nombre de blocs et
    la facon d'echantillonner les blocs sont **identiques** à ceux du diagnostic
    single-prompt J14 : la seule difference est le champ ``prompt`` ajoute quand
    ``prompt_label`` n'est pas None (diagnostic multi-prompt). Quand ``topk`` est
    fourni (borne {k, max, results, count}), les N premiers marginaux vus sont
    mesurés pour l'ecart de la cible tronquee.
    """
    support = _build_restricted_support(model, prompt_ids, block_size, vocab_width, device)
    sequences, log_joint = support.enumerate_joint()
    rng = np.random.default_rng(seed + 1000 * block_size + vocab_width)
    base_blocks: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for block_index in range(blocks):
        block = _sample_block(support, sequences, log_joint, rng)
        masked = _masked_positions(block_size, rng)
        base_blocks.append((block, masked))

    config_rows: list[dict] = []
    numerical_max_error = 0.0
    max_distribution_error = 0.0
    topk_count = 0
    for noise in NOISE_LEVELS:
        perturbation_rng = np.random.default_rng(
            seed + 100_000 * block_size + 10_000 * vocab_width + int(noise * 10)
        )
        perturbation = perturbation_rng.normal(size=sequences.shape[0])
        perturbed_log = log_joint + noise * perturbation
        noise_rng = np.random.default_rng(
            seed + 1000 * block_size + vocab_width + int(noise * 10)
        )
        for block_index, (block, masked) in enumerate(base_blocks):
            compatible = np.ones(sequences.shape[0], dtype=bool)
            for position, token_id in enumerate(block):
                if position not in masked:
                    compatible &= sequences[:, position] == token_id
            future_index = sample_compatible_index(perturbed_log, compatible, noise_rng)
            future_block = tuple(int(token) for token in sequences[future_index])

            visible = {
                position: block[position]
                for position in range(block_size)
                if position not in masked
            }
            exact = exact_token_posteriors(
                sequences, log_joint, visible=visible, supports=support.supports
            )
            if block_index == 0:
                agreement = verify_enumeration_agreement(
                    sequences, log_joint, visible, support.supports
                )
                numerical_max_error = max(numerical_max_error, agreement)
            targets = _causal_and_future_targets(
                support, masked, future_block, model, prompt_ids, device
            )
            for position in masked:
                causal, future = targets[position]
                exact_posterior = exact[position]
                max_distribution_error = max(
                    max_distribution_error,
                    abs(float(np.sum(causal)) - 1.0),
                    abs(float(np.sum(future)) - 1.0),
                    abs(float(np.sum(exact_posterior)) - 1.0),
                )
                causal_kl = kl_exact_to_target(exact_posterior, causal)
                future_kl = kl_exact_to_target(exact_posterior, future)
                visible_future = any(
                    later > position and later not in masked for later in range(block_size)
                )
                record: dict = {
                    "config_index": f"N{block_size}_V{vocab_width}",
                    "block_index": block_index,
                    "position": position,
                    "noise": noise,
                    "causal_kl": causal_kl,
                    "future_kl": future_kl,
                    "delta_kl": causal_kl - future_kl,
                    "visible_future": visible_future,
                }
                if prompt_label is not None:
                    record["prompt"] = prompt_label
                config_rows.append(record)
                if topk is not None and topk_count < topk["max"]:
                    full = _full_causal_at_position(
                        model, prompt_ids, future_block[:position], device
                    )
                    gap = truncate_causal_gap(full, topk["k"])
                    gap["prompt"] = prompt_label
                    topk["results"].append(gap)
                    topk_count += 1
            elapsed = time.time() - started
            if elapsed > max_seconds:
                raise RuntimeError(
                    f"budget GPU maximal depasse ({elapsed:.1f}s > {max_seconds:.1f}s)"
                )
    return config_rows, numerical_max_error, max_distribution_error


def _run_multiprompt(args, model, tokenizer, device: torch.device) -> int:
    """Diagnostic Annexe D multi-prompt (J15) : variabilité inter-prompt + troncature top-k.

    Chaque prompt construit son propre support restreint ; les marginaux portent
    le label du prompt. Agrégat per-prompt, médiane, test des signes, et (optionnel)
    écart de la cible tronquée à k. Inférence seule : aucun entraînement. Le format
    du résultat reprend celui du diagnostic single-prompt avec une section
    ``multiprompt`` supplementaire ; il n'affecte pas le chemin J14.
    """
    started = time.time()
    prompts = _load_prompt_set(args.prompt_file)
    topk_enabled = args.top_k > 0
    topk: dict | None = (
        {"k": args.top_k, "max": args.topk_max, "results": [], "count": 0}
        if topk_enabled
        else None
    )

    print("[1/4] chargement des prompts", flush=True)
    records_all: list[dict] = []
    config_summaries: list[dict] = []
    numerical_max_error = 0.0
    max_distribution_error = 0.0

    for block_size, vocab_width in PRIMARY_CONFIGURATIONS:
        config_started = time.time()
        print(f"[2/4] configuration N={block_size}, V={vocab_width}", flush=True)
        config_rows: list[dict] = []
        for prompt_id, prompt_text in prompts:
            prompt_ids = tokenizer(prompt_text, return_tensors="pt").to(device)["input_ids"][0]
            rows_block, nme, mde = _process_configuration(
                model,
                prompt_ids,
                block_size,
                vocab_width,
                device,
                seed=args.seed,
                blocks=args.blocks,
                started=started,
                max_seconds=args.max_seconds,
                prompt_label=prompt_id,
                topk=topk,
            )
            numerical_max_error = max(numerical_max_error, nme)
            max_distribution_error = max(max_distribution_error, mde)
            config_rows.extend(rows_block)
        records_all.extend(config_rows)
        config_summary = _summarize_config(block_size, vocab_width, config_rows)
        config_summary["bootstrap_ci95_delta_kl"] = list(
            bootstrap_delta_ci(
                config_rows,
                args.bootstrap_seed,
                args.bootstrap_iterations,
                group_fields=("prompt", "block_index"),
            )
        )
        for noise, summary in config_summary["by_noise"].items():
            noise_records = [row for row in config_rows if str(row["noise"]) == noise]
            summary["bootstrap_ci95_delta_kl"] = list(
                bootstrap_delta_ci(
                    noise_records,
                    args.bootstrap_seed,
                    args.bootstrap_iterations,
                    group_fields=("prompt", "block_index"),
                )
            )
        config_summary["wall_seconds"] = round(time.time() - config_started, 3)
        config_summaries.append(config_summary)

    print("[3/4] aggregation", flush=True)
    agg = evaluate_multiprompt(
        records_all,
        seed=args.bootstrap_seed,
        iterations=args.bootstrap_iterations,
    )
    decision = evaluate_gate(
        records_all,
        seed=args.bootstrap_seed,
        iterations=args.bootstrap_iterations,
        minimum_marginals=2000,
        minimum_blocks_per_configuration=100,
        numerical_max_error=numerical_max_error,
        max_distribution_error=max_distribution_error,
    )

    topk_summary: dict | None = None
    if topk_enabled:
        results = topk["results"]
        if results:
            discarded = np.mean([r["discarded_mass"] for r in results])
            tv = np.mean([r["tv_distance"] for r in results])
            max_disc = max(r["discarded_mass"] for r in results)
            topk_summary = {
                "enabled": True,
                "k": args.top_k,
                "n_marginals_measured": len(results),
                "mean_discarded_mass": float(discarded),
                "max_discarded_mass": float(max_disc),
                "mean_tv_distance": float(tv),
            }
        else:
            topk_summary = {"enabled": True, "k": args.top_k, "n_marginals_measured": 0}

    report = {
        "kind": "dopd_annexe_d_multiprompt",
        "protocol": {
            "spec": "modules/dopd/docs/DOPD_SPEC.md#6.9",
            "multiprompt": True,
            "model": args.model,
            "model_source": _resolve_model_source(args.model),
            "device": args.device,
            "dtype": args.dtype,
            "torch_version": torch.__version__,
            "transformers_version": __import__("transformers").__version__,
            "python_version": platform.python_version(),
            "seed": args.seed,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_iterations": args.bootstrap_iterations,
            "blocks_per_configuration": args.blocks,
            "configurations": [list(item) for item in PRIMARY_CONFIGURATIONS],
            "noise_levels": list(NOISE_LEVELS),
            "mask_retain_probability": MASK_RETAIN_PROBABILITY,
            "prompt_file": str(args.prompt_file),
            "prompts": [{"id": pid, "text": text} for pid, text in prompts],
            "top_k": args.top_k if topk_enabled else None,
            "producer_commit": _assert_producer_committed(),
            "source_sha256": {
                relative: sha256_file(str(_REPO_ROOT / relative))
                for relative in PRODUCER_SOURCES
            },
        },
        "runtime": {
            "wall_seconds": round(time.time() - started, 3),
            "max_rss_mib": round(_rss_mib(), 1),
            "cuda_allocated_peak_mib": (
                round(torch.cuda.max_memory_allocated() / 1024**2, 1)
                if args.device == "cuda"
                else None
            ),
            "cuda_reserved_peak_mib": (
                round(torch.cuda.max_memory_reserved() / 1024**2, 1)
                if args.device == "cuda"
                else None
            ),
        },
        "numerical_integrity": {
            "enumeration_max_error": numerical_max_error,
            "max_distribution_sum_error": max_distribution_error,
            "finite": decision.numeric_integrity_met,
        },
        "gate": {
            "status": decision.status,
            "primary_delta_kl_positive": decision.primary_delta_positive,
            "primary_ci95_excludes_zero": decision.primary_ci_excludes_zero,
            "minimum_marginals_met": decision.minimum_marginals_met,
            "numeric_integrity_met": decision.numeric_integrity_met,
            "reserve_required": decision.reserve_required,
            "reserve_configurations": list(decision.reserve_configurations),
            "reserve_noise_levels": list(decision.reserve_noise_levels),
        },
        "multiprompt": {
            "aggregate_delta_kl": agg["mean_delta_kl"],
            "aggregate_ci95_delta_kl": agg["aggregate_ci95_delta_kl"],
            "median_delta_kl": agg["median_delta_kl"],
            "sign_test": agg["sign_test"],
            "n_prompts": agg["n_prompts"],
            "n_prompts_ci_excludes_zero": agg["n_prompts_ci_excludes_zero"],
            "proportion_prompts_significant": agg["proportion_prompts_significant"],
            "per_prompt": agg["per_prompt"],
            "prompt_ci95": agg["prompt_ci95"],
            "decision": {
                "aggregate_ci_excludes_zero": bool(
                    (agg["aggregate_ci95_delta_kl"][0] > 0.0)
                    or (agg["aggregate_ci95_delta_kl"][1] < 0.0)
                ),
                "median_positive": bool(agg["median_delta_kl"] > 0.0),
                "prompts_meet_threshold": bool(
                    agg["proportion_prompts_significant"] >= PROMPT_SIGNIFICANT_THRESHOLD
                ),
            },
            "top_k": topk_summary,
        },
        "configurations": config_summaries,
        "rows": records_all,
        "training": {
            "performed": False,
            "optimizer": None,
            "backward": False,
            "weights_updated": False,
            "requires_grad_all_false": all(not p.requires_grad for p in model.parameters()),
        },
        "lineage": _ManifestBuilder(_assert_producer_committed()).build_manifest(
            artifact=ArtifactRef(path=args.output, kind="eval", hashes={}),
            parent=ArtifactRef(path=args.model, kind="checkpoint", hashes={}),
            config=LineageWriter.config_block(
                DOPDConfig(),
                diagnostic_seed=args.seed,
                bootstrap_seed=args.bootstrap_seed,
                model_id=args.model,
                dtype=args.dtype,
                device=args.device,
                prompt="multi-prompt",
                blocks_per_configuration=args.blocks,
                configurations=[list(item) for item in PRIMARY_CONFIGURATIONS],
                noise_levels=list(NOISE_LEVELS),
                mask_retain_probability=MASK_RETAIN_PROBABILITY,
                bootstrap_iterations=args.bootstrap_iterations,
            ),
            data={
                "teacher": args.model,
                "support_construction": "appendix_d_sequential_restricted",
                "prompt_file": str(args.prompt_file),
                "source_sha256": {
                    relative: sha256_file(str(_REPO_ROOT / relative))
                    for relative in PRODUCER_SOURCES
                },
            },
            verification={
                "kind": "annexe_d_multiprompt",
                "gate_status": decision.status,
                "primary_ci95": list(agg["aggregate_ci95_delta_kl"]),
                "numerical_max_error": numerical_max_error,
                "max_distribution_sum_error": max_distribution_error,
            },
            command=(
                "python modules/dopd/harness/diagnostic_annexe_d.py "
                f"--model {args.model} --device {args.device} --dtype {args.dtype} "
                f"--prompt-file {args.prompt_file} --blocks {args.blocks} "
                f"--seed {args.seed} --bootstrap-seed {args.bootstrap_seed} "
                f"--local-files-only --output {args.output}"
                + (f" --top-k {args.top_k}" if topk_enabled else "")
            ),
        ).to_dict(),
    }

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload + "\n", encoding="utf-8")
    print(f"resultat ecrit : {output_path}")
    print(
        f"porte multi-prompt : delta KL agrge={agg['mean_delta_kl']:.6f} ; "
        f"médiane={agg['median_delta_kl']:.6f} ; "
        f"IC95={agg['aggregate_ci95_delta_kl']} ; "
        f"prompts significatifs={agg['n_prompts_ci_excludes_zero']}/{agg['n_prompts']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.time()
    producer_commit = _assert_producer_committed()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERREUR : CUDA demande mais indisponible", file=sys.stderr)
        return 2
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    source = _resolve_model_source(args.model)

    print(f"[1/5] chargement {args.model} ({args.dtype}, {args.device})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        source, dtype=dtype, local_files_only=args.local_files_only
    )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # J15 : mode multi-prompt (variabilité inter-prompt + troncature top-k).
    # Ce branchement empêche tout contact avec le chemin single-prompt J14,
    # dont la sortie reste byte-à-byte identique.
    if args.prompt_file is not None:
        return _run_multiprompt(args, model, tokenizer, device)

    prompt_ids = tokenizer(args.prompt, return_tensors="pt").to(device)["input_ids"][0]
    rows: list[dict] = []
    config_summaries: list[dict] = []
    numerical_max_error = 0.0
    max_distribution_error = 0.0

    for block_size, vocab_width in PRIMARY_CONFIGURATIONS:
        config_started = time.time()
        print(f"[2/5] configuration N={block_size}, V={vocab_width}", flush=True)
        support = _build_restricted_support(
            model, prompt_ids, block_size, vocab_width, device
        )
        sequences, log_joint = support.enumerate_joint()
        rng = np.random.default_rng(args.seed + 1000 * block_size + vocab_width)
        base_blocks: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for block_index in range(args.blocks):
            block = _sample_block(support, sequences, log_joint, rng)
            masked = _masked_positions(block_size, rng)
            base_blocks.append((block, masked))

        config_rows: list[dict] = []
        for noise in NOISE_LEVELS:
            print(f"      sigma={noise}", flush=True)
            # Equation 30 : un bruit Z_X par bloc complet du support, tire une
            # seule fois par configuration et par niveau sigma, puis reutilise
            # pour tous les etats de base.
            perturbation_rng = np.random.default_rng(
                args.seed
                + 100_000 * block_size
                + 10_000 * vocab_width
                + int(noise * 10)
            )
            perturbation = perturbation_rng.normal(size=sequences.shape[0])
            perturbed_log = log_joint + noise * perturbation
            noise_rng = np.random.default_rng(
                args.seed
                + 1000 * block_size
                + vocab_width
                + int(noise * 10)
            )
            for block_index, (block, masked) in enumerate(base_blocks):
                # Les positions masquees restent libres ; seules les positions
                # visibles de l'etat contraignent la completion compatible.
                compatible = np.ones(sequences.shape[0], dtype=bool)
                for position, token_id in enumerate(block):
                    if position not in masked:
                        compatible &= sequences[:, position] == token_id
                future_index = sample_compatible_index(perturbed_log, compatible, noise_rng)
                future_block = tuple(int(token) for token in sequences[future_index])

                visible = {
                    position: block[position]
                    for position in range(block_size)
                    if position not in masked
                }
                exact = exact_token_posteriors(
                    sequences, log_joint, visible=visible, supports=support.supports
                )
                # Controle d'integrite : l'enumeration de reference, plus lente
                # mais independante, est confrontee sur un petit cas par
                # configuration et par niveau de bruit (criteres J14 6.9.3).
                if block_index == 0:
                    agreement = verify_enumeration_agreement(
                        sequences, log_joint, visible, support.supports
                    )
                    numerical_max_error = max(numerical_max_error, agreement)
                targets = _causal_and_future_targets(
                    support,
                    masked,
                    future_block,
                    model,
                    prompt_ids,
                    device,
                )
                for position in masked:
                    causal, future = targets[position]
                    exact_posterior = exact[position]
                    max_distribution_error = max(
                        max_distribution_error,
                        abs(float(np.sum(causal)) - 1.0),
                        abs(float(np.sum(future)) - 1.0),
                        abs(float(np.sum(exact_posterior)) - 1.0),
                    )
                    causal_kl = kl_exact_to_target(exact_posterior, causal)
                    future_kl = kl_exact_to_target(exact_posterior, future)
                    # Un futur visible existe a droite de la position si au moins
                    # une position ulterieure est conservee (non masquee).
                    visible_future = any(
                        later > position and later not in masked for later in range(block_size)
                    )
                    config_rows.append(
                        {
                            "config_index": f"N{block_size}_V{vocab_width}",
                            "block_index": block_index,
                            "position": position,
                            "noise": noise,
                            "causal_kl": causal_kl,
                            "future_kl": future_kl,
                            "delta_kl": causal_kl - future_kl,
                            "visible_future": visible_future,
                        }
                    )
                elapsed = time.time() - started
                if elapsed > args.max_seconds:
                    raise RuntimeError(
                        f"budget GPU maximal depasse ({elapsed:.1f}s > {args.max_seconds:.1f}s)"
                    )
        rows.extend(config_rows)
        config_summary = _summarize_config(block_size, vocab_width, config_rows)
        config_summary["bootstrap_ci95_delta_kl"] = list(
            bootstrap_delta_ci(
                config_rows,
                args.bootstrap_seed,
                args.bootstrap_iterations,
                group_fields=("block_index",),
            )
        )
        for noise, summary in config_summary["by_noise"].items():
            noise_records = [row for row in config_rows if str(row["noise"]) == noise]
            summary["bootstrap_ci95_delta_kl"] = list(
                bootstrap_delta_ci(
                    noise_records,
                    args.bootstrap_seed,
                    args.bootstrap_iterations,
                    group_fields=("block_index",),
                )
            )
        config_summary["wall_seconds"] = round(time.time() - config_started, 3)
        config_summaries.append(config_summary)

    all_records = rows
    aggregate = _summarize_config(None, None, all_records)
    primary_ci = list(
        bootstrap_delta_ci(all_records, args.bootstrap_seed, args.bootstrap_iterations)
    )
    decision = evaluate_gate(
        all_records,
        seed=args.bootstrap_seed,
        iterations=args.bootstrap_iterations,
        minimum_marginals=2000,
        minimum_blocks_per_configuration=100,
        numerical_max_error=numerical_max_error,
        max_distribution_error=max_distribution_error,
    )
    gate_status = decision.status

    checkpoint_hashes: dict[str, str] = {}
    snapshot = Path(source)
    if snapshot.is_dir():
        for name in ("config.json", "model.safetensors"):
            candidate = snapshot / name
            if candidate.is_file():
                checkpoint_hashes[name] = sha256_file(str(candidate))
    # Empreintes des sources productrices : elles permettent de verifier a
    # posteriori que le code execute est celui du commit consigne.
    source_hashes = {
        relative: sha256_file(str(_REPO_ROOT / relative))
        for relative in PRODUCER_SOURCES
    }
    command = (
        "python modules/dopd/harness/diagnostic_annexe_d.py "
        f"--model {args.model} --device {args.device} --dtype {args.dtype} "
        f"--blocks {args.blocks} --seed {args.seed} "
        f"--bootstrap-seed {args.bootstrap_seed} --local-files-only "
        f"--output {args.output}"
    )
    builder = _ManifestBuilder(producer_commit)
    manifest = builder.build_manifest(
        artifact=ArtifactRef(path=args.output, kind="eval", hashes={}),
        parent=ArtifactRef(path=args.model, kind="checkpoint", hashes=checkpoint_hashes),
        config=LineageWriter.config_block(
            DOPDConfig(),
            diagnostic_seed=args.seed,
            bootstrap_seed=args.bootstrap_seed,
            model_id=args.model,
            dtype=args.dtype,
            device=args.device,
            prompt=args.prompt,
            blocks_per_configuration=args.blocks,
            configurations=[list(item) for item in PRIMARY_CONFIGURATIONS],
            noise_levels=list(NOISE_LEVELS),
            mask_retain_probability=MASK_RETAIN_PROBABILITY,
            bootstrap_iterations=args.bootstrap_iterations,
        ),
        data={
            "prompt_sha256": _sha256_text(args.prompt),
            "teacher": args.model,
            "support_construction": "appendix_d_sequential_restricted",
            "source_sha256": source_hashes,
        },
        verification={
            "kind": "annexe_d_exact_enumeration",
            "gate_status": gate_status,
            "primary_ci95": list(primary_ci),
            "numerical_max_error": numerical_max_error,
            "max_distribution_sum_error": max_distribution_error,
        },
        command=command,
    )

    report = {
        "kind": "dopd_annexe_d_diagnostic",
        "protocol": {
            "spec": "modules/dopd/docs/DOPD_SPEC.md#6.9",
            "prompt": args.prompt,
            "model": args.model,
            "model_source": source,
            "device": args.device,
            "dtype": args.dtype,
            "torch_version": torch.__version__,
            "transformers_version": __import__("transformers").__version__,
            "python_version": platform.python_version(),
            "seed": args.seed,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_iterations": args.bootstrap_iterations,
            "blocks_per_configuration": args.blocks,
            "configurations": [list(item) for item in PRIMARY_CONFIGURATIONS],
            "noise_levels": list(NOISE_LEVELS),
            "mask_retain_probability": MASK_RETAIN_PROBABILITY,
            "producer_commit": producer_commit,
            "source_sha256": source_hashes,
        },
        "runtime": {
            "wall_seconds": round(time.time() - started, 3),
            "max_rss_mib": round(_rss_mib(), 1),
            "cuda_allocated_peak_mib": (
                round(torch.cuda.max_memory_allocated() / 1024**2, 1)
                if args.device == "cuda"
                else None
            ),
            "cuda_reserved_peak_mib": (
                round(torch.cuda.max_memory_reserved() / 1024**2, 1)
                if args.device == "cuda"
                else None
            ),
        },
        "numerical_integrity": {
            "enumeration_max_error": numerical_max_error,
            "max_distribution_sum_error": max_distribution_error,
            "finite": decision.numeric_integrity_met,
        },
        "gate": {
            "status": gate_status,
            "primary_delta_kl_positive": decision.primary_delta_positive,
            "primary_ci95_excludes_zero": decision.primary_ci_excludes_zero,
            "minimum_marginals_met": decision.minimum_marginals_met,
            "numeric_integrity_met": decision.numeric_integrity_met,
            "reserve_required": decision.reserve_required,
            "reserve_configurations": list(decision.reserve_configurations),
            "reserve_noise_levels": list(decision.reserve_noise_levels),
            "blocks_per_configuration": decision.blocks_per_configuration,
        },
        "aggregate": {
            **aggregate,
            "bootstrap_ci95_delta_kl": list(primary_ci),
        },
        "configurations": config_summaries,
        "rows": all_records,
        "training": {
            "performed": False,
            "optimizer": None,
            "backward": False,
            "weights_updated": False,
            "requires_grad_all_false": all(not p.requires_grad for p in model.parameters()),
        },
        "lineage": manifest.to_dict(),
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload + "\n", encoding="utf-8")
    print(f"resultat ecrit : {output_path}")
    print(f"porte : {gate_status}; delta KL={aggregate['mean_delta_kl']:.6f}; IC95={primary_ci}")
    return 0


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _summarize_config(
    block_size: int | None,
    vocab_width: int | None,
    records: list[dict],
) -> dict:
    by_noise = {
        str(noise): summarize_marginals([row for row in records if row["noise"] == noise])
        for noise in NOISE_LEVELS
    }
    return {
        "block_size": block_size,
        "vocab_width": vocab_width,
        **summarize_marginals(records),
        "by_noise": by_noise,
    }


if __name__ == "__main__":
    raise SystemExit(main())
