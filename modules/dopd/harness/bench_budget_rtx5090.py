#!/usr/bin/env python3
"""Micro-benchmarks FORWARD ONLY pour le chiffrage du budget RTX 5090 (J14).

Ce script mesure quatre grandeurs qui bornent un entraînement d-OPD local :

- le débit de **préremplissage** (tokens/s) ;
- le débit de **décodage** avec cache KV, par taille de lot et longueur de
  contexte, et la taille du cache KV (analytique et mesurée) ;
- le débit de **scoring professeur** : un forward de ``k`` branches en
  parallèle partageant le même préfixe, qui est le poste dominant du budget ;
- le coût du **scoring avec cache de préfixe partagé** (App. B), comparé à
  géométrie égale au scoring sans cache : c'est le facteur qui décide si un
  protocole d'entraînement local est réaliste.

Ce que le script ne fait jamais : aucun backward, aucun optimizer, aucun
``requires_grad``, aucune mise à jour de poids, aucun rollout de modèle. Le
checkpoint teacher est gelé et lu seulement.

Les identifiants de token sont tirés au hasard **avec une graine explicite** :
ces mesures sont des débits, pas des résultats scientifiques ; la graine rend
la re-exécution reproductible, elle ne transforme pas la mesure en expérience.

Commande (RTX 5090, cache Hugging Face redirigé dans le dépôt) :

    HF_HOME="$PWD/_artefact/hf-cache" python \\
        modules/dopd/harness/bench_budget_rtx5090.py \\
        --model Qwen/Qwen3-0.6B --seed 20261023 \\
        --output modules/dopd/eval/budget_rtx5090/bench_rtx5090.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_HF_CACHE = _REPO_ROOT / "_artefact" / "hf-cache"
if not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = str(_DEFAULT_HF_CACHE)

import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

PREFILL_LENGTHS = (512, 2048, 4096)
DECODE_CONTEXTS = (1024, 4096)
DECODE_BATCHES = (1, 4, 8, 16, 32, 64)
DECODE_TOKENS = 32
BRANCH_SEQUENCE_LENGTHS = (512, 1024, 2048, 4096)
BRANCH_COUNTS = (16, 32, 64, 128)
BRANCH_REPETITIONS = 2
PREFIX_CACHE_CONTEXTS = (256, 512, 1024, 4096)
# Suffixes de bloc d-OPD (N = 4 : le suffixe d'une position masquee fait au
# plus N - 1 tokens) et suffixes longs, pour situer la borne haute.
PREFIX_CACHE_SUFFIXES = (1, 2, 3, 4, 64, 128, 256)
# k = 16 (candidats) et 2k + 1 = 33 (candidats + branche de reference, App. B).
PREFIX_CACHE_BRANCHES = (16, 33)


def sha256_file(path: Path) -> str:
    """Empreinte SHA-256 d'un fichier, en lecture seule."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    """Commit git courant, ou ``None`` hors d'un dépôt."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return completed.stdout.strip() or None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Lit les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


def resolve_model(model: str) -> str:
    """Retrouve le checkpoint local, en privilégiant le cache du dépôt."""
    hub = Path(os.environ["HF_HOME"]) / "hub"
    fallback = Path.home() / ".cache/huggingface/hub"
    for root in (hub, fallback):
        snapshots = root / f"models--{model.replace('/', '--')}" / "snapshots"
        if snapshots.is_dir():
            for snapshot in sorted(
                snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
            ):
                if (snapshot / "config.json").is_file():
                    return str(snapshot)
    return model


def kv_cache_mib(config: object, batch_size: int, sequence_length: int) -> float:
    """Taille analytique du cache KV (K et V) en MiB."""
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    elements = (
        2
        * config.num_hidden_layers
        * batch_size
        * sequence_length
        * config.num_key_value_heads
        * head_dim
    )
    return elements * 2 / 1024**2


def freeze_model(model: object) -> None:
    """Gèle le modèle : mode ``eval`` et aucun gradient activé.

    Le budget est mesuré en inférence seule ; cette fonction rend cet
    invariant explicite et vérifiable.
    """
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def measure_prefill(model: object, seed: int) -> list[dict]:
    """Mesure le débit de préremplissage (batch 1) et le pic d'allocation."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    entries: list[dict] = []
    for length in PREFILL_LENGTHS:
        ids = torch.randint(
            1000, 100000, (1, length), dtype=torch.long, device="cuda", generator=generator
        )
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(input_ids=ids, logits_to_keep=1)
            torch.cuda.synchronize()
            start = time.perf_counter()
            repetitions = 3
            for _ in range(repetitions):
                model(input_ids=ids, logits_to_keep=1)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        entries.append(
            {
                "sequence_length": length,
                "repetitions": repetitions,
                "seconds": round(elapsed, 4),
                "tokens_per_second": round(repetitions * length / elapsed, 1),
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
            }
        )
        del ids
        torch.cuda.empty_cache()
    return entries


def measure_decode(model: object, config: object, seed: int, weights_mib: float) -> list[dict]:
    """Mesure le débit de décodage avec cache KV et la taille du cache."""
    generator = torch.Generator(device="cuda").manual_seed(seed + 1)
    entries: list[dict] = []
    for context_length in DECODE_CONTEXTS:
        for batch_size in DECODE_BATCHES:
            torch.cuda.reset_peak_memory_stats()
            prompt = torch.randint(
                1000,
                100000,
                (batch_size, context_length),
                dtype=torch.long,
                device="cuda",
                generator=generator,
            )
            entry: dict = {
                "context_length": context_length,
                "batch_size": batch_size,
                "tokens": DECODE_TOKENS,
                "kv_cache_mib_theory": round(kv_cache_mib(config, batch_size, context_length), 1),
            }
            try:
                with torch.no_grad():
                    out = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
                    next_token = out.logits[:, -1:].argmax(dim=-1)
                    past = out.past_key_values
                    del out
                    torch.cuda.synchronize()
                    after_prefill = torch.cuda.memory_allocated()
                    start = time.perf_counter()
                    for _ in range(DECODE_TOKENS):
                        out = model(
                            input_ids=next_token,
                            past_key_values=past,
                            use_cache=True,
                            logits_to_keep=1,
                        )
                        past = out.past_key_values
                        next_token = out.logits[:, -1:].argmax(dim=-1)
                        del out
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    after_decode = torch.cuda.memory_allocated()
                entry.update(
                    {
                        "seconds": round(elapsed, 4),
                        "tokens_per_second": round(batch_size * DECODE_TOKENS / elapsed, 1),
                        "kv_cache_mib_measured": round(
                            (after_prefill - weights_mib * 1024**2) / 1024**2, 1
                        ),
                        "allocated_after_decode_mib": round(after_decode / 1024**2, 1),
                        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
                    }
                )
                del past, prompt, next_token
            except torch.cuda.OutOfMemoryError:
                entry["error"] = "OOM"
            torch.cuda.empty_cache()
            entries.append(entry)
    return entries


def measure_teacher_branches(model: object, seed: int) -> list[dict]:
    """Mesure le débit d'un forward de scoring professeur (lot de branches)."""
    generator = torch.Generator(device="cuda").manual_seed(seed + 2)
    entries: list[dict] = []
    for sequence_length in BRANCH_SEQUENCE_LENGTHS:
        for branches in BRANCH_COUNTS:
            ids = torch.randint(
                1000,
                100000,
                (branches, sequence_length),
                dtype=torch.long,
                device="cuda",
                generator=generator,
            )
            torch.cuda.reset_peak_memory_stats()
            entry: dict = {
                "sequence_length": sequence_length,
                "batch_branches": branches,
                "repetitions": BRANCH_REPETITIONS,
            }
            try:
                with torch.no_grad():
                    model(input_ids=ids, logits_to_keep=1)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(BRANCH_REPETITIONS):
                        model(input_ids=ids, logits_to_keep=1)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                entry.update(
                    {
                        "seconds": round(elapsed, 4),
                        "tokens_per_second": round(
                            BRANCH_REPETITIONS * branches * sequence_length / elapsed, 1
                        ),
                        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
                    }
                )
            except torch.cuda.OutOfMemoryError:
                entry["error"] = "OOM"
            del ids
            torch.cuda.empty_cache()
            entries.append(entry)
    return entries


def measure_prefix_cache_scoring(model: object, seed: int) -> list[dict]:
    """Mesure le scoring professeur avec **cache de préfixe partagé**.

    Le préfixe causal (prompt + préfixe du bloc) est calculé **une fois**, puis
    ``batch_repeat_interleave`` duplique le cache sur les branches. C'est
    l'optimisation de l'App. B du papier, dont le coût détermine le budget
    d'entraînement local.

    La géométrie reproduit celle du scoring d-OPD (eq. 11–14, App. B) :

    - le **préfixe causal** ``(q, z, Â_i)`` est partagé par toutes les
      branches ;
    - le **candidat** ``v`` diffère d'une branche à l'autre (token à la
      position masquée ``i``) ;
    - le **suffixe complété** ``Ĉ_i`` est partagé (rollout complété) ;
    - le prior causal ``r_i(v)`` est le logit du dernier token du préfixe ;
    - le score futur ``s_i(v)`` est la somme des log-probabilités du suffixe
      sous le candidat.

    Deux modes sont mesurés sur le **même job** :

    - **avec cache** : préremplissage du préfixe (une fois), duplication du
      cache, puis chaque branche ne calcule que ``[v, Ĉ_i]`` ;
    - **sans cache** : le contexte complet ``(préfixe, v, Ĉ_i)`` est rejoué
      pour chaque branche — c'est le mode de `_suffix_loglikelihoods` du
      diagnostic.

    L'unité de coût est le **job de scoring** (une position masquée × `k`
    candidats) : c'est l'unité que la boucle d'entraînement paie.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed + 3)
    entries: list[dict] = []
    for branches in PREFIX_CACHE_BRANCHES:
        for context_length in PREFIX_CACHE_CONTEXTS:
            for suffix_length in PREFIX_CACHE_SUFFIXES:
                prefix = torch.randint(
                    1000,
                    100000,
                    (1, context_length),
                    dtype=torch.long,
                    device="cuda",
                    generator=generator,
                )
                candidates = torch.randint(
                    1000,
                    100000,
                    (branches, 1),
                    dtype=torch.long,
                    device="cuda",
                    generator=generator,
                )
                # Le suffixe complète est partage : une seule realisation.
                suffix = torch.randint(
                    1000,
                    100000,
                    (1, suffix_length),
                    dtype=torch.long,
                    device="cuda",
                    generator=generator,
                )
                entry: dict = {
                    "context_length": context_length,
                    "suffix_length": suffix_length,
                    "batch_branches": branches,
                }
                try:
                    with torch.no_grad():
                        # --- mode cache : prefill + branches [v, suffixe]
                        # Warmup sur un cache **jetable** : un forward avec
                        # `past_key_values` avance le cache en place, un
                        # warmup partage polluerait la passe mesuree.
                        warm_out = model(input_ids=prefix, use_cache=True, logits_to_keep=1)
                        warm_cache = warm_out.past_key_values
                        del warm_out
                        warm_cache.batch_repeat_interleave(branches)
                        warm_input = torch.cat((candidates, suffix.expand(branches, -1)), dim=1)
                        warm = model(
                            input_ids=warm_input,
                            past_key_values=warm_cache,
                            use_cache=True,
                            logits_to_keep=warm_input.shape[1],
                        )
                        del warm, warm_cache
                        torch.cuda.synchronize()

                        torch.cuda.reset_peak_memory_stats()
                        out = model(input_ids=prefix, use_cache=True, logits_to_keep=1)
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        out = model(input_ids=prefix, use_cache=True, logits_to_keep=1)
                        torch.cuda.synchronize()
                        prefill_seconds = time.perf_counter() - start
                        prior_logits = out.logits[:, -1, :].float()
                        cache = out.past_key_values
                        del out
                        # La duplication du cache sur les branches est un coût
                        # reel (copie du KV cache) : elle fait partie du job.
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        cache.batch_repeat_interleave(branches)
                        torch.cuda.synchronize()
                        duplicate_seconds = time.perf_counter() - start
                        branch_input = torch.cat((candidates, suffix.expand(branches, -1)), dim=1)
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        out = model(
                            input_ids=branch_input,
                            past_key_values=cache,
                            use_cache=True,
                            logits_to_keep=branch_input.shape[1],
                        )
                        torch.cuda.synchronize()
                        branch_seconds = time.perf_counter() - start
                        # log p(suffix | prefix, v) : le logit a la position du
                        # candidat predit le premier token du suffixe.
                        branch_logits = out.logits[:, :suffix_length, :].float()
                        suffix_log_probs = torch.log_softmax(branch_logits, dim=-1)
                        targets = suffix.expand(branches, -1)
                        suffix_scores = (
                            suffix_log_probs.gather(-1, targets.unsqueeze(-1))
                            .squeeze(-1)
                            .sum(dim=-1)
                        )
                        prior_log_probs = torch.log_softmax(prior_logits, dim=-1)
                        # Le prior est calcule une seule fois (batch 1) : il est
                        # elargi aux branches, chacune lisant son candidat.
                        prior_scores = (
                            prior_log_probs.expand(branches, -1).gather(-1, candidates).squeeze(-1)
                        )
                        cache_peak_mib = torch.cuda.max_memory_allocated() / 1024**2
                        del (
                            out,
                            cache,
                            branch_logits,
                            suffix_log_probs,
                            prior_logits,
                            prior_log_probs,
                        )

                        # --- mode sans cache : contexte complet rejoue
                        full = torch.cat(
                            (
                                prefix.expand(branches, -1),
                                candidates,
                                suffix.expand(branches, -1),
                            ),
                            dim=1,
                        )
                        # Il faut S + 1 logits : celui qui predit le candidat
                        # (indice P - 1) et ceux qui predisent le suffixe.
                        keep = suffix_length + 2
                        torch.cuda.reset_peak_memory_stats()
                        model(input_ids=full, logits_to_keep=keep)
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        out = model(input_ids=full, logits_to_keep=keep)
                        torch.cuda.synchronize()
                        nocache_seconds = time.perf_counter() - start
                        full_logits = out.logits[:, -keep:, :].float()
                        full_log_probs = torch.log_softmax(full_logits, dim=-1)
                        # Les logits gardes couvrent les positions P-1 .. P+S :
                        # l'indice 0 predit le candidat (position P), l'indice
                        # j+1 predit le token de suffixe j (position P+j).
                        full_targets = suffix.expand(branches, -1)
                        full_suffix_scores = (
                            full_log_probs[:, 1 : suffix_length + 1, :]
                            .gather(-1, full_targets.unsqueeze(-1))
                            .squeeze(-1)
                            .sum(dim=-1)
                        )
                        full_prior_scores = (
                            full_log_probs[:, 0, :].gather(-1, candidates).squeeze(-1)
                        )
                        nocache_peak_mib = torch.cuda.max_memory_allocated() / 1024**2
                        del out, full, full_logits, full_log_probs
                    entry.update(
                        {
                            "prefill_seconds": round(prefill_seconds, 4),
                            "duplicate_seconds": round(duplicate_seconds, 4),
                            "branch_seconds": round(branch_seconds, 4),
                            "cache_seconds": round(
                                prefill_seconds + duplicate_seconds + branch_seconds, 4
                            ),
                            "nocache_seconds": round(nocache_seconds, 4),
                            "speedup": round(
                                nocache_seconds
                                / (prefill_seconds + duplicate_seconds + branch_seconds),
                                2,
                            ),
                            "cache_peak_allocated_mib": round(cache_peak_mib, 1),
                            "nocache_peak_allocated_mib": round(nocache_peak_mib, 1),
                            # Ecart de coherence entre les deux modes : il
                            # mesure l'ecart BF16 du calcul, pas sa logique.
                            "max_abs_suffix_score_diff": round(
                                float((suffix_scores - full_suffix_scores).abs().max()),
                                6,
                            ),
                            "max_abs_prior_diff": round(
                                float((prior_scores - full_prior_scores).abs().max()),
                                6,
                            ),
                        }
                    )
                    del prefix, candidates, suffix
                except torch.cuda.OutOfMemoryError:
                    entry["error"] = "OOM"
                torch.cuda.empty_cache()
                entries.append(entry)
    return entries


def main(argv: list[str] | None = None) -> int:
    """Exécute les trois campagnes de mesure et écrit le JSON."""
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("ERREUR : CUDA indisponible ; ce benchmark vise une carte NVIDIA", file=os.sys.stderr)
        return 2
    torch.manual_seed(args.seed)
    source = resolve_model(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        source, dtype=torch.bfloat16, local_files_only=args.local_files_only
    )
    model.to("cuda")
    freeze_model(model)

    config = model.config
    torch.cuda.reset_peak_memory_stats()
    weights_mib = torch.cuda.memory_allocated() / 1024**2
    result: dict = {
        "kind": "dopd_budget_benchmark",
        "model": args.model,
        "model_source": source,
        "device": "cuda",
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_mib": round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1),
        "dtype": "bfloat16",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": args.seed,
        "producer_commit": git_commit(),
        "source_sha256": {
            "modules/dopd/harness/bench_budget_rtx5090.py": sha256_file(Path(__file__).resolve())
        },
        "note": (
            "forward only : aucun backward, aucun optimizer, aucun poids modifie ; "
            "mesures de debit, pas resultats scientifiques"
        ),
        "parameters": sum(p.numel() for p in model.parameters()),
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_key_value_heads": config.num_key_value_heads,
        "num_attention_heads": config.num_attention_heads,
        "head_dim": getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
        "vocab_size": config.vocab_size,
        "weights_mib": round(weights_mib, 1),
        "prefill": [],
        "decode": [],
        "teacher_branches": [],
        "prefix_cache_scoring": [],
        "prefix_cache_scoring_fp32_control": [],
    }

    result["prefill"] = measure_prefill(model, args.seed)
    result["decode"] = measure_decode(model, config, args.seed, weights_mib)
    result["teacher_branches"] = measure_teacher_branches(model, args.seed)
    result["prefix_cache_scoring"] = measure_prefix_cache_scoring(model, args.seed)
    result["cuda_reserved_peak_mib"] = round(torch.cuda.max_memory_reserved() / 1024**2, 1)

    # Controle de coherence en FP32 : il distingue un ecart de precision
    # BF16 (attendu, de l'ordre de 1e-1 en log-prob) d'une erreur
    # d'indexation (de l'ordre de 1e+1). Un seul cas est rejoue : la
    # geometrie representative (prefixe 1024, k = 16, suffixe de bloc N=4).
    fp32_model = AutoModelForCausalLM.from_pretrained(
        source, dtype=torch.float32, local_files_only=args.local_files_only
    )
    fp32_model.to("cuda")
    freeze_model(fp32_model)
    saved_contexts = PREFIX_CACHE_CONTEXTS
    saved_suffixes = PREFIX_CACHE_SUFFIXES
    saved_branches = PREFIX_CACHE_BRANCHES
    globals()["PREFIX_CACHE_CONTEXTS"] = (1024,)
    globals()["PREFIX_CACHE_SUFFIXES"] = (3,)
    globals()["PREFIX_CACHE_BRANCHES"] = (16,)
    try:
        result["prefix_cache_scoring_fp32_control"] = measure_prefix_cache_scoring(
            fp32_model, args.seed
        )
    finally:
        globals()["PREFIX_CACHE_CONTEXTS"] = saved_contexts
        globals()["PREFIX_CACHE_SUFFIXES"] = saved_suffixes
        globals()["PREFIX_CACHE_BRANCHES"] = saved_branches
        del fp32_model
        torch.cuda.empty_cache()

    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output is None:
        print(payload)
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(f"resultat ecrit : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
