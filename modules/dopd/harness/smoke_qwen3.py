#!/usr/bin/env python3
"""Smoke de chargement de Qwen3-0.6B (J13, phase 2) — **aucun entraînement**.

Ce script est **hors du paquet pur** `academiia.dopd` : il peut donc utiliser
`torch` et `transformers` (règle ADR-001 : ces dépendances sont interdites
dans `modules/dopd/dopd/`, pas dans les harnais).

Objectif : valider que le chemin de développement retenu par d-OPD — le
professeur AR gelé **Qwen3-0.6B** et l'élève initialisé depuis le même
checkpoint (DOPD_SPEC §6.3) — est **chargeable et utilisable** sur cette
machine, avant toute intégration lourde.

Ce que fait le script, dans l'ordre :

1. charge le **tokenizer** ;
2. charge le **modèle** en ``eval()``, sans gradient ;
3. effectue un **forward minimal** sur quelques tokens ;
4. effectue une **génération triviale** (quelques tokens) ;
5. **échantillonne le prior causal** ``r_i(v)`` sur une position — la
   distribution dont d-OPD a besoin pour la cible future-aware (eq. 11) ;
6. rend un **manifeste de lignée** ADR-002 (``producer.module = "dopd"``).

Ce que le script ne fait **jamais** : aucun optimizer, aucun backward, aucun
``requires_grad``, aucune mise à jour de poids, aucun rollout, aucun
benchmark. Les poids du checkpoint ne sont pas modifiés.

Commande (CPU par défaut) :

    python modules/dopd/harness/smoke_qwen3.py --model Qwen/Qwen3-0.6B

Commande avec GPU (optionnelle, déclarée) :

    python modules/dopd/harness/smoke_qwen3.py --model Qwen/Qwen3-0.6B --device cuda

Sortie : un rapport JSON sur stdout (mémoire mesurée, temps, empreintes) et,
avec ``--output``, un fichier JSON écrit **dans le dépôt** (jamais ailleurs).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

# Regle 9 d'AGENTS.md : aucune ecriture hors du depot. Les caches HuggingFace
# sont donc rediriges vers le depot si l'utilisateur ne les a pas deja.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_HF_CACHE = _REPO_ROOT / "_artefact" / "hf-cache"
if not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = str(_DEFAULT_HF_CACHE)

import torch  # noqa: E402  (import tardif : le script est un harnais, pas la lib)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from academiia.core import sha256_file  # noqa: E402
from academiia.dopd import (  # noqa: E402
    DEVELOPMENT_MODEL_ID,
    DOPDConfig,
    LineageManifest,
)
from academiia.dopd.interfaces import (  # noqa: E402
    ArtifactRef,
    LineageWriter,
)


def _hf_cache_dirs() -> list[Path]:
    """Repertoires de cache HuggingFace consultables, par ordre de priorite."""
    candidates = []
    for env in ("HF_HOME", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(env)
        if value:
            candidates.append(Path(value) / "hub")
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")
    seen = set()
    unique = []
    for path in candidates:
        resolved = str(path)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _resolve_model_source(model: str) -> str:
    """Resout un identifiant HF vers un snapshot local deja telecharge.

    Le smoke doit fonctionner hors ligne quand le checkpoint est deja dans un
    cache HuggingFace. La resolution ne **lit** que le cache : aucune
    ecriture hors du depot n'est faite (regle 9 d'AGENTS.md).

    Renvoie l'identifiant tel quel si aucun snapshot local n'est trouve : le
    chargement se fera alors par le reseau.
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
            (p for p in snapshots.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for snapshot in candidates:
            if (snapshot / "config.json").is_file():
                return str(snapshot)
    return model


class _ManifestBuilder(LineageWriter):
    """Construit le manifeste ADR-002 sans ecrire sur disque.

    Le contrat `LineageWriter` impose `write()` ; ici l'ecriture est volontairement
    faite par l'appelant (le script ecrit un rapport JSON, pas un `lineage.json`
    a cote d'un checkpoint).
    """

    def write(self, manifest: LineageManifest) -> str:
        raise NotImplementedError(
            "le smoke n'ecrit pas de lineage.json : il produit un rapport JSON"
        )


def _rss_mib() -> float:
    """Memoire residente du processus, en MiB (Linux : ru_maxrss est en KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke de chargement Qwen3-0.6B pour d-OPD (aucun entrainement)."
    )
    parser.add_argument(
        "--model",
        default=DEVELOPMENT_MODEL_ID,
        help=f"identifiant HuggingFace du checkpoint (defaut : {DEVELOPMENT_MODEL_ID})",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="device de chargement (defaut : cpu)",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("float32", "bfloat16", "float16"),
        help="dtype des poids (defaut : bfloat16, comme le papier sur Qwen3)",
    )
    parser.add_argument(
        "--prompt",
        default="The capital of France is",
        help="prompt du forward et de la generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="tokens generes (defaut : 8)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="chemin du rapport JSON, relatif a la racine du depot (optionnel)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="interdire le telechargement : utiliser uniquement le cache local",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.time()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERREUR : --device cuda demande mais CUDA n'est pas disponible", file=sys.stderr)
        return 2
    if args.device == "cpu" and torch.cuda.is_available():
        print(
            "NOTE : CUDA est disponible mais --device cpu a ete demande ; "
            "le smoke reste sur CPU (mode par defaut de la mission).",
            file=sys.stderr,
        )

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    source = _resolve_model_source(args.model)

    print(f"[1/6] chargement du tokenizer : {args.model}", flush=True)
    if source != args.model:
        print(f"      snapshot local : {source}", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=args.local_files_only
    )
    tokenizer_seconds = time.time() - t0

    print(f"[2/6] chargement du modele ({args.dtype}, {args.device})", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )
    model.to(device)
    model.eval()
    # Aucun entrainement : les poids sont figes et le graphe est coupe.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_seconds = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())

    print("[3/6] forward minimal", flush=True)
    encoded = tokenizer(args.prompt, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        outputs = model(**encoded)
    forward_seconds = time.time() - t0
    logits = outputs.logits
    if logits.shape[-1] != model.config.vocab_size:
        raise RuntimeError(
            f"taille de vocabulaire inattendue : {logits.shape[-1]} "
            f"!= {model.config.vocab_size}"
        )

    print("[4/6] generation triviale", flush=True)
    t0 = time.time()
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generate_seconds = time.time() - t0
    completion = tokenizer.decode(generated[0], skip_special_tokens=True)

    print("[5/6] echantillonnage du prior causal r_i(v) (eq. 11)", flush=True)
    # Position finale du prompt : le prior causal conditionne sur le prefixe
    # complet. C'est exactement la quantite que la cible future-aware
    # repondere (eq. 15-18).
    with torch.no_grad():
        last_logits = logits[0, -1].float()
    prior = torch.softmax(last_logits, dim=-1)
    top_values, top_ids = torch.topk(prior, k=5)
    prior_sum = float(prior.sum())

    print("[6/6] construction du manifeste ADR-002", flush=True)
    builder = _ManifestBuilder()
    # ADR-002 : les hashes sont des sha256 de FICHIER, recalculables par un
    # tiers. On hache les fichiers de configuration et de poids du snapshot.
    checkpoint_hashes: dict[str, str] = {}
    snapshot = Path(source)
    if snapshot.is_dir():
        for name in ("config.json", "model.safetensors"):
            candidate = snapshot / name
            if candidate.is_file():
                checkpoint_hashes[name] = sha256_file(str(candidate))
    manifest = builder.build_manifest(
        artifact=ArtifactRef(path="_artefact/dopd_smoke", kind="report"),
        parent=ArtifactRef(
            path=args.model, kind="checkpoint", hashes=checkpoint_hashes
        ),
        config=LineageWriter.config_block(
            DOPDConfig(),
            model_id=args.model,
            dtype=args.dtype,
            device=args.device,
        ),
        data={"prompts_sha256": None, "verifier": "none (smoke de chargement)"},
        verification={
            "kind": "load_smoke",
            "forward_ok": True,
            "generate_ok": True,
            "prior_sum": prior_sum,
        },
        command="python modules/dopd/harness/smoke_qwen3.py "
        f"--model {args.model} --device {args.device} --dtype {args.dtype}",
    )

    rss = _rss_mib()
    report = {
        "kind": "dopd_load_smoke",
        "model": args.model,
        "model_source": source,
        "device": args.device,
        "dtype": args.dtype,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "n_parameters": n_params,
        "vocab_size": int(model.config.vocab_size),
        "timings_seconds": {
            "tokenizer": round(tokenizer_seconds, 3),
            "model_load": round(model_seconds, 3),
            "forward": round(forward_seconds, 3),
            "generate": round(generate_seconds, 3),
            "total": round(time.time() - started, 3),
        },
        "memory": {
            "max_rss_mib": round(rss, 1),
            "cuda_allocated_mib": (
                round(torch.cuda.memory_allocated() / 1024**2, 1)
                if args.device == "cuda"
                else None
            ),
        },
        "prompt": args.prompt,
        "completion": completion,
        "new_tokens": int(generated.shape[-1] - encoded["input_ids"].shape[-1]),
        "causal_prior": {
            "sum": prior_sum,
            "argmax_token_id": int(top_ids[0]),
            "argmax_token": tokenizer.decode([int(top_ids[0])]),
            "top5": [
                {
                    "token_id": int(token_id),
                    "token": tokenizer.decode([int(token_id)]),
                    "probability": round(float(value), 6),
                }
                for token_id, value in zip(top_ids, top_values)
            ],
        },
        "training": {
            "performed": False,
            "optimizer": None,
            "backward": False,
            "weights_updated": False,
            "requires_grad_all_false": all(
                not p.requires_grad for p in model.parameters()
            ),
        },
        "lineage": manifest.to_dict(),
    }

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    print("\n=== rapport ===")
    print(payload)

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = _REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
        print(f"\nrapport ecrit : {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
