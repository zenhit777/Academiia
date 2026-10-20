#!/usr/bin/env python3
"""Empreinte des items évalués — preuve d'identité des jeux (J6bis D2).

Pour que deux évaluations soient comparables, il ne suffit pas qu'elles
utilisent « la même tâche » : il faut la **preuve** que les mêmes items ont été
évalués, dans le même ordre.

Ce script lit les `samples_*.jsonl` d'un run (fork lm-eval de Dream) et
calcule :

* `n_items`            : nombre d'items distincts ;
* `items_sha256`       : SHA-256 de la liste triée des couples
  `(doc_id, doc_hash)` — c'est l'**identité du jeu d'items** ;
* `items_order_sha256` : SHA-256 de la liste **dans l'ordre d'évaluation** ;
* `prompts_sha256`     : SHA-256 des `prompt_hash` (le prompt exact envoyé au
  modèle, après chat template) ;
* `targets_sha256`     : SHA-256 des `target_hash` ;
* `config`             : le bloc `config` du `results_*.json` (protocole).

Deux runs sont **comparables** si et seulement si `items_sha256`,
`prompts_sha256` et `config` sont identiques. C'est le contrôle demandé par le
protocole G1 (`docs/design/G1_PROTOCOL.md`).

Usage :
    python scripts/eval_item_fingerprint.py --root eval/dream_a2d
    python scripts/eval_item_fingerprint.py --run eval/dream_a2d/dream_7b_base/gsm8k_cot_a2d
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path


def _sha256_list(values) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(str(value).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def fingerprint(run_dir: Path) -> dict:
    samples = sorted(glob.glob(str(run_dir / "samples_*.jsonl")))
    if not samples:
        return {"run": str(run_dir), "error": "aucun samples_*.jsonl"}
    items = []
    prompts = []
    targets = []
    # Tous les fichiers d'echantillons : un run MATH contient un fichier par
    # sous-tache minerva_math_* (175 items au total).
    for sample_file in samples:
        with open(sample_file) as f:
            for line in f:
                record = json.loads(line)
                key = (record.get("doc_id"), record.get("doc_hash"))
                if key not in items:
                    items.append(key)
                if record.get("prompt_hash") and record["prompt_hash"] not in prompts:
                    prompts.append(record["prompt_hash"])
                if record.get("target_hash") and record["target_hash"] not in targets:
                    targets.append(record["target_hash"])
    results = sorted(glob.glob(str(run_dir / "results_*.json")))
    config = {}
    n_samples = {}
    git_hash = None
    if results:
        with open(results[0]) as f:
            data = json.load(f)
        config = data.get("config", {})
        n_samples = data.get("n-samples", {})
        git_hash = data.get("git_hash")
    return {
        "run": str(run_dir),
        "n_items": len(items),
        "items_sha256": _sha256_list(sorted(f"{a}:{b}" for a, b in items)),
        "items_order_sha256": _sha256_list(f"{a}:{b}" for a, b in items),
        "prompts_sha256": _sha256_list(sorted(prompts)),
        "targets_sha256": _sha256_list(sorted(targets)),
        "n_samples": n_samples,
        "git_hash": git_hash,
        "config": config,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="eval/dream_a2d",
                        help="racine des resultats (<root>/<modele>/<tache>)")
    parser.add_argument("--run", default=None,
                        help="un seul dossier de run")
    parser.add_argument("--out", default=None, help="fichier JSON de sortie")
    args = parser.parse_args()

    if args.run:
        runs = [Path(args.run)]
    else:
        root = Path(args.root)
        runs = sorted(p for p in root.glob("*/*") if p.is_dir())

    reports = []
    for run_dir in runs:
        report = fingerprint(run_dir)
        reports.append(report)
        if "error" in report:
            print(f"{run_dir}: {report['error']}")
            continue
        print(f"{run_dir}")
        print(f"    n_items          = {report['n_items']}  {report['n_samples']}")
        print(f"    items_sha256     = {report['items_sha256']}")
        print(f"    prompts_sha256   = {report['prompts_sha256']}")
        print(f"    config           = {json.dumps(report['config'], sort_keys=True)}")

    payload = {"runs": reports}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nJSON ecrit : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
