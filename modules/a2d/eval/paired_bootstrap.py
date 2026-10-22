#!/usr/bin/env python3
"""Écart apparié entre deux évaluations + IC bootstrap apparié (J6bis D2).

Deux évaluations ne sont comparables que si elles portent sur les **mêmes
items**. Ce script :

1. extrait la **correction par item** de chaque run (`samples_*.jsonl` du fork
   lm-eval de Dream) ;
2. apparie les items par `(doc_id, doc_hash)` et **refuse** toute paire si les
   jeux d'items diffèrent ;
3. calcule l'écart de précision, les paires discordantes (type McNemar) et
   l'**intervalle de confiance bootstrap apparié** (percentile 95 %,
   10 000 ré-échantillonnages des items, graine fixée).

Métriques disponibles :

* `exact_match` + `--filter <nom>` : lu tel quel dans les échantillons
  (GSM8K : `strict-match`, `flexible-extract` ; MATH : `none` = Minerva) ;
* `math_verify` : **recalculé** avec la fonction `process_results` du fork
  lui-même (`lm_eval.tasks.minerva_math.utils`), donc avec exactement la même
  logique de normalisation et de vérification symbolique.

Prérequis pour `math_verify` : `math_verify`, `sympy`, `antlr4-python3-runtime`
(venv du harnais). Le script s'exécute donc avec
`harness/dream/.venv/bin/python scripts/paired_bootstrap.py ...`.

Usage :
    harness/dream/.venv/bin/python scripts/paired_bootstrap.py \
        --run-a eval/dream_a2d/dream_7b_base/gsm8k_cot_a2d \
        --run-b eval/dream_a2d/dream_7b_instruct/gsm8k_cot_a2d \
        --metric exact_match --filter flexible-extract
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FORK = REPO_ROOT / "harness" / "dream" / ".fork" / "eval_instruct"


def _load_process_results():
    """Charge `process_results` du fork (logique MATH du papier)."""
    if str(FORK) not in sys.path:
        sys.path.insert(0, str(FORK))
    from lm_eval.tasks.minerva_math.utils import process_results
    return process_results


def per_item(run_dir: Path, metric: str, filter_name: str | None):
    """Retourne {item_key: correct} pour un run."""
    samples = sorted(glob.glob(str(run_dir / "samples_*.jsonl")))
    if not samples:
        raise SystemExit(f"ERREUR: aucun samples_*.jsonl dans {run_dir}")
    process_results = _load_process_results() if metric == "math_verify" else None
    out = {}
    for sample_file in samples:
        with open(sample_file) as f:
            for line in f:
                record = json.loads(line)
                if filter_name is not None and record.get("filter") != filter_name:
                    continue
                key = (record.get("doc_id"), record.get("doc_hash"))
                if metric == "math_verify":
                    results = process_results(record["doc"], record.get("filtered_resps") or [])
                    value = results.get(metric, 0)
                else:
                    value = record.get("exact_match", 0)
                out[key] = 1 if float(value) == 1.0 else 0
    return out


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math_sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci95(diffs, iters: int = 10000, seed: int = 0):
    """IC bootstrap apparié 95 % de la moyenne de `diffs` (percentiles 2,5 / 97,5).

    Rééchantillonnage **apparié** : les écarts par item sont tirés avec remise
    (le même item est tiré pour les deux modèles, l'écart est déjà apparié).
    """
    n = len(diffs)
    if n == 0:
        return (None, None)
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        total = 0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    return (means[int(0.025 * (iters - 1))], means[int(0.975 * (iters - 1))])


def math_sqrt(x: float) -> float:
    import math
    return math.sqrt(x)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", required=True, help="run de reference (A)")
    parser.add_argument("--run-b", required=True, help="run compare (B)")
    parser.add_argument("--metric", default="exact_match",
                        help="exact_match | math_verify")
    parser.add_argument("--filter", dest="filter_name", default=None,
                        help="nom de filtre (strict-match, flexible-extract, none)")
    parser.add_argument("--iters", type=int, default=10000,
                        help="nombre de re-echantillonnages bootstrap")
    parser.add_argument("--seed", type=int, default=0, help="graine du bootstrap")
    parser.add_argument("--json", action="store_true", help="sortie JSON")
    args = parser.parse_args()

    a = per_item(Path(args.run_a), args.metric, args.filter_name)
    b = per_item(Path(args.run_b), args.metric, args.filter_name)
    keys_a, keys_b = set(a), set(b)
    if keys_a != keys_b:
        only_a = sorted(keys_a - keys_b)[:5]
        only_b = sorted(keys_b - keys_a)[:5]
        print(f"ERREUR: jeux d'items differents (A: {len(keys_a)}, B: {len(keys_b)}).")
        print(f"       absents de B : {only_a}")
        print(f"       absents de A : {only_b}")
        return 2

    keys = sorted(keys_a)
    n = len(keys)
    hits_a = sum(a[k] for k in keys)
    hits_b = sum(b[k] for k in keys)
    diffs = [b[k] - a[k] for k in keys]
    b_only = sum(1 for k in keys if b[k] == 1 and a[k] == 0)
    a_only = sum(1 for k in keys if a[k] == 1 and b[k] == 0)
    both = sum(1 for k in keys if a[k] == 1 and b[k] == 1)
    neither = sum(1 for k in keys if a[k] == 0 and b[k] == 0)

    lo, hi = bootstrap_ci95(diffs, iters=args.iters, seed=args.seed)
    diff = (hits_b - hits_a) / n

    report = {
        "metric": args.metric,
        "filter": args.filter_name,
        "run_a": args.run_a,
        "run_b": args.run_b,
        "n_items": n,
        "acc_a": hits_a / n,
        "acc_b": hits_b / n,
        "difference": diff,
        "bootstrap_ci95": [lo, hi],
        "bootstrap_iters": args.iters,
        "bootstrap_seed": args.seed,
        "discordant_b_only": b_only,
        "discordant_a_only": a_only,
        "concordant_both": both,
        "concordant_neither": neither,
        "wilson_a": wilson(hits_a, n),
        "wilson_b": wilson(hits_b, n),
    }
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"metrique          : {args.metric}"
              + (f" (filtre {args.filter_name})" if args.filter_name else ""))
        print(f"items apparies   : {n}")
        print(f"precision A      : {100 * report['acc_a']:.2f} %")
        print(f"precision B      : {100 * report['acc_b']:.2f} %")
        print(f"ecart B - A      : {100 * diff:+.2f} points")
        print(f"IC bootstrap 95% : [{100 * lo:+.2f} ; {100 * hi:+.2f}] points "
              f"({args.iters} re-echantillonnages, graine {args.seed})")
        print(f"discordantes     : B seul {b_only}, A seul {a_only}, "
              f"both {both}, neither {neither}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
