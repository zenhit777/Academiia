#!/usr/bin/env python3
"""Plancher de bruit de décodage — deux runs du MÊME modèle, graines différentes (J7, M2).

Le décodage de Dream-7B est **stochastique en principe** (`alg=entropy`,
`temperature=0,1`) et **dépendant du lot** (§3.3 de
`modules/a2d/docs/design/G1_PROTOCOL.md`). Ce script **mesure** si rejouer le même modèle
avec une autre graine change les réponses — et de combien : c'est le
**plancher de bruit** dû au seul décodage, comparé sur deux runs du même
modèle. La mesure J7 (2026-10-21) a trouvé **0,00 pt** à lot 4 : les 300
réponses brutes étaient byte-identiques (voir `modules/a2d/docs/reports/J7.md` §5) ; le
script reste l'outil de mesure, pas une prévision du résultat.

1. **contrôle bloquant** : les jeux d'items et les prompts doivent être
   **identiques** (`items_sha256`, `prompts_sha256`, calculés comme
   `modules/a2d/eval/eval_item_fingerprint.py`) ;
2. comparaison **item par item** : réponses brutes identiques, réponses filtrées
   identiques, verdicts identiques ;
3. écart de précision apparié + **IC bootstrap apparié 95 %** (même algorithme
   et même graine que `modules/a2d/eval/paired_bootstrap.py`).

Usage :
    python3 modules/a2d/eval/seed_noise_floor.py \
        --run-a modules/a2d/eval/dream_a2d/dream_7b_instruct/gsm8k_cot_a2d \
        --run-b modules/a2d/eval/dream_a2d_seed1/dream_7b_instruct/gsm8k_cot_a2d \
        --filter flexible-extract
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from academiia.a2d.eval.eval_item_fingerprint import fingerprint
from academiia.core.stats import bootstrap_ci95, wilson


def load_items(run_dir: Path, filter_name: str | None):
    """{cle_item: (reponse_brute, reponse_filtree, correct)} pour un run."""
    samples = sorted(glob.glob(str(run_dir / "samples_*.jsonl")))
    if not samples:
        raise SystemExit(f"ERREUR: aucun samples_*.jsonl dans {run_dir}")
    out = {}
    for sample_file in samples:
        with open(sample_file) as f:
            for line in f:
                record = json.loads(line)
                if filter_name is not None and record.get("filter") != filter_name:
                    continue
                key = (record.get("doc_id"), record.get("doc_hash"))
                raw = record.get("resps") or []
                raw_text = raw[0][0] if raw and raw[0] else ""
                filtered = record.get("filtered_resps") or []
                filtered_text = filtered[0] if filtered else ""
                out[key] = (raw_text, filtered_text,
                            1 if float(record.get("exact_match", 0)) == 1.0 else 0)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", required=True, help="run de reference (graine de reference)")
    parser.add_argument("--run-b", required=True, help="run rejoue (graine alternative)")
    parser.add_argument("--filter", dest="filter_name", default="flexible-extract",
                        help="filtre lm_eval (flexible-extract, strict-match, none)")
    parser.add_argument("--iters", type=int, default=10000,
                        help="nombre de re-echantillonnages bootstrap")
    parser.add_argument("--seed", type=int, default=0, help="graine du bootstrap")
    parser.add_argument("--json", action="store_true", help="sortie JSON")
    args = parser.parse_args()

    run_a, run_b = Path(args.run_a), Path(args.run_b)
    fp_a, fp_b = fingerprint(run_a), fingerprint(run_b)
    for fp, run in ((fp_a, run_a), (fp_b, run_b)):
        if "error" in fp:
            print(f"ERREUR: {run} : {fp['error']}")
            return 2

    same_items = (fp_a["items_sha256"] == fp_b["items_sha256"]
                  and fp_a["prompts_sha256"] == fp_b["prompts_sha256"])
    if not same_items:
        print("ERREUR: jeux d'items ou prompts DIFFERENTS entre les deux runs : "
              "le replay n'est pas comparable.")
        print(f"       A items={fp_a['items_sha256'][:16]} prompts={fp_a['prompts_sha256'][:16]}")
        print(f"       B items={fp_b['items_sha256'][:16]} prompts={fp_b['prompts_sha256'][:16]}")
        return 2

    a = load_items(run_a, args.filter_name)
    b = load_items(run_b, args.filter_name)
    if set(a) != set(b):
        print(f"ERREUR: items apparies differents (A: {len(a)}, B: {len(b)}).")
        return 2
    keys = sorted(a)
    n = len(keys)

    same_raw = sum(1 for k in keys if a[k][0] == b[k][0])
    same_filtered = sum(1 for k in keys if a[k][1] == b[k][1])
    same_verdict = sum(1 for k in keys if a[k][2] == b[k][2])
    hits_a = sum(a[k][2] for k in keys)
    hits_b = sum(b[k][2] for k in keys)
    diffs = [b[k][2] - a[k][2] for k in keys]
    abs_diffs = [abs(d) for d in diffs]
    lo, hi = bootstrap_ci95(diffs, iters=args.iters, seed=args.seed)

    report = {
        "run_a": str(run_a),
        "run_b": str(run_b),
        "filter": args.filter_name,
        "n_items": n,
        "items_sha256": fp_a["items_sha256"],
        "prompts_sha256": fp_a["prompts_sha256"],
        "identical_items_and_prompts": True,
        "identical_raw_responses": same_raw,
        "identical_filtered_responses": same_filtered,
        "identical_verdicts": same_verdict,
        "pct_identical_raw_responses": 100.0 * same_raw / n,
        "pct_identical_filtered_responses": 100.0 * same_filtered / n,
        "pct_identical_verdicts": 100.0 * same_verdict / n,
        "acc_a": hits_a / n,
        "acc_b": hits_b / n,
        "difference": (hits_b - hits_a) / n,
        "bootstrap_ci95": [lo, hi],
        "bootstrap_iters": args.iters,
        "bootstrap_seed": args.seed,
        "max_abs_item_difference": max(abs_diffs) if abs_diffs else 0,
        "mean_abs_item_difference": (sum(abs_diffs) / n) if n else 0.0,
        "discordant_b_only": sum(1 for k in keys if b[k][2] == 1 and a[k][2] == 0),
        "discordant_a_only": sum(1 for k in keys if a[k][2] == 1 and b[k][2] == 0),
        "wilson_a": wilson(hits_a, n),
        "wilson_b": wilson(hits_b, n),
    }
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print(f"filtre              : {args.filter_name}")
    print(f"items apparies      : {n}  (items_sha256 / prompts_sha256 identiques : oui)")
    print(f"reponses BRUTES identiques   : {same_raw}/{n} = {100.0 * same_raw / n:.1f} %")
    print(f"reponses FILTREES identiques : {same_filtered}/{n} = {100.0 * same_filtered / n:.1f} %")
    print(f"verdicts identiques         : {same_verdict}/{n} = {100.0 * same_verdict / n:.1f} %")
    print(f"precision A         : {100 * report['acc_a']:.2f} % "
          f"(IC Wilson [{100 * report['wilson_a'][0]:.2f} ; {100 * report['wilson_a'][1]:.2f}])")
    print(f"precision B         : {100 * report['acc_b']:.2f} % "
          f"(IC Wilson [{100 * report['wilson_b'][0]:.2f} ; {100 * report['wilson_b'][1]:.2f}])")
    print(f"ecart B - A         : {100 * report['difference']:+.2f} points")
    print(f"IC bootstrap 95 %   : [{100 * lo:+.2f} ; {100 * hi:+.2f}] points "
          f"({args.iters} re-echantillonnages, graine {args.seed})")
    print(f"ecart absolu max par item : {report['max_abs_item_difference']} "
          f"(moyenne {report['mean_abs_item_difference']:.4f})")
    print(f"discordantes        : B seul {report['discordant_b_only']}, "
          f"A seul {report['discordant_a_only']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
