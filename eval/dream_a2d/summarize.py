#!/usr/bin/env python3
"""Resume des evaluations Dream (J6, porte G1) : scores + intervalle de confiance.

Lit les `results_*.json` produits par le fork lm-eval de Dream dans
`eval/dream_a2d/<model>_<tache>/`, agrege les metriques, calcule l'intervalle de
confiance de Wilson a 95 % pour les taux d'accuracy, et ecrit un resume JSON +
markdown.

    python eval/dream_a2d/summarize.py --root eval/dream_a2d \
        --out eval/dream_a2d/SUMMARY.json --markdown eval/dream_a2d/SUMMARY.md
"""

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

# Cibles du papier (A2D Table 16, PDF p. 25)
TARGETS = {
    "dream_7b_base": {"IFEval": 49.7, "GSM8K": 66.5, "MATH-500": 20.8},
    "dream_7b_instruct": {"IFEval": 63.0, "GSM8K": 83.0, "MATH-500": 38.4},
}
TASK_TO_BENCH = {
    "gsm8k_cot_a2d": "GSM8K",
    "minerva_math": "MATH-500",
    "ifeval_a2d": "IFEval",
}


def wilson(k: int, n: int, z: float = 1.96):
    """Intervalle de confiance de Wilson a 95 % (z = 1.96)."""
    if n == 0:
        return (None, None)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_results(root):
    runs = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "**", "results_*.json"), recursive=True)):
        with open(path) as f:
            data = json.load(f)
        model_dir = os.path.basename(os.path.dirname(path))
        # le repertoire de sortie est <model>_<tache> ; le modele est le 1er segment
        outer = os.path.basename(os.path.dirname(os.path.dirname(path)))
        run_key = outer
        runs.setdefault(run_key, {"results": data, "path": path, "model_dir": model_dir})
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="eval/dream_a2d")
    parser.add_argument("--out", default=None)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()

    runs = load_results(args.root)
    summary = {}
    for run_key, info in runs.items():
        data = info["results"]
        n_samples = {t: r.get("samples", []) for t, r in data.get("results", {}).items()}
        entry = {"n_prompts": {t: len(v) for t, v in n_samples.items()},
                 "metrics": {}}
        for task, res in data.get("results", {}).items():
            for metric, value in res.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                stderr = res.get(metric + "_stderr")
                n = entry["n_prompts"].get(task, 0)
                lo, hi = wilson(round(value * n), n) if n else (None, None)
                entry["metrics"][f"{task}/{metric}"] = {
                    "value": value, "stderr": stderr, "n": n,
                    "wilson95": [lo, hi],
                }
        summary[run_key] = entry

    # table de synthese : une ligne par (modele, tache)
    rows = []
    for run_key, entry in sorted(summary.items()):
        model = run_key.rsplit("_", 1)[0] if run_key.endswith("_a2d") else run_key
        for key, m in entry["metrics"].items():
            task, metric = key.split("/")
            bench = TASK_TO_BENCH.get(task, task)
            rows.append({
                "run": run_key, "model": model, "task": task, "benchmark": bench,
                "metric": metric, "value": m["value"], "n": m["n"],
                "wilson95": m["wilson95"],
                "target": TARGETS.get(model, {}).get(bench),
            })

    out = {"runs": summary, "rows": rows,
           "targets_a2d_table16": TARGETS,
           "note": ("Effectifs reduits : le budget GPU local ne permet pas les jeux complets "
                    "(mesure : 162 s/prompt pour IFEval a 1280 pas de debruitage). "
                    "L'intervalle de Wilson est donne pour chaque score.")}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"JSON ecrit : {args.out}")
    if args.markdown:
        lines = ["# Resume des evaluations Dream-7B (J6)", "",
                 "| modele | tache | metrique | score | n | IC 95 % (Wilson) | cible Table 16 |",
                 "|---|---|---|---|---|---|---|"]
        for r in rows:
            lo, hi = r["wilson95"] if r["wilson95"] else (None, None)
            ci = f"[{100*lo:.1f}, {100*hi:.1f}]" if lo is not None else "—"
            tgt = f"{r['target']}" if r["target"] is not None else "—"
            lines.append(f"| {r['model']} | {r['benchmark']} | {r['metric']} | "
                         f"{100*r['value']:.1f} % | {r['n']} | {ci} | {tgt} |")
        lines += ["", out["note"], ""]
        with open(args.markdown, "w") as f:
            f.write("\n".join(lines))
        print(f"Markdown ecrit : {args.markdown}")
    print(json.dumps([{k: r[k] for k in ("model", "benchmark", "metric", "value", "n", "target")}
                      for r in rows], indent=2))


if __name__ == "__main__":
    main()
