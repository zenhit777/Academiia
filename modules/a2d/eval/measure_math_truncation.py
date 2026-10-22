#!/usr/bin/env python3
"""Mesure la troncature des generations MATH au budget de decodage (CPU).

Critere (identique a `modules/a2d/eval/make_math_trunc_subset.py`) :

    tokens(reponse) >= budget  ET  aucun marqueur de reponse finale
    (`\\boxed{...}` ou « answer is »)

Un item « tronque » est donc un item dont la generation a **epuise le budget
sans conclure** : c'est le seul cas ou un budget plus grand peut changer le
verdict. Les generations qui atteignent le budget avec une reponse finale
(`\\boxed`) ne sont pas comptees comme tronquees.

Usage :
    python modules/a2d/eval/measure_math_truncation.py \
        --samples-root modules/a2d/eval/dream_a2d \
        --tokenizer checkpoints/dream_7b_instruct \
        --output modules/a2d/eval/dream_a2d/TRUNCATION_512.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

MODELS = ("dream_7b_base", "dream_7b_instruct",
          "dream_transfer_a09", "dream_merge_l06")
FINAL_ANSWER_RE = re.compile(r"\\boxed|answer is", re.IGNORECASE)


def count_model(root: Path, model: str, tokenizer, budget: int) -> dict:
    files = sorted(glob.glob(str(root / model / "minerva_math_*" / "samples_*.jsonl")))
    if not files:
        raise FileNotFoundError(f"aucun echantillon MATH pour {model} sous {root}")
    total = exhausted = exhausted_with_answer = 0
    per_subject: dict[str, dict] = {}
    for path in files:
        match = re.search(r"samples_minerva_math_([a-z_]+)_\d{4}-", os.path.basename(path))
        if not match:
            raise ValueError(f"nom d'echantillon non reconnu : {path}")
        subject = match.group(1)
        counts = per_subject.setdefault(subject, {
            "n": 0, "exhausted": 0, "exhausted_with_answer": 0,
            "truncated_without_answer": 0})
        for line in open(path, encoding="utf-8"):
            sample = json.loads(line)
            response = sample["resps"][0][0]
            length = len(tokenizer(response).input_ids)
            total += 1
            counts["n"] += 1
            if length < budget:
                continue
            exhausted += 1
            counts["exhausted"] += 1
            if FINAL_ANSWER_RE.search(response):
                exhausted_with_answer += 1
                counts["exhausted_with_answer"] += 1
            else:
                counts["truncated_without_answer"] += 1
    return {
        "model": model,
        "n_items": total,
        "budget_tokens": budget,
        "n_exhausted": exhausted,
        "n_exhausted_with_final_answer": exhausted_with_answer,
        "n_truncated_without_final_answer": exhausted - exhausted_with_answer,
        "per_subject": dict(sorted(per_subject.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", default="modules/a2d/eval/dream_a2d")
    parser.add_argument("--tokenizer", default="checkpoints/dream_7b_instruct")
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--output", default="modules/a2d/eval/dream_a2d/TRUNCATION_512.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    report = {
        "criterion": ("tokens(response) >= budget AND no final answer marker "
                      "(\\boxed or 'answer is')"),
        "tokenizer": args.tokenizer,
        "budget_tokens": args.budget,
        "models": [count_model(Path(args.samples_root), model, tokenizer, args.budget)
                   for model in args.models],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    for stats in report["models"]:
        print(f"{stats['model']:22s} n={stats['n_items']:3d} "
              f"budget epuise={stats['n_exhausted']:3d} "
              f"tronques sans reponse={stats['n_truncated_without_final_answer']:3d}")
    print(f"Sortie : {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
