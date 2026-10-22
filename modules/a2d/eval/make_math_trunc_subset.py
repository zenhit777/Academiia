#!/usr/bin/env python3
"""Construit le sous-ensemble des items MATH tronques au budget de decodage.

Objectif (mission J8, Phase A) : isoler les problemes MATH-175 dont la
generation a **epuise le budget** de tokens sans produire de reponse finale
(`\\boxed{...}` ou « answer is »). Ces items sont ceux pour lesquels le budget
de 512 tokens peut masquer un gain du transfert A2D.

Le sous-ensemble est ecrit au format JSONL consomme par la tache lm-eval
`modules/a2d/harness/dream/tasks/math_trunc_a2d.yaml` ; un manifeste porte la provenance
(modele, budget, critere, empreintes SHA-256 des fichiers d'echantillons lus).

Usage :

    python modules/a2d/eval/make_math_trunc_subset.py \
        --samples-root modules/a2d/eval/dream_a2d \
        --model dream_transfer_a09 \
        --tokenizer checkpoints/dream_7b_instruct \
        --budget 512 \
        --output modules/a2d/harness/dream/tasks/math_trunc_a2d.jsonl

Le script ne lit que des fichiers de resultats deja versionnes : aucun GPU,
aucune ecriture hors du depot.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from pathlib import Path

SUBJECTS = (
    "algebra",
    "counting_and_prob",
    "geometry",
    "intermediate_algebra",
    "num_theory",
    "prealgebra",
    "precalc",
)

#: Marqueurs d'une reponse finale (format libre de MATH cote Dream).
FINAL_ANSWER_RE = re.compile(r"\\boxed|answer is", re.IGNORECASE)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text).input_ids)


def collect(args, tokenizer) -> tuple[list[dict], dict]:
    root = Path(args.samples_root) / args.model
    files = sorted(glob.glob(str(root / "minerva_math_*" / "samples_*.jsonl")))
    if not files:
        raise SystemExit(f"aucun echantillon trouve sous {root}")

    selected: list[dict] = []
    seen: set[tuple[str, int]] = set()
    total = 0
    exhausted = 0
    exhausted_with_answer = 0
    per_subject: dict[str, dict] = {}

    for path in files:
        match = re.search(r"samples_minerva_math_([a-z_]+)_\d{4}-", os.path.basename(path))
        if not match:
            raise SystemExit(f"nom d'echantillon non reconnu : {path}")
        subject = match.group(1)
        counts = per_subject.setdefault(subject, {"n": 0, "exhausted": 0,
                                                  "exhausted_with_answer": 0,
                                                  "selected": 0})
        for line in open(path, encoding="utf-8"):
            sample = json.loads(line)
            total += 1
            counts["n"] += 1
            response = sample["resps"][0][0]
            length = token_count(tokenizer, response)
            if length < args.budget:
                continue
            exhausted += 1
            counts["exhausted"] += 1
            has_answer = bool(FINAL_ANSWER_RE.search(response))
            if has_answer:
                exhausted_with_answer += 1
                counts["exhausted_with_answer"] += 1
                if args.keep_complete:
                    pass
                else:
                    continue
            doc = sample["doc"]
            key = (subject, int(sample["doc_id"]))
            if key in seen:
                raise SystemExit(f"item duplique : {key}")
            seen.add(key)
            selected.append({
                "subject": subject,
                "doc_id": int(sample["doc_id"]),
                "problem": doc["problem"],
                "solution": doc["solution"],
                "answer": doc["answer"],
                "source_sample_sha256": sha256_file(path),
            })
            counts["selected"] += 1

    selected.sort(key=lambda d: (d["subject"], d["doc_id"]))
    manifest = {
        "model": args.model,
        "samples_root": str(Path(args.samples_root)),
        "tokenizer": str(Path(args.tokenizer)),
        "budget_tokens": args.budget,
        "criterion": ("tokens(response) >= budget AND no final answer marker "
                      "(\\boxed or 'answer is')" if not args.keep_complete
                      else "tokens(response) >= budget"),
        "n_items_total": total,
        "n_exhausted": exhausted,
        "n_exhausted_with_final_answer": exhausted_with_answer,
        "n_exhausted_without_final_answer": exhausted - exhausted_with_answer,
        "n_selected": len(selected),
        "per_subject": dict(sorted(per_subject.items())),
        "source_files": {Path(p).name: sha256_file(p) for p in files},
    }
    return selected, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", default="modules/a2d/eval/dream_a2d")
    parser.add_argument("--model", default="dream_transfer_a09",
                        help="modele dont les echantillons definissent les items tronques")
    parser.add_argument("--tokenizer", default="checkpoints/dream_7b_instruct")
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--output", default="modules/a2d/harness/dream/tasks/math_trunc_a2d.jsonl")
    parser.add_argument("--manifest", default=None,
                        help="manifeste JSON (defaut : <output>.manifest.json)")
    parser.add_argument("--keep-complete", action="store_true",
                        help="garder aussi les generations completes qui ont "
                             "consomme tout le budget")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rows, manifest = collect(args, tokenizer)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest["output"] = str(output)
    manifest["output_sha256"] = sha256_file(str(output))
    manifest_path = Path(args.manifest) if args.manifest else Path(str(output) + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")

    print(f"Items tronques : {manifest['n_selected']} / {manifest['n_items_total']} "
          f"(budget {args.budget} tokens, modele {args.model})")
    print(f"  generations ayant epuise le budget : {manifest['n_exhausted']}")
    print(f"  dont sans reponse finale          : {manifest['n_exhausted_without_final_answer']}")
    print(f"  sortie : {output}")
    print(f"  manifeste : {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
