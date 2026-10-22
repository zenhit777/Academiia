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

Provenance (audit N1, F8) : le champ `git_hash` des `results_*.json` est le
commit du **fork Dream** (`DreamLM/Dream`), pas un commit d'Academiia — il est
donc publié sous le nom `fork_git_hash`, et le commit d'Academiia au moment du
run sous `academiia_head_at_run`.

Usage :
    python scripts/eval_item_fingerprint.py --root eval/dream_a2d
    python scripts/eval_item_fingerprint.py --run eval/dream_a2d/dream_7b_base/gsm8k_cot_a2d
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Horodatage encode dans les noms de fichiers du fork lm-eval
#: (`results_2026-10-20T00-53-03.608851.json`) : instant de debut du run.
_RUN_STAMP = re.compile(r"_(\d{4})-(\d{2})-(\d{2})T(\d{2})-(\d{2})-(\d{2})(?:\.(\d+))?")

_COMMITS_CACHE: list = []


def _sha256_list(values) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(str(value).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _commits_by_time() -> list:
    """Historique du depot : [(epoch_committer, hash)] du plus ancien au plus recent.

    Retourne [] si git est indisponible ou si le depot n'est pas un depot git.
    """
    if _COMMITS_CACHE:
        return _COMMITS_CACHE
    try:
        out = subprocess.run(["git", "log", "--pretty=%H %ct"], cwd=str(REPO_ROOT),
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    commits = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2:
            commits.append((int(parts[1]), parts[0]))
    commits.reverse()
    _COMMITS_CACHE.extend(commits)
    return _COMMITS_CACHE


def _run_start_epoch(paths) -> float:
    """Instant de debut du run, lu dans le nom des fichiers du fork.

    Repli : date de modification du fichier le plus recent.
    """
    stamps = []
    for path in paths:
        match = _RUN_STAMP.search(Path(path).name)
        if not match:
            continue
        year, month, day, hour, minute, second, micro = match.groups()
        try:
            moment = dt.datetime(int(year), int(month), int(day), int(hour),
                                 int(minute), int(second), int(micro or 0))
        except ValueError:
            continue
        stamps.append(moment.timestamp())
    if stamps:
        return min(stamps)
    mtimes = [Path(path).stat().st_mtime for path in paths]
    return max(mtimes) if mtimes else 0.0


def _head_at(epoch: float):
    """Commit du depot qui etait HEAD a l'instant `epoch` (None si aucun)."""
    head = None
    for timestamp, commit in _commits_by_time():
        if timestamp <= epoch:
            head = commit
        else:
            break
    return head


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
    fork_git_hash = None
    if results:
        with open(results[0]) as f:
            data = json.load(f)
        config = data.get("config", {})
        n_samples = data.get("n-samples", {})
        # `git_hash` des results_*.json = commit du FORK Dream, pas d'Academiia.
        fork_git_hash = data.get("git_hash")
    run_start = _run_start_epoch(samples + results)
    return {
        "run": str(run_dir),
        "n_items": len(items),
        "items_sha256": _sha256_list(sorted(f"{a}:{b}" for a, b in items)),
        # Ordre d'evaluation = ordre de premiere apparition dans les
        # samples_*.jsonl (le fork ecrit les echantillons dans l'ordre ou il
        # les a evaluates). Sans le `sorted`, ce hash mesurerait l'ordre ;
        # avec, il dupliquerait `items_sha256` (cf. valeurs de reference
        # versionnees dans eval/dream_a2d/ITEM_FINGERPRINTS.json).
        "items_order_sha256": _sha256_list(f"{a}:{b}" for a, b in items),
        "prompts_sha256": _sha256_list(sorted(prompts)),
        "targets_sha256": _sha256_list(sorted(targets)),
        "n_samples": n_samples,
        "fork_git_hash": fork_git_hash,
        "academiia_head_at_run": _head_at(run_start),
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
        print(f"    fork_git_hash    = {report['fork_git_hash']}  (fork Dream)")
        print(f"    academiia_head  = {report['academiia_head_at_run']}")
        print(f"    config           = {json.dumps(report['config'], sort_keys=True)}")

    payload = {
        "note": (
            "Provenance : `fork_git_hash` = commit du fork Dream (DreamLM/Dream) recopie "
            "depuis les results_*.json ; ce n'est PAS un commit d'Academiia. "
            "`academiia_head_at_run` = commit du depot Academiia qui etait HEAD au debut "
            "du run (horodatage du nom de fichier du fork, compare a l'historique git) ; "
            "l'arbre de travail pouvait contenir des modifications non commitees (le "
            "harnais Dream n'a ete versionne dans le depot qu'apres les runs, cf. "
            "9ae02a1)."),
        "runs": reports,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nJSON ecrit : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
