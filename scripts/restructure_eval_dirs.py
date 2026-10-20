#!/usr/bin/env python3
"""Restructure les dossiers de résultats d'évaluation (mission J6bis, C6).

Problème : le fork lm-eval crée un sous-dossier dont le NOM encode le chemin
absolu du checkpoint (`__mnt__d__Intelligence_Artificielle__Projects__…`).
Ces noms ne sont pas portables et exposent l'arborescence de la machine.

Ce script réorganise

    eval/dream_a2d/<model>_<tache>/__mnt__d__…__checkpoints__<model>/<fichiers>

en

    eval/dream_a2d/<model>/<tache>/<fichiers>

**Le contenu des `.json` / `.jsonl` n'est pas modifié** : les chemins absolus
qui y figurent (metadata `pretrained`, `config.model_args`, `model_name`) sont
écrits par le harnais et constituent la provenance du run ; c'est leur présence
qui est consignée, pas leur suppression.

Un fichier de correspondance (`RESTRUCTURE_MAP.json`) est écrit à la racine du
dossier de résultats : ancien chemin → nouveau chemin, fichiers déplacés,
taille et empreinte de chaque fichier (avant/après, pour prouver que le
contenu n'a pas changé).

Usage :
    python scripts/restructure_eval_dirs.py --root eval/dream_a2d [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

ENCODED_PREFIX = "__"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_encoded_dirs(directory: Path):
    """Sous-dossiers dont le nom encode un chemin absolu.

    Le nom encode est forme de segments separes par `__` (`__mnt__d__…`) ;
    `__pycache__` n'en fait pas partie et est exclu.
    """
    for entry in sorted(directory.iterdir()):
        if not entry.is_dir() or not entry.name.startswith(ENCODED_PREFIX):
            continue
        if entry.name == "__pycache__":
            continue
        if entry.name.count("__") < 2:
            continue
        yield entry


def model_alias(encoded_name: str) -> str:
    """Alias du checkpoint : dernier segment du nom encode.

    `__mnt__d__...__checkpoints__dream_7b_base` -> `dream_7b_base`.
    """
    return encoded_name.split("__")[-1]


def task_name(run_dir_name: str, model: str) -> str:
    """Nom de tache : nom du dossier de run minus le préfixe modele."""
    prefix = model + "_"
    if run_dir_name.startswith(prefix):
        return run_dir_name[len(prefix):]
    return run_dir_name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="eval/dream_a2d",
                        help="racine des resultats a restructurer")
    parser.add_argument("--dry-run", action="store_true",
                        help="ecrit le plan sans deplacer les fichiers")
    parser.add_argument("--map", default=None,
                        help="fichier de correspondance (defaut : <root>/RESTRUCTURE_MAP.json)")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERREUR: {root} n'est pas un dossier")
        return 2
    map_path = Path(args.map) if args.map else root / "RESTRUCTURE_MAP.json"

    entries = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        encoded = list(find_encoded_dirs(run_dir))
        if not encoded:
            continue
        if len(encoded) > 1:
            print(f"ERREUR: {run_dir} contient {len(encoded)} dossiers encodes : "
                  "ambiguite, traitement manuel requis")
            return 1
        encoded_dir = encoded[0]
        model = model_alias(encoded_dir.name)
        task = task_name(run_dir.name, model)
        target = root / model / task
        files = sorted(p for p in encoded_dir.iterdir() if p.is_file())
        entries.append({
            "source_run_dir": str(run_dir),
            "source_encoded_dir": str(encoded_dir),
            "target_dir": str(target),
            "model": model,
            "task": task,
            "files": [{"name": p.name, "size": p.stat().st_size,
                       "sha256": sha256(p)} for p in files],
        })

    if not entries:
        print(f"Aucun dossier encode trouve dans {root} : rien a restructurer.")
        return 0

    payload = {
        "tool": "scripts/restructure_eval_dirs.py",
        "root": str(root),
        "dry_run": args.dry_run,
        "note": ("Le CONTENU des .json / .jsonl n'est pas modifie : les chemins "
                 "absolus qu'ils contiennent sont la provenance du run. Seuls les "
                 "NOMS de dossiers sont rendu portables."),
        "mappings": entries,
    }

    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    for entry in entries:
        source = Path(entry["source_encoded_dir"])
        target = Path(entry["target_dir"])
        target.mkdir(parents=True, exist_ok=True)
        for meta in entry["files"]:
            src = source / meta["name"]
            dst = target / meta["name"]
            if dst.exists():
                if sha256(dst) == meta["sha256"]:
                    src.unlink()
                    continue
                raise SystemExit(f"ERREUR: {dst} existe avec un contenu different")
            shutil.move(str(src), str(dst))
            digest = sha256(dst)
            if digest != meta["sha256"]:
                raise SystemExit(f"ERREUR: empreinte non preservee pour {dst}")
        # dossiers intermediaires vides
        for directory in (source, Path(entry["source_run_dir"])):
            try:
                directory.rmdir()
            except OSError:
                pass
        print(f"{entry['source_run_dir']} -> {entry['target_dir']} "
              f"({len(entry['files'])} fichier(s))")

    with open(map_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Correspondance ecrite : {map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
