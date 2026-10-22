#!/usr/bin/env python3
"""Vérificateur de liens relatifs des fichiers Markdown (J9, phase 2.4).

Contrôle que chaque lien **relatif** d'un `.md` du dépôt pointe vers un
fichier ou un dossier **existant**. Les liens externes (`http(s)`, `mailto`),
les ancres (`#section`) et les liens vers des chemins hors dépôt
(`checkpoints/`, `_artefact/`, dossiers de travail ignorés par git) sont
ignorés.

Les **rapports historiques** (`modules/a2d/docs/reports/**`) et les **archives
legacy** (`modules/a2d/legacy/docs/**`) ne sont **pas** contrôlés : leurs
chemins sont ceux de l'époque (bandeau de migration).

Usage :
    python tools/check_links.py            # exit 1 si un lien est cassé
    python tools/check_links.py --quiet    # n'affiche que les erreurs
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Répertoires non contrôlés (historique, artefacts, poids).
SKIP_DIRS = {
    ".git", "_artefact", "checkpoints", ".venv", ".fork", ".cache",
    "__pycache__", "node_modules", ".pytest_cache",
}
#: Arborescences explicitement historiques (bandeau de migration).
SKIP_TREES = (
    "modules/a2d/docs/reports",
    "modules/a2d/legacy/docs",
)
#: Préfixes de liens ignorés (hors dépôt, non versionnés).
IGNORED_PREFIXES = ("checkpoints/", "_artefact/", "eval/", "http", "ftp")

LINK_RE = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)]+)\)")


def markdown_files() -> list:
    out = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        rel_dir = os.path.relpath(dirpath, REPO_ROOT)
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        if rel_dir == ".":
            rel_dir = ""
        for name in filenames:
            if name.endswith(".md"):
                out.append(Path(dirpath) / name)
    return sorted(out)


def check_file(path: Path) -> list:
    """Retourne la liste des liens cassés [(cible, raison)]."""
    broken = []
    text = path.read_text(encoding="utf-8", errors="replace")
    rel = str(path.relative_to(REPO_ROOT))
    historical = any(rel == tree or rel.startswith(tree + os.sep)
                     for tree in SKIP_TREES)
    for raw in LINK_RE.findall(text):
        target = raw.strip()
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        if not target or target.startswith("#"):
            continue
        if target.startswith(("http://", "https://", "mailto:", "ftp://")):
            continue
        # lien vers une ancre dans un autre fichier : on ne garde que le chemin
        target = target.split("#", 1)[0].split("?", 1)[0]
        if not target:
            continue
        if historical:
            # Document historique : seul le bandeau de migration est contrôlé
            # (les autres liens pointent vers les chemins de l'époque).
            if "path-migration.md" not in target:
                continue
        if target.startswith("/"):
            resolved = REPO_ROOT / target.lstrip("/")
        else:
            resolved = (path.parent / unquote(target)).resolve()
        if any(target.startswith(prefix) for prefix in IGNORED_PREFIXES):
            continue
        if resolved.exists():
            continue
        broken.append((raw, str(resolved.relative_to(REPO_ROOT))
                       if resolved.is_relative_to(REPO_ROOT) else str(resolved)))
    return broken


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true",
                        help="n'afficher que les erreurs")
    args = parser.parse_args(argv)

    files = markdown_files()
    failures = []
    for path in files:
        for target, resolved in check_file(path):
            failures.append((path.relative_to(REPO_ROOT), target, resolved))
    if not args.quiet:
        print(f"{len(files)} fichiers Markdown contrôlés")
    for rel, target, resolved in failures:
        print(f"LIEN CASSE {rel}: {target} -> {resolved}", file=sys.stderr)
    if failures:
        print(f"{len(failures)} lien(s) cassé(s)", file=sys.stderr)
        return 1
    print("OK : tous les liens relatifs existent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
