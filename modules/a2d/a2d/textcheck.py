"""Contrôles de forme des réponses générées (parties pures, sans réseau).

Extraits de l'ancien `scripts/eval_compare.py` pour que les tests importent le
paquet installé plutôt que d'ajouter un chemin à `sys.path`.
"""

from __future__ import annotations

import json
import re
import unicodedata


def normalize(text: str) -> str:
    """Minuscules, sans accents, espaces normalises (comparaison de contenu)."""
    text = text.replace("œ", "oe").replace("Œ", "OE")
    text = text.replace("æ", "ae").replace("Æ", "AE")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


def strip_reasoning(text: str) -> str:
    """Retire le prefixe 'thought' des sorties du modele de diffusion."""
    out = text
    for _ in range(4):
        s = out.lstrip()
        if s.lower().startswith("thought"):
            out = s[len("thought"):].lstrip()
            continue
        out = s
        break
    return out


def check_format(text: str, fmt: str) -> bool:
    """Verifie le format demande (json / list / one_line)."""
    body = strip_reasoning(text)
    if fmt == "json":
        for candidate in (body, *re.findall(r"[\{\[].*[\}\]]", body, flags=re.S)):
            try:
                json.loads(candidate)
                return True
            except (json.JSONDecodeError, TypeError):
                continue
        return False
    if fmt == "list":
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        bullets = [ln for ln in lines
                   if re.match(r"^([-*•]|\d+[.)])\s+", ln)]
        return len(bullets) >= 2 or len(lines) >= 3
    if fmt == "one_line":
        return len([ln for ln in body.splitlines() if ln.strip()]) == 1
    return True
