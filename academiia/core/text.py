"""Heuristiques de texte génériques (détection de génération dégénérée).

Utilisé par le client d'évaluation A2D et par le smoke test vLLM : la même
définition, une seule source.
"""

from __future__ import annotations


def degenerate(text: str) -> str | None:
    """Retourne une description de répétition dégénérée évidente, sinon None."""
    if not text:
        return "texte vide"
    words = text.split()
    # Un même mot répété quasi partout.
    if len(words) >= 8:
        uniq_ratio = len(set(words)) / len(words)
        if uniq_ratio <= 0.2:
            return f"vocabulaire dégénéré ({len(set(words))} mots uniques / {len(words)})"
    # Un n-gramme (4-16 chars) répété de très nombreuses fois.
    for n in (4, 8, 16):
        if len(text) >= n * 10:
            chunk = text[:n]
            if text.count(chunk) >= 10:
                return f"répétition dégénérée : {chunk!r} × {text.count(chunk)}"
    return None
