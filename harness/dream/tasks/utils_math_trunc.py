"""Utilitaires de la tache `math_trunc_a2d` (diagnostic J8).

La tache reutilise **exactement** le prompt et le scoring de
`lm_eval/tasks/minerva_math/utils.py` du fork Dream : seul le jeu d'items
change (sous-ensemble local des problemes tronques a 512 tokens).
"""

from __future__ import annotations

from lm_eval.tasks.minerva_math.utils import (  # noqa: F401
    doc_to_text,
    process_results,
)


def process_docs(dataset):
    """Pas de retraitement : le JSONL porte deja `problem`, `solve`, `answer`."""
    return dataset
