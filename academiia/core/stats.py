"""Statistiques génériques : IC bootstrap apparié et intervalle de Wilson.

Algorithmes **figés** : le verdict G1 (``modules/a2d/eval/g1_verdict.py``) doit
rester numériquement identique après extraction (même tirage, même graine,
mêmes percentiles). Ne pas « améliorer » sans rejouer la comparaison J9.
"""

from __future__ import annotations

import math
import random


def wilson(k: int, n: int, z: float = 1.96):
    """Intervalle de confiance de Wilson à 95 % (z = 1.96)."""
    if n == 0:
        return (None, None)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci95(diffs, iters: int = 10000, seed: int = 0):
    """IC bootstrap apparié 95 % de la moyenne de `diffs` (percentiles 2,5 / 97,5).

    Rééchantillonnage **apparié** : les écarts par item sont tirés avec remise
    (le même item est tiré pour les deux modèles, l'écart est déjà apparié).
    """
    n = len(diffs)
    if n == 0:
        return (None, None)
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        total = 0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    return (means[int(0.025 * (iters - 1))], means[int(0.975 * (iters - 1))])
