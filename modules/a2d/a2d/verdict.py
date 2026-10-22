"""Verdicts mécaniques de la porte G1 (parties pures, sans I/O).

Extraits de l'ancien `scripts/g1_verdict.py` : le script reste la CLI qui lit
les résultats et calcule les IC ; les règles de verdict vivent ici pour être
testées sur le paquet installé, sans aménagement de `sys.path`.
"""

from __future__ import annotations


def verdict_gain(criterion, threshold: float, diff: float, lo: float, hi: float):
    """C1/C2 : point estimate >= seuil ET borne basse > 0.

    `threshold` est exprime en POINTS ; `diff`, `lo` et `hi` sont des
    proportions (0,1433 = +14,33 points) — la conversion est faite ici.
    """
    diff_pt, lo_pt, hi_pt = diff * 100.0, lo * 100.0, hi * 100.0
    ok_point = diff_pt >= threshold
    ok_ci = lo_pt > 0.0
    if ok_point and ok_ci:
        state = "ATTEINT"
    elif not ok_point and ok_ci:
        state = "NON ATTEINT (point estimate sous le seuil, IC exclut 0)"
    elif ok_point and not ok_ci:
        state = "NON ATTEINT (borne basse <= 0)"
    else:
        state = "NON ATTEINT (point estimate sous le seuil et IC contient 0)"
    return {
        "criterion": criterion,
        "type": "gain",
        "seuil_points": threshold,
        "point_estimate": diff_pt,
        "borne_basse": lo_pt,
        "borne_haute": hi_pt,
        "condition_point_estimate": ok_point,
        "condition_borne_basse": ok_ci,
        "etat": state,
        "atteint": ok_point and ok_ci,
    }


def verdict_non_regression(criterion, threshold: float, diff: float, lo: float, hi: float):
    """C3/C4 : non-regression « non refutee » / « etablie » / « non etablie ».

    `threshold` est en POINTS (-2) ; `diff`, `lo`, `hi` sont des proportions.

    Le troisieme cas (borne basse < -2) signifie que la degradation de 2
    points **ne peut pas etre exclue** : le critere n'est pas atteint, mais
    l'estimation ponctuelle peut rester positive — le libelle le dit sans
    surinterpreter.
    """
    diff_pt, lo_pt, hi_pt = diff * 100.0, lo * 100.0, hi * 100.0
    if lo_pt < threshold:
        state = ("non-regression NON ETABLIE (borne basse < -2 : la degradation "
                 "de 2 points ne peut pas etre exclue)")
        atteint = False
    elif lo_pt >= 0.0:
        state = "non-regression ETABLIE (borne basse >= 0)"
        atteint = True
    else:
        state = "non-regression NON REFUTEE (borne basse >= -2)"
        atteint = True
    return {
        "criterion": criterion,
        "type": "non-regression",
        "seuil_points": threshold,
        "point_estimate": diff_pt,
        "borne_basse": lo_pt,
        "borne_haute": hi_pt,
        "etat": state,
        "atteint": atteint,
    }
