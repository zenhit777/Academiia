"""Tests de `scripts/g1_verdict.py` (logique pure des verdicts, sans GPU).

La faute la plus grave possible ici est un **mélange d'unités** : les écarts
et les bornes sont des **proportions** (0,1433 = +14,33 points) alors que les
seuils des critères sont en **points** (+10, +8, −2). Un test le verrouille.

Lancer :
    pytest scripts/tests -q
"""

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS / "eval"))

from g1_verdict import verdict_gain, verdict_non_regression  # noqa: E402


def test_gain_met_avec_proportion_et_seuil_en_points():
    """+14,33 points >= +8 et borne basse > 0 : ATTEINT (faut-il 0,1433, pas 14,33)."""
    v = verdict_gain("C2", 8.0, 0.1433333333, 0.0933333333, 0.1966666667)
    assert v["atteint"] is True
    assert v["etat"] == "ATTEINT"
    assert abs(v["point_estimate"] - 14.33) < 0.01
    assert abs(v["borne_basse"] - 9.33) < 0.01


def test_gain_point_estimate_sous_le_seuil_but_ic_exclut_zero():
    """+9,71 points < +10 : le critère n'est PAS atteint, même si l'IC exclut 0."""
    v = verdict_gain("C1", 10.0, 0.0971428571, 0.0285714286, 0.1657142857)
    assert v["atteint"] is False
    assert v["condition_point_estimate"] is False
    assert v["condition_borne_basse"] is True
    assert "sous le seuil" in v["etat"]


def test_gain_borne_basse_nulle_ou_negative():
    """Point estimate au seuil mais IC contenant 0 : NON ATTEINT."""
    v = verdict_gain("C1", 10.0, 0.12, -0.01, 0.25)
    assert v["atteint"] is False
    assert v["condition_point_estimate"] is True
    assert v["condition_borne_basse"] is False
    assert "borne basse" in v["etat"]


def test_non_regression_non_refutee_au_dessus_de_moins_deux():
    """Borne basse à -1,9 point : non-régression NON RÉFUTÉE (critère atteint)."""
    v = verdict_non_regression("C4", -2.0, 0.0166666667, -0.019, 0.0566666667)
    assert v["atteint"] is True
    assert "NON REFUTEE" in v["etat"]


def test_non_regression_non_etablie_sous_moins_deux():
    """Borne basse à -2,33 points : le critère n'est PAS atteint.

    L'estimation ponctuelle reste positive (+1,67) : le libellé doit dire
    « non établie » (la dégradation ne peut pas être exclue), pas « régression ».
    """
    v = verdict_non_regression("C4", -2.0, 0.0166666667, -0.0233333333, 0.0566666667)
    assert v["atteint"] is False
    assert "NON ETABLIE" in v["etat"]
    assert v["point_estimate"] > 0


def test_non_regression_etablie_si_borne_basse_positive():
    v = verdict_non_regression("C3", -2.0, 0.01, 0.005, 0.05)
    assert v["atteint"] is True
    assert "ETABLIE" in v["etat"]


def test_bornes_limites_exactes():
    """Borne basse exactement à -2 : non réfutée ; exactement à 0 : établie."""
    v1 = verdict_non_regression("C3", -2.0, 0.0, -0.02, 0.02)
    assert v1["atteint"] is True and "NON REFUTEE" in v1["etat"]
    v2 = verdict_non_regression("C3", -2.0, 0.0, 0.0, 0.02)
    assert v2["atteint"] is True and "ETABLIE" in v2["etat"]
    v3 = verdict_gain("C2", 8.0, 0.08, 0.01, 0.15)
    assert v3["atteint"] is True
