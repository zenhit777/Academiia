"""Tests de `modules/a2d/eval/eval_compare.py` (parties pures, sans réseau ni GPU).

Cibles : les cas limites du contrôle de format et de la normalisation des
réponses (préfixe « thought » du modèle de diffusion, JSON dans un bloc de
code, listes à puces). Lancer :
    pytest modules/a2d/tests -q
"""

from academiia.a2d.eval import eval_compare  # noqa: E402


def test_normalize_strip_accents_case_and_spaces():
    assert eval_compare.normalize("  Lisbonne  ") == "lisbonne"
    assert eval_compare.normalize("Égypte") == "egypte"
    assert eval_compare.normalize("Nouvelle\nDelhi") == "nouvelle delhi"
    assert eval_compare.normalize("Cœur") == "coeur"


def test_strip_reasoning_removes_diffusion_prefix():
    assert eval_compare.strip_reasoning("thought\n7 × 8 font 56.") == "7 × 8 font 56."
    assert eval_compare.strip_reasoning("thoughtthoughtLa photosynthèse") == "La photosynthèse"
    assert eval_compare.strip_reasoning("Réponse normale") == "Réponse normale"


def test_check_format_json_plain_and_fenced():
    assert eval_compare.check_format('{"city": "Rome"}', "json")
    assert eval_compare.check_format('thought\n{"city": "Rome"}', "json")
    assert eval_compare.check_format('thought\n```\n{"city": "Madrid"}\n```', "json")
    assert eval_compare.check_format('thought\n```\n{\n  "city": "Madrid",\n  """: "Spain"\n}\n```',
                                     "json") is False


def test_check_format_json_rejects_prose():
    assert eval_compare.check_format("La capitale de l'Italie est Rome.", "json") is False


def test_check_format_list():
    assert eval_compare.check_format("- pommes\n- poires\n- prunes", "list")
    assert eval_compare.check_format("1. pommes\n2. poires\n3. prunes", "list")
    assert eval_compare.check_format("pommes", "list") is False


def test_check_format_one_line():
    assert eval_compare.check_format("thought\nLe train glisse silencieusement.", "one_line")
    assert eval_compare.check_format("thought\nPremière phrase.\nDeuxième phrase.", "one_line") is False
