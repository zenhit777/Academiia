"""Test torch-isole (T8 du PRE §8) : `build_economical_support` ↔ `_build_restricted_support`.

Le fichier est saute si `torch` est absent (`importorskip` au niveau fichier, J15 P3).
Les tests purs T1–T7 vivent dans `test_dopd_harness_j18r_decision.py`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from academiia.dopd.diagnostic import RestrictedSupport  # noqa: E402

_TESTS = Path(__file__).resolve().parent
_HARNESS_DIR = _TESTS.parent / "harness"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


j18r = _load("dopd_harness_j18r_decision_for_t8", _HARNESS_DIR / "run_j18r_decision.py")
annexe_d = _load("dopd_annexe_d_for_t8", _HARNESS_DIR / "diagnostic_annexe_d.py")

VOCAB = 12
DOMINANCE = 8.0


def logits_for(prefix: tuple[int, ...]) -> np.ndarray:
    """Logits deterministes, avec des ecarts (> 1e-4) assez larges pour un top-k stable."""
    dominant = (7 * sum(prefix) + 3) % VOCAB
    logits = -DOMINANCE + 0.02 * np.arange(VOCAB, dtype=np.float64)
    logits[dominant] = 0.0
    return logits


def _torch_row(prefix: tuple[int, ...]):
    return torch.log_softmax(torch.tensor(logits_for(prefix), dtype=torch.float32), dim=-1).numpy().astype(np.float64)


class FakeModel:
    """Modele duck-typed : `config.vocab_size` et `__call__(input_ids=...)` -> `.logits`."""

    def __init__(self) -> None:
        self.config = type("Config", (), {"vocab_size": VOCAB})()
        self.calls = 0

    def __call__(self, input_ids):
        self.calls += 1
        rows = [torch.tensor(logits_for(tuple(int(token) for token in row)), dtype=torch.float32) for row in input_ids.tolist()]
        return type("Output", (), {"logits": torch.stack([row.unsqueeze(0) for row in rows])})()


def _option_a_forward(prefixes):
    """Option A : la meme operation que le chemin J15 (log_softmax float32 -> float64)."""
    return np.stack([_torch_row(tuple(prefix)) for prefix in prefixes])


def _option_b_forward(prefixes):
    """Option B : voie NumPy pure en float64, sans matrice plein vocabulaire persistee."""
    rows = []
    for prefix in prefixes:
        values = logits_for(tuple(prefix))
        shift = float(np.max(values))
        rows.append((values - shift) - np.log(np.exp(values - shift).sum()))
    return np.stack(rows)


def test_t8_option_a_reproduces_the_j15_support_bit_for_bit():
    model = FakeModel()
    prompt_ids = torch.tensor([], dtype=torch.long)
    reference = annexe_d._build_restricted_support(model, prompt_ids, 4, 6, torch.device("cpu"), batch_size=32)
    economical = j18r.build_economical_support(_option_a_forward, 4, 6, option="A", batch_size=32)
    assert isinstance(economical, RestrictedSupport)
    for position in range(4):
        assert np.array_equal(reference.supports[position], economical.supports[position]), f"support diverge a {position}"
        assert np.max(np.abs(reference.conditional_log_probs[position] - economical.conditional_log_probs[position])) == 0.0


def test_t8_option_b_stays_within_one_e_minus_four_and_keeps_the_same_sets():
    model = FakeModel()
    prompt_ids = torch.tensor([], dtype=torch.long)
    reference = annexe_d._build_restricted_support(model, prompt_ids, 4, 16, torch.device("cpu"), batch_size=32)
    economical = j18r.build_economical_support(_option_b_forward, 4, 16, option="B", arithmetic_dtype="float32", batch_size=32)
    for position in range(4):
        gap = float(np.max(np.abs(reference.conditional_log_probs[position] - economical.conditional_log_probs[position])))
        assert gap <= 1e-4
        assert sorted(reference.supports[position].tolist()) == sorted(economical.supports[position].tolist())


def test_t8_peak_estimate_respects_the_24_gib_budget_at_n4_v16():
    economical = j18r.support_build_peak_bytes(4, 16, 151936, economical=True)
    assert economical["peak_bytes"] <= j18r.MAX_RSS_GIB * 1024**3
    j18r.assert_support_build_fits(4, 16, 151936, economical=True, max_rss_gib=j18r.MAX_RSS_GIB)
    # le chemin J15 (matrice plein vocabulaire, double copie) depasse le plafond a V=24
    j15 = j18r.support_build_peak_bytes(4, 24, 151936, economical=False)
    assert j15["peak_bytes"] > j18r.MAX_RSS_GIB * 1024**3
    with pytest.raises(MemoryError):
        j18r.assert_support_build_fits(4, 24, 151936, economical=False, max_rss_gib=j18r.MAX_RSS_GIB)
