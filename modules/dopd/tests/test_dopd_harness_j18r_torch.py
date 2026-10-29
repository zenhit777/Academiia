"""Test torch-isole (T8 du PRE §8) : `build_economical_support` ↔ `_build_restricted_support`.

Le fichier est saute si `torch` est absent (`importorskip` au niveau fichier, J15 P3).
Les tests purs T1–T7 vivent dans `test_dopd_harness_j18r_decision.py`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from academiia.dopd.diagnostic import RestrictedSupport  # noqa: E402
from academiia.dopd.interfaces import ScoringRequest  # noqa: E402

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

VOCAB = 256
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
    """Modele duck-typed de la fixture : `config`, `__call__` -> `.logits` + `.past_key_values`,
    gelable (`eval()`, `parameters()` vides). Les logits sont deterministes (`logits_for`) ;
    le cache est l'objet minimal a `.layers` (K, V) que le prefill du scorer consume."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            vocab_size=VOCAB,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_hidden_layers=2,
            sliding_window=None,
            layer_types=None,
        )
        self.calls = 0
        self.device = "cpu"

    def to(self, device):
        self.device = str(device)
        return self

    def eval(self):
        return self

    def parameters(self):
        return []

    def _cache(self, batch: int, length: int):
        layers = [
            SimpleNamespace(
                keys=torch.zeros((batch, self.config.num_key_value_heads, length, self.config.head_dim), dtype=torch.float32, device=self.device),
                values=torch.zeros((batch, self.config.num_key_value_heads, length, self.config.head_dim), dtype=torch.float32, device=self.device),
            )
            for _ in range(self.config.num_hidden_layers)
        ]
        return SimpleNamespace(layers=layers)

    def __call__(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False, logits_to_keep=None):
        self.calls += 1
        rows = []
        for row in input_ids.tolist():
            tokens = tuple(int(token) for token in row)
            if not tokens:
                # Contexte vide (J15 évalue le support sans prompt) : une seule position.
                rows.append(
                    torch.stack(
                        [torch.tensor(logits_for(()), dtype=torch.float32, device=self.device)]
                    )
                )
                continue
            # Une ligne de logits **par position** : la position `i` ne voit que `tokens[:i+1]`.
            rows.append(
                torch.stack(
                    [
                        torch.tensor(logits_for(tokens[: index + 1]), dtype=torch.float32, device=self.device)
                        for index in range(len(tokens))
                    ]
                )
            )
        logits = torch.stack(rows)
        # `logits_to_keep` : Hugging Face ne rend que les `n` dernières positions.
        if logits_to_keep is not None and 0 < int(logits_to_keep) < logits.shape[1]:
            logits = logits[:, -int(logits_to_keep) :, :]
        cache = self._cache(input_ids.shape[0], input_ids.shape[1]) if use_cache else None
        return SimpleNamespace(logits=logits, past_key_values=cache)


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


# ---------------------------------------------------------------------------
# R-J18R-12 — préfixe conditionné sur le prompt, scoré par le `TeacherScorer` réel
# ---------------------------------------------------------------------------

scorer_module = _load("dopd_teacher_scorer_qwen3_for_torch", _HARNESS_DIR / "teacher_scorer_qwen3.py")
Qwen3TeacherScorer = scorer_module.Qwen3TeacherScorer


def test_r_j18r_12_scorer_refuses_the_empty_relative_prefix() -> None:
    """D4 : à `position = 0` le préfixe **relatif** du bloc est vide ; le scorer de production
    le refuse (`teacher_scorer_qwen3.py:287-289`, `:378-381`). Le provider doit donc porter le
    prompt (eq. 14), comme le support J15 (`prompt_ids + prefix`)."""
    scorer = Qwen3TeacherScorer(FakeModel(), device="cpu")
    request = ScoringRequest(prefix_ids=(), suffix_ids=(1,), candidate_ids=(2,), reference_id=2)
    with pytest.raises(ValueError, match="prefix_ids est vide"):
        scorer.score(request)
    with pytest.raises(ValueError, match="prefix_ids est vide"):
        scorer.replay_job([], [2], [1])


def test_r_j18r_12_provider_conditions_the_prefix_on_prompt_ids() -> None:
    model = FakeModel()
    provider = j18r.ProductionScoreProvider(Qwen3TeacherScorer(model, device="cpu"), prompt_ids=(1, 2, 3))
    prior, scores, reference = provider.job(prefix=(), position=0, tokens=[4, 5], suffix=(6,), reference_token=4)
    assert prior.shape == (VOCAB,)
    assert scores.shape == (2,)
    assert math.isfinite(reference)
    replayed_prior, replayed_scores = provider.replay_job(prefix=(), candidates=[4, 5], suffix=(6,))
    assert replayed_prior.shape == (2,)
    assert replayed_scores.shape == (2,)


# ---------------------------------------------------------------------------
# T5 complété — chemin `execute` entier sur la fixture (GPU si disponible)
# ---------------------------------------------------------------------------


def _scratch_output(name: str) -> Path:
    """Sortie de test dans `_artefact/` (règle 9), en chemin **relatif** au dépôt."""
    root = _TESTS.parents[2]
    out = root / "_artefact" / "j18r_torch_run" / name
    out.mkdir(parents=True, exist_ok=True)
    return out.relative_to(root)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _FakeBatchEncoding(dict):
    """`BatchEncoding` minimal : porte `input_ids` et se déplace (`.to(device)`)."""

    def to(self, device):
        return _FakeBatchEncoding(input_ids=self["input_ids"].to(device))


class FakeTokenizer:
    """Tokenizer duck-typed : mêmes identifiants en base et sous gabarit de conversation.

    R-J18R-13 : `apply_chat_template` rend un **texte** (`tokenize=False`) ; le rendu est
    ensuite encodé par le tokenizer. La voie `tokenize=True` n'est plus employée.
    """

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        if return_tensors == "pt":
            return _FakeBatchEncoding(input_ids=torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long))
        return _FakeBatchEncoding(input_ids=[1, 2, 3, 4, 5, 6])

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    ):
        assert tokenize is False
        return "<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>\n<|im_start|>assistant\n"


def _raw_rows(model, prompt_ids, arm: str, blocks: int) -> list[dict]:
    """Shard brut §7.3b de la fixture : marginaux **réels** du modèle minuscule, non étiquetés
    (J15 n'étiquette pas ; J18R étiquette à l'ingestion, R-J18R-08).

    Les clés `(prompt, config_index, block_index, position, noise)` sont **alignées** sur celles
    de l'artefact J15 gelé (`N4_V6`, prompt `p01`) : l'ancrage C9 est donc *mesurable*, et son
    échec porte sur les **valeurs** (le modèle minuscule ne reproduit pas Qwen3-0.6B)."""
    forward = j18r.model_forward(model, prompt_ids, model.device, arithmetic_dtype="float64")
    support = j18r.build_economical_support(forward, 4, 6, option="A", arithmetic_dtype="float64")
    sequences, log_joint = support.enumerate_joint()
    provider = j18r.ProductionScoreProvider(Qwen3TeacherScorer(model, device=model.device), prompt_ids)
    observations = j18r.process_configuration(
        arm=arm,
        prompt="p01",
        config_index="N4_V6",
        support=support,
        sequences=sequences,
        log_joint=log_joint,
        provider=provider,
        seed=j18r.SEED_ANCHOR,
        blocks=blocks,
        integrity=j18r.IntegrityStats(),
        replay_stats=j18r.ReplayStats(),
        k=None,
    )
    return [{key: value for key, value in row.items() if key != "arm"} for row in map(lambda o: o.record(), observations)]


def test_j18r_execute_full_run_on_tiny_gpu_model(monkeypatch) -> None:
    """T5 complet : `execute` entier sur la fixture, GPU réel. Sorties dans `_artefact/`
    (règle 9). L'ancrage d'un modèle minuscule ne satisfait pas C9 mesuré sur l'artefact J15
    gelé : `STOP` (priorité 1, §4.5) est la décision **attendue**, et les publications sont
    écrites (§7.2)."""
    if not torch.cuda.is_available():
        pytest.skip("pas de GPU sur ce poste")
    transformers = pytest.importorskip("transformers")
    model = FakeModel()
    monkeypatch.setattr(j18r, "load_scorer_module", lambda: scorer_module)
    monkeypatch.setattr(
        scorer_module, "load_teacher", lambda name, *, device, dtype: (model.to(device), name)
    )
    monkeypatch.setattr(j18r, "verify_model_source_in_repo", lambda model, root: {"model_source_in_repo": True})
    monkeypatch.setattr(j18r, "load_prompt_set", lambda path: [("p01", "alpha"), ("p02", "beta")])
    monkeypatch.setattr(transformers, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda source: FakeTokenizer()))

    def fake_subprocess(invocation, root, **kwargs):
        tokens = shlex.split(invocation)
        output = root / Path(tokens[tokens.index("--output") + 1])
        arm = "anchor" if "anchor" in output.name else "sensitivity_fp32"
        rows = _raw_rows(model, torch.tensor([1, 2, 3, 4, 5], device=model.device), arm, 2)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
        return {
            "returncode": 0,
            "group": {
                "pgid": 0,
                "limit_mib": kwargs.get("limit_mib"),
                "limit_applied": True,
                "stop_after_seconds": kwargs.get("stop_after_seconds"),
                "stop_signal": None,
                "survivors_after_stop": 0,
                "verified_after_stop": True,
                "elapsed_seconds": 0.0,
                "cgroup_memory_peak_bytes": None,
            },
            "resisted": False,
        }

    monkeypatch.setattr(j18r, "run_arm_subprocess", fake_subprocess)
    out = _scratch_output("execute_gpu")
    argv = [
        "--model", "Qwen/Qwen3-0.6B", "--device", "cuda", "--dtype-anchor", "bfloat16",
        "--dtype-decision", "float32", "--dtype-control", "float32", "--dtype-sensitivity", "float32",
        "--prompt-file", "modules/dopd/eval/annexe_d_qwen3_0.6b_multiprompt/prompts.json",
        "--blocks", "1", "--seed", "20261027", "--bootstrap-seed", "20261028",
        "--bootstrap-iterations", "200", "--anchor-seed", "20261023", "--anchor-bootstrap-seed", "20261024",
        "--replay-fraction", "0.05", "--replay-tolerance", "1e-3", "--n6-chat-template",
        "--n6-max-seconds", "1800", "--max-seconds", "14400", "--max-gpu-mib", "16384",
        "--max-rss-gib", "24", "--output-dir", str(out),
    ]
    outcome = j18r.execute(j18r.build_parser().parse_args(argv), argv)
    base = _TESTS.parents[2] / out
    assert outcome["results_written"] is True
    results = json.loads((base / "results.json").read_text(encoding="utf-8"))
    assert results["runtime"]["device"] == "cuda"
    assert results["runtime"]["gpu_peak_mib"] > 0.0
    assert results["runtime"]["dtypes"]["anchor"] == "bfloat16"
    # C9 : le mini-modèle ne reproduit pas Qwen3-0.6B, mais le vol d'essai (1 bloc) ne peut
    # **pas** conclure à une dérive (échantillon non conforme au PRE) → issue INVALIDE.
    assert results["criteria"]["C9"]["evaluable"] is False
    assert results["criteria"]["C9"]["evaluable_reason"] == "insufficient_blocks"
    assert results["criteria"]["C9"]["holds"] is False
    assert results["decision"]["status"] == "INVALIDE"
    assert results["decision"]["priority"] == 2
    # §7.2 : les bruts ne sont pas étiquetés, l'ancrage a son shard étiqueté, `results.json` hashé
    raw = json.loads((base / "arm_anchor_raw.json").read_text(encoding="utf-8"))
    assert raw["rows"] and not [row for row in raw["rows"] if "arm" in row]
    labeled = json.loads((base / "arm_anchor.json").read_text(encoding="utf-8"))
    assert all(row["arm"] == "anchor" for row in labeled["records"])
    assert not (base / "arm_sensitivity.json").exists()
    sidecar = json.loads((base / "lineage.json").read_text(encoding="utf-8"))
    produced = sorted(
        [
            "arm_anchor_raw.json",
            "arm_sensitivity_raw.json",
            "arm_anchor.json",
            "arm_control.json",
            "arm_decision.json",
            "arm_n6.json",
            "results.json",
        ]
    )
    assert sorted(sidecar["artifact"]["hashes"]) == produced
    for name, digest in sidecar["artifact"]["hashes"].items():
        assert digest == _sha256(base / name), name
