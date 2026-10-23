"""Tests des contrats d'interface d-OPD (J13).

Ces tests vérifient les **contrats**, pas des modèles : ils utilisent des
implémentations factices (mocks) de `TeacherScorer`, `RolloutRecorder`,
`TaskVerifier` et `LineageWriter`. Aucun test ne charge de checkpoint, n'utilise
torch ni un GPU.

    pytest modules/dopd/tests/test_dopd_interfaces.py -q
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from academiia.dopd import (
    ARTIFACT_KINDS,
    DEVELOPMENT_MODEL_ID,
    LINEAGE_SCHEMA,
    PRODUCER_MODULE,
    ArtifactRef,
    CompletedBlock,
    DenoisingState,
    DOPDConfig,
    LineageManifest,
    LineageWriter,
    RolloutRecorder,
    ScoringRequest,
    ScoringResult,
    TaskVerifier,
    TeacherScorer,
)
from academiia.dopd.config import PACKAGE_VERSION
from academiia.dopd.target import compute_future_aware_target

# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class FakeTeacherScorer(TeacherScorer):
    """Professeur factice : prior uniforme et scores de suffixe linéaires.

    Reproduit la structure attendue par `TeacherScorer.score` : un prior plein
    vocabulaire et un score par branche, aligné sur la liste des branches.
    """

    def __init__(self, vocab_size: int = 8, score_slope: float = -0.5):
        self._vocab_size = vocab_size
        self._score_slope = score_slope
        self.calls: list[tuple] = []

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def causal_prior(self, prefix_ids):
        self.calls.append(("prior", tuple(prefix_ids)))
        return np.full(self._vocab_size, 1.0 / self._vocab_size, dtype=np.float64)

    def suffix_loglikelihood(self, prefix_ids, candidate_ids, suffix_ids):
        self.calls.append(("suffix", tuple(prefix_ids), tuple(candidate_ids), tuple(suffix_ids)))
        return np.array(
            [self._score_slope * float(token) for token in candidate_ids],
            dtype=np.float64,
        )


class FakeRolloutRecorder(RolloutRecorder):
    """Enregistreur factice : deux états visités pour le bloc 0, un pour le bloc 1."""

    def __init__(self):
        self._states: list[DenoisingState] = []
        self._blocks: list[CompletedBlock] = []

    def record_state(self, state: DenoisingState) -> None:
        self._states.append(state)

    def record_block(self, block: CompletedBlock) -> None:
        self._blocks.append(block)

    def states(self, block_index: int | None = None) -> tuple[DenoisingState, ...]:
        if block_index is None:
            return tuple(self._states)
        return tuple(s for s in self._states if s.block_index == block_index)

    def blocks(self) -> tuple[CompletedBlock, ...]:
        return tuple(self._blocks)


class FakeVerifier(TaskVerifier):
    """Vérificateur factice : accepte les rolls contenant « OK »."""

    def __init__(self, accepted: tuple[str, ...] = ("OK",)):
        self._accepted = accepted
        self.seen: list[str] = []

    def verify(self, completion: str) -> bool:
        self.seen.append(completion)
        return any(marker in completion for marker in self._accepted)


class FakeLineageWriter(LineageWriter):
    """Écrivain factice : garde le manifeste en mémoire, n'écrit rien."""

    def __init__(self):
        self.written: list[LineageManifest] = []

    def write(self, manifest: LineageManifest) -> str:
        self.written.append(manifest)
        return "lineage.json"


# ---------------------------------------------------------------------------
# TeacherScorer
# ---------------------------------------------------------------------------


def test_teacher_scorer_is_abstract():
    """Le contrat est abstrait : on ne l'instancie pas directement."""
    with pytest.raises(TypeError):
        TeacherScorer()  # type: ignore[abstract]


def test_teacher_scorer_contract_signature():
    """Les trois méthodes du contrat ont une signature stable."""
    assert inspect.isabstract(TeacherScorer)
    assert TeacherScorer.__abstractmethods__ == frozenset(
        {"vocab_size", "causal_prior", "suffix_loglikelihood"}
    )
    prior = inspect.signature(TeacherScorer.causal_prior)
    assert list(prior.parameters) == ["self", "prefix_ids"]
    suffix = inspect.signature(TeacherScorer.suffix_loglikelihood)
    assert list(suffix.parameters) == [
        "self", "prefix_ids", "candidate_ids", "suffix_ids",
    ]


def test_teacher_scorer_score_batches_one_call():
    """`score` fait un seul prior et un seul lot de branches (App. B)."""
    scorer = FakeTeacherScorer(vocab_size=8)
    request = ScoringRequest(
        prefix_ids=(1, 2, 3),
        suffix_ids=(9, 9),
        candidate_ids=(4, 5, 6),
        reference_id=7,
    )
    result = scorer.score(request)

    kinds = [call[0] for call in scorer.calls]
    assert kinds == ["prior", "suffix"], "un prior + un lot de suffixe"

    suffix_call = scorer.calls[1]
    assert suffix_call[2] == (4, 5, 6, 7), "les candidats + la branche de référence"

    assert isinstance(result, ScoringResult)
    assert result.prior.shape == (8,)
    assert result.candidate_ids == (4, 5, 6)
    assert result.candidate_loglikelihoods.shape == (3,)
    assert result.reference_loglikelihood == pytest.approx(-0.5 * 7)


def test_teacher_scorer_score_does_not_duplicate_reference_candidate():
    """Si x̂_i est déjà dans V_i, la branche de référence n'est pas dupliquée."""
    scorer = FakeTeacherScorer(vocab_size=8)
    request = ScoringRequest(
        prefix_ids=(1,),
        suffix_ids=(),
        candidate_ids=(4, 7),
        reference_id=7,
    )
    scorer.score(request)
    assert scorer.calls[1][2] == (4, 7)


def test_teacher_scorer_score_rejects_mismatched_prior():
    """Un prior de mauvaise taille est refusé : le contrat est explicite."""
    class BadScorer(FakeTeacherScorer):
        def causal_prior(self, prefix_ids):
            return np.ones(3, dtype=np.float64) / 3.0

    request = ScoringRequest(
        prefix_ids=(1,), suffix_ids=(), candidate_ids=(2,), reference_id=3
    )
    with pytest.raises(ValueError, match="vocab_size"):
        BadScorer(vocab_size=8).score(request)


def test_teacher_scorer_score_rejects_mismatched_scores():
    """Un score par branche mal aligné est refusé."""
    class BadScorer(FakeTeacherScorer):
        def suffix_loglikelihood(self, prefix_ids, candidate_ids, suffix_ids):
            return np.zeros(1, dtype=np.float64)

    request = ScoringRequest(
        prefix_ids=(1,), suffix_ids=(), candidate_ids=(2, 3), reference_id=4
    )
    with pytest.raises(ValueError, match="aligné"):
        BadScorer(vocab_size=8).score(request)


def test_scoring_result_rejects_invalid_prior():
    """Le prior doit être une vraie distribution : c'est la base du fallback."""
    with pytest.raises(ValueError, match="sommer à 1"):
        ScoringResult(
            prior=np.array([0.5, 0.2]),
            candidate_ids=(1,),
            candidate_loglikelihoods=np.array([-1.0]),
            reference_loglikelihood=-1.0,
        )
    with pytest.raises(ValueError, match="négative"):
        ScoringResult(
            prior=np.array([1.5, -0.5]),
            candidate_ids=(1,),
            candidate_loglikelihoods=np.array([-1.0]),
            reference_loglikelihood=-1.0,
        )


def test_scoring_result_accepts_non_finite_scores():
    """Un score non fini n'est pas une erreur du scorer : la cible le dégrade."""
    result = ScoringResult(
        prior=np.array([0.5, 0.5]),
        candidate_ids=(1,),
        candidate_loglikelihoods=np.array([np.nan]),
        reference_loglikelihood=-1.0,
    )
    assert np.isnan(result.candidate_loglikelihoods[0])


def test_scoring_result_rejects_size_mismatch():
    """Les scores et les identifiants de candidats doivent être alignés."""
    with pytest.raises(ValueError, match="même taille"):
        ScoringResult(
            prior=np.array([0.5, 0.5]),
            candidate_ids=(1, 2),
            candidate_loglikelihoods=np.array([-1.0]),
            reference_loglikelihood=-1.0,
        )


def test_scoring_request_rejects_duplicates_and_empty():
    """Une requête de scoring est validée à la construction."""
    with pytest.raises(ValueError, match="vide"):
        ScoringRequest(prefix_ids=(1,), suffix_ids=(), candidate_ids=(), reference_id=2)
    with pytest.raises(ValueError, match="doublons"):
        ScoringRequest(
            prefix_ids=(1,), suffix_ids=(), candidate_ids=(2, 2), reference_id=3
        )


def test_scorer_output_feeds_the_target():
    """Chaîne complète : sortie du scorer -> cible future-aware (eq. 18).

    Vérifie l'intégration entre le contrat et la brique de cible : le prior et
    les scores du scorer sont directement exploitables par
    `compute_future_aware_target`.
    """
    scorer = FakeTeacherScorer(vocab_size=8, score_slope=-0.5)
    request = ScoringRequest(
        prefix_ids=(1, 2),
        suffix_ids=(9, 9),
        candidate_ids=(4, 5),
        reference_id=7,
    )
    result = scorer.score(request)
    target = compute_future_aware_target(
        result.prior,
        candidate_indices=result.candidate_ids,
        candidate_loglikelihoods=result.candidate_loglikelihoods,
        reference_loglikelihood=result.reference_loglikelihood,
    )
    assert target.applied
    assert target.target.shape == (8,)
    assert target.target.sum() == pytest.approx(1.0, abs=1e-6)
    # Le candidat de référence garde un delta nul : son poids est inchangé.
    assert result.candidate_loglikelihoods[0] == pytest.approx(
        result.reference_loglikelihood + 0.5 * (7 - 4)
    )


# ---------------------------------------------------------------------------
# RolloutRecorder
# ---------------------------------------------------------------------------


def test_rollout_recorder_is_abstract():
    with pytest.raises(TypeError):
        RolloutRecorder()  # type: ignore[abstract]


def test_rollout_recorder_contract_signature():
    assert inspect.isabstract(RolloutRecorder)
    assert RolloutRecorder.__abstractmethods__ == frozenset(
        {"record_state", "record_block", "states", "blocks"}
    )


def test_rollout_recorder_records_states_and_blocks():
    """Les états visités et les blocs complétés sont enregistrés séparément."""
    recorder = FakeRolloutRecorder()
    recorder.record_state(DenoisingState(0, 0, (5, 5, 5, 5), (0, 1, 2, 3)))
    recorder.record_state(DenoisingState(0, 1, (5, 6, 5, 5), (2, 3)))
    recorder.record_state(DenoisingState(1, 0, (7, 7, 7, 7), (0, 1)))
    recorder.record_block(CompletedBlock(0, (5, 6, 7, 8)))
    recorder.record_block(CompletedBlock(1, (7, 8, 9, 10)))

    assert len(recorder.states()) == 3
    assert len(recorder.states(0)) == 2
    assert len(recorder.states(1)) == 1
    assert [b.block_index for b in recorder.blocks()] == [0, 1]
    assert recorder.blocks()[0].token_ids == (5, 6, 7, 8)


def test_rollout_recorder_sample_state_skips_fully_visible_states():
    """Un état sans position masquée n'a rien à corriger : il n'est jamais tiré.

    Règle du papier (App. C.2) : on échantillonne parmi les tours visités où
    le bloc a au moins une position masquée.
    """
    recorder = FakeRolloutRecorder()
    recorder.record_state(DenoisingState(0, 0, (5, 6, 7, 8), ()))  # tout visible
    recorder.record_state(DenoisingState(0, 1, (5, 6, 5, 5), (2,)))

    rng = np.random.default_rng(0)
    for _ in range(20):
        state = recorder.sample_state(0, rng)
        assert state.step == 1
        assert state.masked_positions == (2,)


def test_rollout_recorder_sample_state_raises_without_candidate():
    """Aucun état éligible : erreur explicite, pas de tirage arbitraire."""
    recorder = FakeRolloutRecorder()
    recorder.record_state(DenoisingState(0, 0, (5, 6, 7, 8), ()))
    with pytest.raises(LookupError, match="position masquée"):
        recorder.sample_state(0, np.random.default_rng(0))


def test_denoising_state_rejects_out_of_range_positions():
    """Une position masquée hors du bloc est refusée."""
    with pytest.raises(ValueError, match="hors de"):
        DenoisingState(0, 0, (1, 2), (5,))


def test_denoising_state_rejects_negative_step():
    with pytest.raises(ValueError, match="step"):
        DenoisingState(0, -1, (1, 2), (0,))


# ---------------------------------------------------------------------------
# TaskVerifier
# ---------------------------------------------------------------------------


def test_task_verifier_is_abstract():
    with pytest.raises(TypeError):
        TaskVerifier()  # type: ignore[abstract]


def test_task_verifier_contract_signature():
    assert inspect.isabstract(TaskVerifier)
    assert TaskVerifier.__abstractmethods__ == frozenset({"verify"})


def test_task_verifier_gate_open_when_verified():
    """Porte ouverte : le vérificateur accepte le rollout."""
    verifier = FakeVerifier()
    config = DOPDConfig()
    assert verifier.gate_open("42 OK", config=config) is True
    assert verifier.correction_label("42 OK", config=config) == 1
    # `correction_label` délègue à `gate_open` : le vérificateur est donc
    # appelé une fois par appel de porte.
    assert verifier.seen == ["42 OK", "42 OK"]


def test_task_verifier_gate_closed_when_rejected():
    """Porte fermée : le rollout garde la cible causale standard."""
    verifier = FakeVerifier()
    config = DOPDConfig()
    assert verifier.gate_open("42 wrong", config=config) is False
    assert verifier.correction_label("42 wrong", config=config) == 0


def test_task_verifier_gate_disabled_skips_verification():
    """Porte désactivée : la correction s'applique inconditionnellement.

    C'est le mode de repli documenté quand aucun vérificateur n'est disponible
    (DOPD_SPEC §3.4).
    """
    verifier = FakeVerifier()
    config = DOPDConfig(correctness_gate=False)
    assert verifier.gate_open("42 wrong", config=config) is True
    assert verifier.correction_label("42 wrong", config=config) == 1
    assert verifier.seen == [], "le vérificateur ne doit pas être appelé"


# ---------------------------------------------------------------------------
# LineageWriter
# ---------------------------------------------------------------------------


def test_lineage_writer_is_abstract():
    with pytest.raises(TypeError):
        LineageWriter()  # type: ignore[abstract]


def test_lineage_writer_contract_signature():
    assert inspect.isabstract(LineageWriter)
    assert LineageWriter.__abstractmethods__ == frozenset({"write"})


def test_lineage_writer_builds_adr002_manifest():
    """Le manifeste respecte le schéma ADR-002 et `producer.module = "dopd"`."""
    writer = FakeLineageWriter()
    manifest = writer.build_manifest(
        artifact=ArtifactRef(path="checkpoints/dopd_smoke", kind="report"),
        parent=ArtifactRef(path="checkpoints/qwen3_0_6b", kind="checkpoint"),
        config=LineageWriter.config_block(DOPDConfig(), seed=1234),
        data={"prompts_sha256": "abc", "verifier": "toy"},
        verification={"kind": "load_smoke"},
        command="python modules/dopd/harness/smoke_qwen3.py",
    )
    payload = manifest.to_dict()

    assert payload["schema"] == LINEAGE_SCHEMA == "academiia.lineage/1"
    assert payload["producer"]["module"] == PRODUCER_MODULE == "dopd"
    assert payload["producer"]["version"] == PACKAGE_VERSION
    assert payload["producer"]["command"].endswith("smoke_qwen3.py")
    assert payload["artifact"]["kind"] == "report"
    assert payload["parent"]["kind"] == "checkpoint"
    assert payload["config"]["block_size"] == 4
    assert payload["config"]["candidate_k"] == 16
    assert payload["config"]["seed"] == 1234
    assert payload["data"]["prompts_sha256"] == "abc"
    assert payload["verification"]["kind"] == "load_smoke"
    # Le bloc `vector` est propre à la fusion A2D : d-OPD n'en produit pas.
    assert "vector" not in payload


def test_lineage_manifest_is_json_serialisable():
    """Le manifeste est directement sérialisable en JSON (écriture future)."""
    writer = FakeLineageWriter()
    manifest = writer.build_manifest(
        artifact=ArtifactRef(path="_artefact/smoke", kind="report"),
        parent=None,
        config=LineageWriter.config_block(DOPDConfig()),
        data={},
        verification={},
        command="python -c 'smoke'",
    )
    payload = json.loads(json.dumps(manifest.to_dict()))
    assert payload["parent"] is None, "un artefact racine déclare parent = null"
    assert payload["schema"] == LINEAGE_SCHEMA


def test_lineage_manifest_rejects_foreign_producer():
    """Un module producteur inconnu est refusé (ADR-002 : `producer.module`)."""
    with pytest.raises(ValueError, match="producer.module"):
        LineageManifest(
            artifact=ArtifactRef(path="_artefact/x", kind="report"),
            parent=None,
            producer={"module": "a2d", "version": "0.1.0"},
            config={},
            data={},
            verification={},
        )


def test_artifact_ref_rejects_absolute_and_escaping_paths():
    """Les chemins d'artefact sont relatifs à la racine du dépôt (ADR-002)."""
    with pytest.raises(ValueError, match="relatif"):
        ArtifactRef(path="/checkpoints/x", kind="checkpoint")
    with pytest.raises(ValueError, match="remonter"):
        ArtifactRef(path="../outside", kind="checkpoint")
    with pytest.raises(ValueError, match="kind"):
        ArtifactRef(path="checkpoints/x", kind="model")


def test_artifact_kinds_match_adr002():
    assert ARTIFACT_KINDS == ("checkpoint", "adapter", "eval", "report")


def test_development_model_is_qwen3_0_6b():
    """Le modèle de développement retenu est Qwen3-0.6B (DOPD_SPEC §6.3)."""
    assert DEVELOPMENT_MODEL_ID == "Qwen/Qwen3-0.6B"


# ---------------------------------------------------------------------------
# Non-régression ADR-001 et packaging
# ---------------------------------------------------------------------------


PACKAGE_DIR = Path(__file__).resolve().parents[1] / "dopd"


def test_interfaces_module_does_not_import_heavy_stack():
    """Le fichier d'interfaces reste pur : ni torch, ni transformers."""
    import ast

    tree = ast.parse((PACKAGE_DIR / "interfaces.py").read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    assert not roots & {"torch", "transformers", "deepspeed"}


def test_interfaces_module_imports_only_core_from_the_repo():
    """ADR-001 : le module d'interfaces n'importe que ``academiia.core``."""
    import ast

    tree = ast.parse((PACKAGE_DIR / "interfaces.py").read_text(encoding="utf-8"))
    repo_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("academiia."):
                repo_modules.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("academiia."):
                    repo_modules.add(alias.name)
    assert repo_modules <= {"academiia.core"}, repo_modules
