"""Tests de MUTATION de la bibliotheque A2D (J6bis B6).

Principe : on produit d'abord un merge CORRECT, puis on le MUTE, et on exige
que la verification (ou le writer) ECHOUE. Un test qui passe est un test qui
prouve qu'un defaut de la sorte serait attrape.

Chaque mutation est un cas de non-regression des constats :
  * BLK-01 : `ablation-dp-minus-tau-d` sans tau_A ne doit pas etre un no-op
    certifie OK ;
  * BLK-02 : un tau_D partiel ne doit pas transformer un merge en transfert
    sans alerte ;
  * MAJ-01 : `verify_merge --mode merge` sans tau_D doit etre refuse ;
  * MAJ-02 : une inversion de signe doit etre detectee (projection signee) ;
  * MAJ-03 : l'intersection des lignes d'embedding doit etre verifiable.

Lancer : pytest scripts/tests -q
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from a2d import (CheckpointPairSource, CoverageError, MergeConfig, MergeError,
                 MissingVectorError, compute_effective_coefficients, get_mapping,
                 merge_checkpoint)

from test_a2d_library import (QWEN, ar_and_diffusion, build_checkpoint, config,
                               pair, tensors_of)

SHARD = "model-00001-of-00001.safetensors"


# --------------------------------------------------------------------------- #
# Outils de mutation
# --------------------------------------------------------------------------- #

def read_shard(path: Path) -> dict:
    return load_file(str(path / SHARD))


def write_shard(path: Path, tensors: dict) -> None:
    save_file(tensors, str(path / SHARD))


def mutate(path: Path, name: str, fn) -> None:
    """Applique `fn(tensor)` au tenseur `name` du checkpoint ecrit."""
    tensors = read_shard(path)
    tensors[name] = fn(tensors[name])
    write_shard(path, tensors)


def verify(merged: Path, reference: Path, **kwargs):
    cfg = config(kwargs.pop("mode"), reference, merged, **kwargs)
    return compute_effective_coefficients(merged, reference, cfg)


# --------------------------------------------------------------------------- #
# Mutations parametrees sur un merge correct (mode merge, lambda=0.6)
# --------------------------------------------------------------------------- #

@pytest.fixture
def merged(tmp_path):
    """Un merge A2D correct : D_p + 0.6*(tau_A - tau_D), 8 tenseurs."""
    ar_base, ar_post, diff_base, diff_post = ar_and_diffusion(tmp_path)
    tau_a = CheckpointPairSource(ar_post, ar_base)
    tau_d = CheckpointPairSource(diff_post, diff_base)
    out = tmp_path / "out"
    merge_checkpoint(config("merge", diff_post, out, lam=0.6, tau_a=tau_a, tau_d=tau_d))
    return {"out": out, "diff_post": diff_post, "diff_base": diff_base,
            "ar_base": ar_base, "ar_post": ar_post, "tau_a": tau_a, "tau_d": tau_d}


def _mutate_sign(merged, name):
    """Inversion de signe : out = ref - lambda*(tau_A - tau_D)."""
    ref = tensors_of(merged["diff_post"])[name]
    out = read_shard(merged["out"])[name]
    mutate(merged["out"], name, lambda t: (2 * ref.to(torch.float32) - t).to(t.dtype))


def _mutate_tau_swap(merged, name):
    """Permutation tau_A <-> tau_D : out = ref + lambda*(tau_D - tau_A)."""
    ref = tensors_of(merged["diff_post"])[name]
    out = read_shard(merged["out"])[name]
    # out actuel = ref + l*(tau_A - tau_D) ; on veut ref + l*(tau_D - tau_A)
    mutate(merged["out"], name, lambda t: (2 * ref.to(torch.float32) - t).to(t.dtype))


def _mutate_unfused(merged, name):
    """Tenseur non fusionne altere (bit-identique casse)."""
    mutate(merged["out"], name, lambda t: t + 1.0)


def _mutate_scale(merged, name):
    """Coefficient faux : out = ref + 0.3*(tau_A - tau_D) au lieu de 0.6."""
    ref = tensors_of(merged["diff_post"])[name].to(torch.float32)
    out = read_shard(merged["out"])[name].to(torch.float32)
    delta = (out - ref) / 0.6
    mutate(merged["out"], name,
           lambda t: (ref + 0.3 * delta).to(t.dtype))


MUTATIONS = [
    pytest.param(_mutate_sign, "model.layers.0.self_attn.q_proj.weight",
                 id="inversion_de_signe"),
    pytest.param(_mutate_tau_swap, "model.layers.0.self_attn.q_proj.weight",
                 id="permutation_tau_A_tau_D"),
    pytest.param(_mutate_scale, "model.layers.0.self_attn.q_proj.weight",
                 id="coefficient_faux"),
]


@pytest.mark.parametrize("mutate_fn,target", MUTATIONS)
def test_mutated_merge_must_fail_verification(merged, mutate_fn, target):
    """Une mutation du merge doit etre detectee par la verification."""
    mutate_fn(merged, target)
    report = verify(merged["out"], merged["diff_post"], mode="merge", lam=0.6,
                    tau_a=merged["tau_a"], tau_d=merged["tau_d"])
    assert not report["ok"], "la mutation n'a pas ete detectee"
    assert report["failures"]


def test_correct_merge_passes_verification(merged):
    """Controle positif : le merge intact passe (sinon les tests mutants sont vides)."""
    report = verify(merged["out"], merged["diff_post"], mode="merge", lam=0.6,
                    tau_a=merged["tau_a"], tau_d=merged["tau_d"])
    assert report["ok"], report["failures"]
    assert report["c_a_eff"] == pytest.approx(0.6, abs=1e-3)
    assert report["c_d_eff"] == pytest.approx(-0.6, abs=1e-3)


def test_wrong_mode_must_fail(merged):
    """Un merge verifie comme un transfert doit echouer (MAJ-02 / mode errone)."""
    report = verify(merged["out"], merged["diff_post"], mode="transfer", alpha=0.6,
                    tau_a=merged["tau_a"], tau_d=merged["tau_d"])
    assert not report["ok"]
    assert any("ULP" in f or "c_a_eff" in f or "non fusionne" in f
               for f in report["failures"]), report["failures"]


def test_alpha_lambda_swap_must_fail(merged):
    """Permutation alpha <-> lambda : le merge 0.6 verifie comme transfert 0.6 echoue."""
    report = verify(merged["out"], merged["diff_post"], mode="transfer", alpha=0.6,
                    tau_a=merged["tau_a"], tau_d=merged["tau_d"])
    assert not report["ok"]


def test_unfused_tensor_altered_must_fail(tmp_path):
    """Un tenseur NON fusionne modifie doit etre detecte (identite binaire)."""
    spec = dict(QWEN)
    spec["model.layers.0.mlp.up_proj.weight"] = (6, 4)
    ar_base, ar_post = pair(tmp_path, spec=QWEN, base_name="ar_base", post_name="ar_post")
    recipient = build_checkpoint(tmp_path, spec, seed=71, name="recipient")
    tau_a = CheckpointPairSource(ar_post, ar_base)
    out = tmp_path / "out"
    result = merge_checkpoint(config("transfer", recipient, out, alpha=0.5, tau_a=tau_a))
    # le tenseur absent de tau_A n'est pas dans le plan : il reste inchange
    assert result.n_fused == len(QWEN)
    mutate(out, "model.layers.0.mlp.up_proj.weight", lambda t: t + 1.0)
    report = verify(out, recipient, mode="transfer", alpha=0.5, tau_a=tau_a)
    assert not report["ok"]
    assert any("NON fusionne modifie" in f for f in report["failures"]), report["failures"]


# --------------------------------------------------------------------------- #
# BLK-01 : ablation-dp-minus-tau-d sans tau_A
# --------------------------------------------------------------------------- #

def test_ablation_minus_without_tau_a_is_not_a_noop(tmp_path):
    """BLK-01 : sans tau_A, l'ablation D_p - lambda*tau_D doit FUSIONNER tau_D."""
    ar_base, ar_post, diff_base, diff_post = ar_and_diffusion(tmp_path)
    tau_d = CheckpointPairSource(diff_post, diff_base)
    out = tmp_path / "out"
    result = merge_checkpoint(config("ablation-dp-minus-tau-d", diff_post, out, lam=0.5,
                                     tau_a=None, tau_d=tau_d))
    assert result.n_fused == len(QWEN)
    src, dst = tensors_of(diff_post), tensors_of(out)
    d_b = tensors_of(diff_base)
    for name in QWEN:
        expected = src[name] - 0.5 * (src[name] - d_b[name])
        assert not torch.equal(src[name], dst[name]), name
        assert torch.allclose(dst[name], expected, atol=1e-6), name
    # la verification doit retrouver le coefficient SIGNE
    report = verify(out, diff_post, mode="ablation-dp-minus-tau-d", lam=0.5,
                    tau_a=None, tau_d=tau_d)
    assert report["ok"], report["failures"]
    assert report["c_d_eff"] == pytest.approx(-0.5, abs=1e-3)


def test_merge_without_tau_a_is_refused(tmp_path):
    """MAJ-01 : le mode merge exige tau_A (et tau_D)."""
    ar_base, ar_post, diff_base, diff_post = ar_and_diffusion(tmp_path)
    tau_d = CheckpointPairSource(diff_post, diff_base)
    with pytest.raises(MissingVectorError):
        merge_checkpoint(config("merge", diff_post, tmp_path / "out", lam=0.6,
                                tau_a=None, tau_d=tau_d))


def test_verify_merge_cli_mode_merge_without_tau_d_is_refused(tmp_path):
    """MAJ-01 (CLI) : `verify_merge --mode merge` sans tau_D sort en erreur."""
    ar_base, ar_post, diff_base, diff_post = ar_and_diffusion(tmp_path)
    tau_a = CheckpointPairSource(ar_post, ar_base)
    out = tmp_path / "out"
    merge_checkpoint(config("transfer", diff_base, out, alpha=0.6, tau_a=tau_a))
    cmd = [sys.executable, str(SCRIPTS / "verify_merge.py"), "--merged", str(out),
           "--reference", str(diff_base), "--mode", "merge",
           "--tau-a-post", str(ar_post), "--tau-a-base", str(ar_base),
           "--lambda", "0.6"]
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 2, done.stdout + done.stderr
    assert "tau_D" in done.stdout and "merge" in done.stdout


# --------------------------------------------------------------------------- #
# BLK-02 : tau_D partiel
# --------------------------------------------------------------------------- #

PARTIAL = {"model.layers.0.self_attn.q_proj.weight": (4, 4)}


def test_partial_tau_d_is_refused_under_strict(tmp_path):
    """BLK-02 : sous `strict`, un tau_D partiel est refuse avec la liste des modules."""
    ar_base, ar_post, diff_base, diff_post = ar_and_diffusion(tmp_path)
    partial_base = build_checkpoint(tmp_path, PARTIAL, seed=81, name="partial_base")
    partial_post = build_checkpoint(tmp_path, PARTIAL, seed=82, name="partial_post")
    tau_a = CheckpointPairSource(ar_post, ar_base)
    tau_d = CheckpointPairSource(partial_post, partial_base)
    with pytest.raises(CoverageError) as excinfo:
        merge_checkpoint(config("merge", diff_post, tmp_path / "out", lam=0.6,
                                tau_a=tau_a, tau_d=tau_d))
    message = str(excinfo.value)
    assert "couverture inegale" in message
    assert "q_proj" not in message.split("couverts par tau_A")[0]
    # la liste des modules fautifs est presente
    assert "o_proj" in message


def test_partial_tau_d_zero_fill_is_declared(tmp_path):
    """BLK-02 : sous `zero-fill`, le module sans tau_D est traite comme nul ET declare."""
    ar_base, ar_post, diff_base, diff_post = ar_and_diffusion(tmp_path)
    partial_base = build_checkpoint(tmp_path, PARTIAL, seed=81, name="partial_base")
    partial_post = build_checkpoint(tmp_path, PARTIAL, seed=82, name="partial_post")
    tau_a = CheckpointPairSource(ar_post, ar_base)
    tau_d = CheckpointPairSource(partial_post, partial_base)
    out = tmp_path / "out"
    result = merge_checkpoint(config("merge", diff_post, out, lam=0.6, tau_a=tau_a,
                                     tau_d=tau_d, coverage="zero-fill"))
    coverage = result.manifest["coverage"]
    assert coverage["policy"] == "zero-fill"
    assert coverage["n_A_only"] == len(QWEN) - 1
    assert coverage["n_both"] == 1
    assert coverage["n_zero_filled"] == len(QWEN) - 1
    # le module sans tau_D est bien ref + lambda*tau_A (tau_D = 0)
    src, dst = tensors_of(diff_post), tensors_of(out)
    ar_b, ar_p = tensors_of(ar_base), tensors_of(ar_post)
    name = "model.layers.0.self_attn.o_proj.weight"
    expected = src[name] + 0.6 * (ar_p[name] - ar_b[name])
    assert torch.allclose(dst[name], expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# Plan vide
# --------------------------------------------------------------------------- #

def test_empty_plan_is_refused(tmp_path):
    """Un plan vide doit etre refuse (B2), sauf --allow-empty."""
    ar_base, ar_post, diff_base, diff_post = ar_and_diffusion(tmp_path)
    other = {"model.layers.9.self_attn.q_proj.weight": (4, 4)}
    other_base = build_checkpoint(tmp_path, other, seed=91, name="other_base")
    other_post = build_checkpoint(tmp_path, other, seed=92, name="other_post")
    tau_d = CheckpointPairSource(other_post, other_base)
    with pytest.raises(MergeError) as excinfo:
        merge_checkpoint(config("ablation-dp-minus-tau-d", diff_post, tmp_path / "out",
                                lam=0.5, tau_a=None, tau_d=tau_d))
    assert "plan de fusion VIDE" in str(excinfo.value)


def test_empty_plan_allowed_explicitly(tmp_path):
    """--allow-empty : le checkpoint est alors une copie assumee de la reference."""
    ar_base, ar_post, diff_base, diff_post = ar_and_diffusion(tmp_path)
    other = {"model.layers.9.self_attn.q_proj.weight": (4, 4)}
    other_base = build_checkpoint(tmp_path, other, seed=91, name="other_base")
    other_post = build_checkpoint(tmp_path, other, seed=92, name="other_post")
    tau_d = CheckpointPairSource(other_post, other_base)
    out = tmp_path / "out"
    result = merge_checkpoint(config("ablation-dp-minus-tau-d", diff_post, out, lam=0.5,
                                     tau_a=None, tau_d=tau_d, allow_empty=True,
                                     verify=False))
    assert result.n_fused == 0
    src, dst = tensors_of(diff_post), tensors_of(out)
    for name in QWEN:
        assert torch.equal(src[name], dst[name]), name


# --------------------------------------------------------------------------- #
# MAJ-03 : intersection des lignes d'embedding
# --------------------------------------------------------------------------- #

def test_embedding_row_intersection_is_verifiable_and_mutable(tmp_path):
    """MAJ-03 : l'intersection des lignes est verifiable, et une mutation est detectee."""
    big = dict(QWEN)
    big["model.embed_tokens.weight"] = (10, 4)
    ar_base, ar_post = pair(tmp_path, spec=big, base_name="ar_base", post_name="ar_post")
    recipient = build_checkpoint(tmp_path, QWEN, seed=31, name="diff_base")
    tau_a = CheckpointPairSource(ar_post, ar_base)
    out = tmp_path / "out"
    merge_checkpoint(config("transfer", recipient, out, alpha=1.0, tau_a=tau_a))
    # la verification ne doit PAS lever ShapeMismatchError
    report = verify(out, recipient, mode="transfer", alpha=1.0, tau_a=tau_a)
    assert report["ok"], report["failures"]
    # une mutation sur une ligne de l'intersection doit etre detectee
    mutate(out, "model.embed_tokens.weight", lambda t: t.clone().index_fill_(
        0, torch.tensor([3]), 0.0))
    report = verify(out, recipient, mode="transfer", alpha=1.0, tau_a=tau_a)
    assert not report["ok"]
    assert report["failures"]


# --------------------------------------------------------------------------- #
# Politique de couverture : choix explicite pour une source LoRA
# --------------------------------------------------------------------------- #

def test_lora_source_requires_explicit_coverage(tmp_path):
    """Une source LoRA dans un merge impose un choix explicite de politique."""
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    import torch.nn as nn

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(4, 4, bias=False)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attn()

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block()])

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    torch.manual_seed(0)
    model = Toy()
    peft_model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"]))
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.full_like(param, 0.5))
    adapter_dir = tmp_path / "adapter"
    peft_model.save_pretrained(str(adapter_dir))

    from a2d import LoraAdapterSource
    ar_base, ar_post, diff_base, diff_post = ar_and_diffusion(tmp_path)
    tau_a = CheckpointPairSource(ar_post, ar_base)
    tau_d = LoraAdapterSource(adapter_dir)
    with pytest.raises(MergeError) as excinfo:
        merge_checkpoint(config("merge", diff_post, tmp_path / "out", lam=0.6,
                                tau_a=tau_a, tau_d=tau_d))
    assert "politique de couverture" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# BF16 : tolerance derivee du residu, ecart en ULP
# --------------------------------------------------------------------------- #

def _bf16_perturb(tensor: torch.Tensor, n_ulp: int) -> torch.Tensor:
    """Decale un tenseur BF16 de `n_ulp` pas unitaires (vers +inf)."""
    bits = tensor.view(torch.int16).to(torch.int32) & 0xFFFF
    bits = bits + n_ulp
    out = (bits | (tensor.view(torch.int16).to(torch.int32) & 0xFFFF0000))
    return out.to(torch.int16).view(torch.bfloat16)


def test_bf16_one_ulp_is_tolerated_and_two_ulp_is_rejected(tmp_path):
    """Le seuil d'ULP est discriminant : 1 ULP passe, 2 ULP echoue."""
    spec = {"model.layers.0.self_attn.q_proj.weight": (256, 256),
            "model.layers.0.mlp.gate_proj.weight": (256, 256)}
    ar_base = build_checkpoint(tmp_path, spec, seed=101, name="ar_base")
    ar_post = build_checkpoint(tmp_path, spec, seed=102, name="ar_post")
    diff_base = build_checkpoint(tmp_path, spec, seed=103, name="diff_base")
    diff_post = build_checkpoint(tmp_path, spec, seed=104, name="diff_post")
    tau_a = CheckpointPairSource(ar_post, ar_base)
    tau_d = CheckpointPairSource(diff_post, diff_base)
    out = tmp_path / "out"
    merge_checkpoint(config("merge", diff_post, out, lam=0.6, tau_a=tau_a, tau_d=tau_d,
                            dtype=torch.bfloat16))
    report = verify(out, diff_post, mode="merge", lam=0.6, tau_a=tau_a, tau_d=tau_d,
                    dtype=torch.bfloat16)
    assert report["ok"], report["failures"]
    assert report["max_ulp_deviation"] == pytest.approx(0.0, abs=1e-9)

    name = "model.layers.0.self_attn.q_proj.weight"
    mutate(out, name, lambda t: _bf16_perturb(t, 1))
    report = verify(out, diff_post, mode="merge", lam=0.6, tau_a=tau_a, tau_d=tau_d,
                    dtype=torch.bfloat16)
    assert report["ok"], report["failures"]
    assert report["max_ulp_deviation"] == pytest.approx(1.0, abs=1e-6)

    merge_checkpoint(config("merge", diff_post, out, lam=0.6, tau_a=tau_a, tau_d=tau_d,
                            dtype=torch.bfloat16, overwrite=True))
    mutate(out, name, lambda t: _bf16_perturb(t, 2))
    report = verify(out, diff_post, mode="merge", lam=0.6, tau_a=tau_a, tau_d=tau_d,
                    dtype=torch.bfloat16)
    assert not report["ok"]
    assert any("ULP" in f for f in report["failures"]), report["failures"]


def test_bf16_bound_covers_the_measured_deviation(tmp_path):
    """La borne de tolerance est derivee du residu BF16 mesure, pas d'une constante."""
    spec = {"model.layers.0.self_attn.q_proj.weight": (256, 256)}
    ar_base = build_checkpoint(tmp_path, spec, seed=111, name="ar_base")
    ar_post = build_checkpoint(tmp_path, spec, seed=112, name="ar_post")
    diff_base = build_checkpoint(tmp_path, spec, seed=113, name="diff_base")
    diff_post = build_checkpoint(tmp_path, spec, seed=114, name="diff_post")
    tau_a = CheckpointPairSource(ar_post, ar_base)
    tau_d = CheckpointPairSource(diff_post, diff_base)
    out = tmp_path / "out"
    merge_checkpoint(config("merge", diff_post, out, lam=0.6, tau_a=tau_a, tau_d=tau_d,
                            dtype=torch.bfloat16))
    report = verify(out, diff_post, mode="merge", lam=0.6, tau_a=tau_a, tau_d=tau_d,
                    dtype=torch.bfloat16)
    family = report["families"]["__all__"]
    # l'ecart mesure est inferieur ou egal a tol + borne BF16
    assert family["deviation_a"] <= family["tolerance"] + family["bf16_bound"]
    assert family["deviation_d"] <= family["tolerance"] + family["bf16_bound"]
    # et la borne BF16 est strictement positive (elle est mesuree, pas nulle)
    assert family["bf16_bound"] > 0.0


# --------------------------------------------------------------------------- #
# CLI : couverture declaree au manifeste
# --------------------------------------------------------------------------- #

def test_cli_merge_declares_coverage(tmp_path):
    """Le manifeste declare la politique et les effectifs de couverture."""
    ar_base, ar_post, diff_base, diff_post = ar_and_diffusion(tmp_path)
    cmd = [sys.executable, str(SCRIPTS / "merge_a2d.py"), "--mode", "merge",
           "--diffusion", str(diff_post),
           "--tau-a-post", str(ar_post), "--tau-a-base", str(ar_base),
           "--tau-d-post", str(diff_post), "--tau-d-base", str(diff_base),
           "--lambda", "0.6", "--output", str(tmp_path / "out")]
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    manifest = json.loads((tmp_path / "out" / "merge_config.json").read_text())
    assert manifest["coverage"]["policy"] == "strict"
    assert manifest["coverage"]["n_both"] == len(QWEN)
    assert manifest["coverage"]["n_A_only"] == 0
    assert manifest["coverage"]["n_D_only"] == 0
    assert manifest["post_write_verification"]["ok"]
    assert manifest["post_write_verification"]["version"] == 2
