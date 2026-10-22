"""Tests du pipeline Dp + alpha*tau_A (fixtures factices, sans GPU ni gros fichiers).

Chaque test cible une garantie de surete du merge/verify. Lancer :
    pytest modules/a2d/tests -q
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

SCRIPTS = Path(__file__).resolve().parents[1]
CLI = SCRIPTS / "cli"
PY = sys.executable
# Depuis J8, le pipeline legacy (Jainius) vit dans modules/a2d/legacy/scripts/.
LEGACY_SCRIPTS = SCRIPTS / "legacy" / "scripts"

Q_PROJ = "model.decoder.layers.0.self_attn.q_proj.weight"
LN = "model.decoder.layers.0.input_layernorm.weight"
EMBED = "model.decoder.embed_tokens.weight"
VISION = "model.vision_tower.std_bias"

SIDECARS = ["tokenizer.json", "tokenizer_config.json", "generation_config.json",
            "chat_template.jinja", "processor_config.json"]


def run(script, *args):
    path = LEGACY_SCRIPTS / script
    if not path.exists():
        path = CLI / script
    return subprocess.run([PY, str(path), *map(str, args)],
                          capture_output=True, text=True)


def build_dp(root: Path, tensors=None, name="dp") -> Path:
    """Mini-checkpoint DiffusionGemma : 3 tenseurs decoder + 1 vision."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    if tensors is None:
        torch.manual_seed(0)
        tensors = {
            Q_PROJ: torch.randn(16, 16, dtype=torch.bfloat16),
            LN: torch.ones(16, dtype=torch.bfloat16),
            EMBED: torch.randn(8, 8, dtype=torch.bfloat16),
            VISION: torch.zeros(4, dtype=torch.bfloat16),
        }
    save_file(tensors, d / "model-00001-of-00001.safetensors")
    (d / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors}}))
    (d / "config.json").write_text(json.dumps({"architectures": ["DiffusionGemmaForBlockDiffusion"]}))
    for f in SIDECARS:
        (d / f).write_text("{}")
    (d / "LICENSE").write_text("dummy license")
    (d / "README.md").write_text("SOURCE CARD")
    return d


def build_tau(root, name="tau", q_val=0.05, q_shape=(16, 16), manifest_n=3) -> Path:
    """Dossier tau avec 3 .npy (q_proj, layernorm, embed) + manifest."""
    t = root / name
    t.mkdir(parents=True, exist_ok=True)
    np.save(t / "model_language_model_layers_0_self_attn_q_proj_weight.npy",
            np.full(q_shape, q_val, dtype=np.float32))
    np.save(t / "model_language_model_layers_0_input_layernorm_weight.npy",
            np.full((16,), 0.5, dtype=np.float32))
    np.save(t / "model_language_model_embed_tokens_weight.npy",
            np.full((8, 8), 0.5, dtype=np.float32))
    (t / "tau_manifest.json").write_text(json.dumps(
        {"n_tensors": manifest_n, "filter": "decoder"}))
    return t


def merge(dp, tau, out, *extra):
    return run("merge_jainius.py", "--diffusiongemma", dp, "--tau-dir", tau,
               "--output", out, *extra)


def test_merge_refuses_nonempty_without_overwrite(tmp_path):
    dp, tau = build_dp(tmp_path), build_tau(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "model-00001-of-00001.safetensors").write_bytes(b"previous shard")
    r = merge(dp, tau, out)
    assert r.returncode == 2, r.stdout + r.stderr


def test_merge_shape_mismatch_exit1(tmp_path):
    dp = build_dp(tmp_path)
    tau = build_tau(tmp_path, q_shape=(8, 8))  # incompatible avec 16x16
    r = merge(dp, tau, tmp_path / "out")
    assert r.returncode == 1, r.stdout + r.stderr


def test_merge_incoherent_tau_manifest_exit2(tmp_path):
    dp = build_dp(tmp_path)
    tau = build_tau(tmp_path, manifest_n=99)  # 3 .npy sur disque
    r = merge(dp, tau, tmp_path / "out")
    assert r.returncode == 2, r.stdout + r.stderr


def test_merge_missing_tau_manifest_exit2(tmp_path):
    dp = build_dp(tmp_path)
    tau = build_tau(tmp_path)
    (tau / "tau_manifest.json").unlink()
    r = merge(dp, tau, tmp_path / "out")
    assert r.returncode == 2, r.stdout + r.stderr
    # ... mais accepte avec --allow-missing-tau-manifest
    r2 = merge(dp, tau, tmp_path / "out2", "--allow-missing-tau-manifest")
    assert r2.returncode == 0, r2.stdout + r2.stderr


def test_merge_preserves_destination_readme(tmp_path):
    dp, tau = build_dp(tmp_path), build_tau(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "README.md").write_text("CUSTOM CARD")
    (out / "model-00001-of-00001.safetensors").write_bytes(b"previous shard")
    r = merge(dp, tau, out, "--overwrite")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (out / "README.md").read_text() == "CUSTOM CARD"


def test_merge_default_leaves_norms_and_embed_untouched(tmp_path):
    dp, tau = build_dp(tmp_path), build_tau(tmp_path)
    out = tmp_path / "out"
    r = merge(dp, tau, out)
    assert r.returncode == 0, r.stdout + r.stderr
    src = load_file(dp / "model-00001-of-00001.safetensors")
    res = load_file(out / "model-00001-of-00001.safetensors")
    assert torch.equal(src[LN], res[LN]), "layernorm non modifiee attendue"
    assert torch.equal(src[EMBED], res[EMBED]), "embedding non modifie attendu"
    assert not torch.equal(src[Q_PROJ], res[Q_PROJ]), "q_proj doit etre merge"


def test_merge_include_norms_embed_opt_in(tmp_path):
    dp, tau = build_dp(tmp_path), build_tau(tmp_path)
    out = tmp_path / "out"
    r = merge(dp, tau, out, "--include-norms-embed")
    assert r.returncode == 0, r.stdout + r.stderr
    res = load_file(out / "model-00001-of-00001.safetensors")
    src = load_file(dp / "model-00001-of-00001.safetensors")
    assert not torch.equal(src[LN], res[LN]), "--include-norms-embed doit merger la norme"


def test_verify_reference_detects_missing_tensor(tmp_path):
    ref = build_dp(tmp_path, name="ref")
    torch.manual_seed(1)
    partial = {Q_PROJ: torch.randn(16, 16, dtype=torch.bfloat16),
               LN: torch.ones(16, dtype=torch.bfloat16),
               EMBED: torch.randn(8, 8, dtype=torch.bfloat16)}  # VISION manquant
    ckpt = build_dp(tmp_path, tensors=partial, name="ckpt")
    r = run("verify_checkpoint.py", "--checkpoint", ckpt, "--reference", ref)
    assert r.returncode == 1, r.stdout + r.stderr
    assert VISION in r.stdout


def test_verify_reference_ok_on_identical(tmp_path):
    ref = build_dp(tmp_path, name="ref")
    ckpt = build_dp(tmp_path, name="ckpt")
    r = run("verify_checkpoint.py", "--checkpoint", ckpt, "--reference", ref)
    assert r.returncode == 0, r.stdout + r.stderr


def test_wrapper_legacy_egal_bibliotheque_a2d(tmp_path):
    """ADR-003 (règle de deux) : le wrapper legacy applique la MÊME règle.

    Le legacy fusionne `D_p + α·τ` (τ en `.npy`) ; la bibliothèque applique
    `D_b + α·τ_A` (τ recalculé à la volée). Sur la même source et le même
    coefficient, le tenseur projeté doit être **identique** — sinon le
    wrapper a divergé de la règle canonique.
    """
    from academiia.a2d import (CheckpointPairSource, MergeConfig, get_mapping,
                                merge_checkpoint)

    dp = build_dp(tmp_path)                      # D_p (noms decoder)
    tau = build_tau(tmp_path)                    # τ legacy (.npy, noms AR)

    # Même τ côté bibliothèque : A_base = 0, A_post = τ (noms AR Gemma 4).
    ar_names = {
        "model.language_model.layers.0.self_attn.q_proj.weight": (16, 16),
        "model.language_model.layers.0.input_layernorm.weight": (16,),
        "model.language_model.embed_tokens.weight": (8, 8),
    }
    tau_values = {
        "model.language_model.layers.0.self_attn.q_proj.weight": 0.05,
        "model.language_model.layers.0.input_layernorm.weight": 0.5,
        "model.language_model.embed_tokens.weight": 0.5,
    }
    for name, values in (("ar_base", 0.0), ("ar_post", None)):
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        tensors = {}
        for tensor_name, shape in ar_names.items():
            if values is None:
                tensors[tensor_name] = torch.full(shape, tau_values[tensor_name],
                                                  dtype=torch.float32)
            else:
                tensors[tensor_name] = torch.zeros(shape, dtype=torch.float32)
        save_file(tensors, d / "model-00001-of-00001.safetensors")
        (d / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {},
             "weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors}}))
        (d / "config.json").write_text(json.dumps({"architectures": ["Toy"]}))

    # 1. le wrapper legacy (τ persisté, sortie BF16)
    legacy_out = tmp_path / "legacy_out"
    r = merge(dp, tau, legacy_out, "--alpha", "0.3")
    assert r.returncode == 0, r.stdout + r.stderr

    # 2. la bibliothèque (τ recalculé à la volée, même mapping, même coefficient)
    cfg = MergeConfig(mode="transfer", recipient=dp, output=tmp_path / "lib_out",
                      alpha=0.3, lam=0.0,
                      tau_a=CheckpointPairSource(tmp_path / "ar_post",
                                                tmp_path / "ar_base"),
                      tau_d=None, mapping=get_mapping("gemma4-diffusiongemma"),
                      dtype=torch.bfloat16)
    merge_checkpoint(cfg)

    legacy_q = load_file(legacy_out / "model-00001-of-00001.safetensors")[Q_PROJ]
    lib_q = load_file(tmp_path / "lib_out" / "model-00001-of-00001.safetensors")[Q_PROJ]
    assert torch.equal(legacy_q, lib_q), "le wrapper legacy a divergé de la bibliothèque"
