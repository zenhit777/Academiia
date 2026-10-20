"""Tests de la bibliotheque A2D (scripts/a2d) — modeles jouets, sans GPU.

Lancer : pytest scripts/tests -q

Garanties verifiees :
  * transfert   : D_b + alpha*tau_A
  * merge       : D_p + lambda*(tau_A - tau_D) ; lambda=0 => D_p ; lambda=1 => transfert alpha=1
  * tau_D absent => le merge est refuse (degradation en transfert, cf. CLI)
  * coefficient nul refuse (F2) : transfert a alpha=0 et merge a lambda=0 sont
    refuses (ils n'ecriraient qu'une copie de la reference) ; l'opt-in
    --allow-empty-coefficient est trace au manifeste
  * --no-verify trace au manifeste (F3)
  * linearite en alpha
  * source LoRA identique a peft.merge_and_unload
  * alpha_eff / lambda_eff : round-trip et detection d'un alpha erroné
  * refus d'ecraser, embeddings de formes differentes, nom cible fantome
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

from a2d import (CheckpointPairSource, LegacyNpySource, LoraAdapterSource,
                 MergeConfig, MergeError, MergeRefused, MissingVectorError,
                 get_mapping, merge_checkpoint)

QWEN = {
    "model.embed_tokens.weight": (8, 4),
    "model.layers.0.self_attn.q_proj.weight": (4, 4),
    "model.layers.0.self_attn.o_proj.weight": (4, 4),
    "model.layers.0.mlp.gate_proj.weight": (6, 4),
    "model.layers.0.mlp.down_proj.weight": (4, 6),
    "model.layers.0.input_layernorm.weight": (4,),
    "model.norm.weight": (4,),
    "lm_head.weight": (8, 4),
}

DG_AR = {
    "model.language_model.layers.0.self_attn.q_proj.weight": (4, 4),
    "model.language_model.layers.0.mlp.gate_proj.weight": (6, 4),
    "model.language_model.layers.0.router.proj.weight": (4, 4),
    "model.language_model.layers.0.input_layernorm.weight": (4,),
    "model.language_model.embed_tokens.weight": (8, 4),
}

DG_DIFF = {
    "model.decoder.layers.0.self_attn.q_proj.weight": (4, 4),
    "model.decoder.layers.0.mlp.gate_proj.weight": (6, 4),
    "model.decoder.layers.0.router.proj.weight": (4, 4),
    "model.decoder.layers.0.input_layernorm.weight": (4,),
    "model.decoder.embed_tokens.weight": (8, 4),
    "model.decoder.layers.0.layer_scalar": (4,),
    "model.encoder.vision_tower.std_bias": (2,),
}


def build_checkpoint(root: Path, spec: dict, seed: int = 0, name: str = "ckpt") -> Path:
    """Ecrit un checkpoint-jouet (safetensors + index + sidecars)."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    tensors = {k: torch.randn(*shape, generator=generator) for k, shape in spec.items()}
    save_file(tensors, str(d / "model-00001-of-00001.safetensors"))
    (d / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": 0},
         "weight_map": {k: "model-00001-of-00001.safetensors" for k in spec}}))
    (d / "config.json").write_text(json.dumps({"architectures": ["Toy"]}))
    for sidecar in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        (d / sidecar).write_text("{}")
    return d


def tensors_of(path: Path) -> dict:
    return load_file(str(path / "model-00001-of-00001.safetensors"))


def pair(root: Path, spec=QWEN, base_name="base", post_name="post", seed_base=1, seed_post=2):
    base = build_checkpoint(root, spec, seed=seed_base, name=base_name)
    post = build_checkpoint(root, spec, seed=seed_post, name=post_name)
    return base, post


def ar_and_diffusion(root: Path, spec=QWEN):
    """Quatre checkpoints distincts : AR (base/post) et diffusion (base/post).

    Le vecteur AR (tau_A) et le vecteur diffusion (tau_D) doivent etre
    distincts, sinon tau_A - tau_D s'annule dans l'A2D-Merge.
    """
    ar_base = build_checkpoint(root, spec, seed=11, name="ar_base")
    ar_post = build_checkpoint(root, spec, seed=12, name="ar_post")
    diff_base = build_checkpoint(root, spec, seed=21, name="diff_base")
    diff_post = build_checkpoint(root, spec, seed=22, name="diff_post")
    return ar_base, ar_post, diff_base, diff_post


def config(mode, recipient, output, alpha=0.0, lam=0.0, tau_a=None, tau_d=None,
           mapping=None, dtype=torch.float32, **kwargs):
    return MergeConfig(mode=mode, recipient=recipient, output=output, alpha=alpha,
                        lam=lam, tau_a=tau_a, tau_d=tau_d,
                        mapping=mapping or get_mapping("identity"),
                        dtype=dtype, **kwargs)


# --------------------------------------------------------------------------- #
# Transfert
# --------------------------------------------------------------------------- #

def test_transfer_is_base_plus_alpha_tau(tmp_path):
    base, post = pair(tmp_path)
    tau = CheckpointPairSource(post, base)
    out = tmp_path / "out"
    result = merge_checkpoint(config("transfer", base, out, alpha=0.5, tau_a=tau))
    src, dst = tensors_of(base), tensors_of(out)
    ref, diff = tensors_of(post), tensors_of(post)
    assert result.n_fused == len(QWEN)
    for name in QWEN:
        expected = src[name] + 0.5 * (diff[name] - src[name])
        assert torch.allclose(dst[name], expected, atol=1e-6), name


def test_transfer_alpha_zero_is_base(tmp_path):
    base, post = pair(tmp_path)
    tau = CheckpointPairSource(post, base)
    out = tmp_path / "out"
    merge_checkpoint(config("transfer", base, out, alpha=0.0, tau_a=tau,
                            allow_empty_coefficient=True))
    src, dst = tensors_of(base), tensors_of(out)
    for name in QWEN:
        assert torch.equal(src[name], dst[name]), name


def test_alpha_linearity(tmp_path):
    base, post = pair(tmp_path)
    tau = CheckpointPairSource(post, base)
    outs = {}
    for alpha in (0.0, 0.5, 1.0):
        out = tmp_path / f"out{alpha}"
        merge_checkpoint(config("transfer", base, out, alpha=alpha, tau_a=tau,
                                allow_empty_coefficient=True))
        outs[alpha] = tensors_of(out)
    for name in QWEN:
        midpoint = 0.5 * (outs[0.0][name] + outs[1.0][name])
        assert torch.allclose(outs[0.5][name], midpoint, atol=1e-6), name


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #

def test_merge_lambda_zero_is_dp(tmp_path):
    ar_base, ar_post, diff_base, diff_post = ar_and_diffusion(tmp_path)
    tau_a = CheckpointPairSource(ar_post, ar_base)
    tau_d = CheckpointPairSource(diff_post, diff_base)
    out = tmp_path / "out"
    merge_checkpoint(config("merge", diff_post, out, lam=0.0, tau_a=tau_a, tau_d=tau_d,
                            allow_empty_coefficient=True))
    src, dst = tensors_of(diff_post), tensors_of(out)
    for name in QWEN:
        assert torch.equal(src[name], dst[name]), name


def test_merge_lambda_one_equals_transfer_alpha_one(tmp_path):
    ar_base, ar_post, diff_base, diff_post = ar_and_diffusion(tmp_path)
    tau_a = CheckpointPairSource(ar_post, ar_base)
    tau_d = CheckpointPairSource(diff_post, diff_base)
    merged = tmp_path / "merged"
    transfer = tmp_path / "transfer"
    merge_checkpoint(config("merge", diff_post, merged, lam=1.0, tau_a=tau_a, tau_d=tau_d))
    merge_checkpoint(config("transfer", diff_base, transfer, alpha=1.0, tau_a=tau_a))
    m, t = tensors_of(merged), tensors_of(transfer)
    for name in QWEN:
        assert torch.allclose(m[name], t[name], atol=1e-6), name


def test_merge_without_tau_d_is_refused(tmp_path):
    ar_base, ar_post, diff_base, diff_post = ar_and_diffusion(tmp_path)
    tau_a = CheckpointPairSource(ar_post, ar_base)
    with pytest.raises(MissingVectorError):
        merge_checkpoint(config("merge", diff_post, tmp_path / "out", lam=0.6, tau_a=tau_a))


def test_merge_formula_matches_manual_computation(tmp_path):
    ar_base, ar_post, diff_base, diff_post = ar_and_diffusion(tmp_path)
    tau_a = CheckpointPairSource(ar_post, ar_base)
    tau_d = CheckpointPairSource(diff_post, diff_base)
    out = tmp_path / "out"
    merge_checkpoint(config("merge", diff_post, out, lam=0.6, tau_a=tau_a, tau_d=tau_d))
    src, dst = tensors_of(diff_post), tensors_of(out)
    ar_b, ar_p = tensors_of(ar_base), tensors_of(ar_post)
    d_b = tensors_of(diff_base)
    for name in QWEN:
        tau_a_v = ar_p[name] - ar_b[name]
        tau_d_v = src[name] - d_b[name]
        expected = src[name] + 0.6 * (tau_a_v - tau_d_v)
        assert torch.allclose(dst[name], expected, atol=1e-6), name


# --------------------------------------------------------------------------- #
# Garde-fou « coefficient nul » (F2)
# --------------------------------------------------------------------------- #

def test_transfer_with_zero_coefficient_is_refused(tmp_path):
    """F2 : un transfert a alpha=0 n'ecrit qu'une copie de D_b : il est refuse."""
    base, post = pair(tmp_path)
    tau = CheckpointPairSource(post, base)
    out = tmp_path / "out"
    with pytest.raises(MergeRefused) as excinfo:
        merge_checkpoint(config("transfer", base, out, alpha=0.0, tau_a=tau))
    message = str(excinfo.value)
    assert "coefficient nul refuse" in message
    assert "alpha=0.0" in message
    # aucun checkpoint n'est ecrit : le no-op ne laisse rien sur le disque
    assert not list(out.glob("*.safetensors"))


def test_merge_with_zero_lambda_is_refused(tmp_path):
    """F2 : un merge a lambda=0 n'ecrit qu'une copie de D_p : il est refuse."""
    ar_base, ar_post, diff_base, diff_post = ar_and_diffusion(tmp_path)
    tau_a = CheckpointPairSource(ar_post, ar_base)
    tau_d = CheckpointPairSource(diff_post, diff_base)
    out = tmp_path / "out"
    with pytest.raises(MergeRefused) as excinfo:
        merge_checkpoint(config("merge", diff_post, out, lam=0.0, tau_a=tau_a, tau_d=tau_d))
    assert "coefficient nul refuse" in str(excinfo.value)
    assert not list(out.glob("*.safetensors"))


def test_zero_coefficient_allowed_explicitly_is_declared(tmp_path):
    """F2 : l'opt-in explicite est trace au manifeste (avertissement + etat)."""
    base, post = pair(tmp_path)
    tau = CheckpointPairSource(post, base)
    out = tmp_path / "out"
    result = merge_checkpoint(config("transfer", base, out, alpha=0.0, tau_a=tau,
                                     allow_empty_coefficient=True))
    guard = result.manifest["coefficient_guard"]
    assert guard["status"] == "bypassed"
    assert guard["zero_coefficient"] is True
    assert guard["allow_empty_coefficient"] is True
    assert guard["plan_has_nonzero_deltas"] is True
    assert any("--allow-empty-coefficient" in w for w in result.manifest["warnings"])
    on_disk = json.loads((out / "merge_config.json").read_text())
    assert on_disk["coefficient_guard"]["status"] == "bypassed"
    assert any("--allow-empty-coefficient" in w for w in on_disk["warnings"])


def test_nonzero_coefficient_is_not_guarded(tmp_path):
    """Un coefficient non nul n'est pas touche par le garde-fou."""
    base, post = pair(tmp_path)
    tau = CheckpointPairSource(post, base)
    out = tmp_path / "out"
    result = merge_checkpoint(config("transfer", base, out, alpha=0.5, tau_a=tau))
    guard = result.manifest["coefficient_guard"]
    assert guard["status"] == "not_applicable"
    assert guard["zero_coefficient"] is False


# --------------------------------------------------------------------------- #
# Mapping : DiffusionGemma (embeddings/normes/head inchanges, routeur declare)
# --------------------------------------------------------------------------- #

def test_diffusiongemma_mapping_policies(tmp_path):
    ar = build_checkpoint(tmp_path, DG_AR, seed=3, name="ar_base")
    ar_post = build_checkpoint(tmp_path, DG_AR, seed=4, name="ar_post")
    diff = build_checkpoint(tmp_path, DG_DIFF, seed=5, name="dp")
    tau = CheckpointPairSource(ar_post, ar)
    mapping = get_mapping("gemma4-diffusiongemma")
    out = tmp_path / "out"
    result = merge_checkpoint(config("ablation-dp-plus-tau-a", diff, out, alpha=0.5,
                                     tau_a=tau, mapping=mapping))
    src, dst = tensors_of(diff), tensors_of(out)
    assert not torch.equal(src["model.decoder.layers.0.self_attn.q_proj.weight"],
                           dst["model.decoder.layers.0.self_attn.q_proj.weight"])
    assert not torch.equal(src["model.decoder.layers.0.mlp.gate_proj.weight"],
                           dst["model.decoder.layers.0.mlp.gate_proj.weight"])
    # routeur : fusionne par defaut, mais declare
    assert not torch.equal(src["model.decoder.layers.0.router.proj.weight"],
                           dst["model.decoder.layers.0.router.proj.weight"])
    # embeddings, normes, layer_scalar, vision : inchanges (A.2)
    for name in ("model.decoder.embed_tokens.weight",
                 "model.decoder.layers.0.input_layernorm.weight",
                 "model.decoder.layers.0.layer_scalar",
                 "model.encoder.vision_tower.std_bias"):
        assert torch.equal(src[name], dst[name]), name
    assert result.manifest["mapping"]["router_policy"] == "merge"


def test_diffusiongemma_router_policy_skip(tmp_path):
    ar = build_checkpoint(tmp_path, DG_AR, seed=3, name="ar_base")
    ar_post = build_checkpoint(tmp_path, DG_AR, seed=4, name="ar_post")
    diff = build_checkpoint(tmp_path, DG_DIFF, seed=5, name="dp")
    tau = CheckpointPairSource(ar_post, ar)
    mapping = get_mapping("gemma4-diffusiongemma", router_policy="skip")
    out = tmp_path / "out"
    merge_checkpoint(config("ablation-dp-plus-tau-a", diff, out, alpha=0.5,
                            tau_a=tau, mapping=mapping))
    src, dst = tensors_of(diff), tensors_of(out)
    assert torch.equal(src["model.decoder.layers.0.router.proj.weight"],
                       dst["model.decoder.layers.0.router.proj.weight"])
    assert not torch.equal(src["model.decoder.layers.0.self_attn.q_proj.weight"],
                           dst["model.decoder.layers.0.self_attn.q_proj.weight"])


# --------------------------------------------------------------------------- #
# Embeddings de formes differentes
# --------------------------------------------------------------------------- #

def test_embedding_rows_intersection(tmp_path):
    # le vecteur AR a 10 lignes de vocabulaire, le receveur 8 : intersection
    big = dict(QWEN)
    big["model.embed_tokens.weight"] = (10, 4)
    ar_base, ar_post = pair(tmp_path, spec=big, base_name="ar_base", post_name="ar_post")
    recipient = build_checkpoint(tmp_path, QWEN, seed=31, name="diff_base")
    tau = CheckpointPairSource(ar_post, ar_base)
    out = tmp_path / "out"
    result = merge_checkpoint(config("transfer", recipient, out, alpha=1.0, tau_a=tau))
    src, dst = tensors_of(recipient), tensors_of(out)
    ar_src, ar_dst = tensors_of(ar_base), tensors_of(ar_post)
    tau_rows = (ar_dst["model.embed_tokens.weight"] - ar_src["model.embed_tokens.weight"])[:8]
    expected = src["model.embed_tokens.weight"] + tau_rows
    assert torch.allclose(dst["model.embed_tokens.weight"], expected, atol=1e-6)
    assert any("embedding" in note for note in result.manifest["notes"])


# --------------------------------------------------------------------------- #
# Source LoRA == peft.merge_and_unload
# --------------------------------------------------------------------------- #

def _toy_lora_model(out_dim=6, in_dim=4):
    """Modele-jouet dont le chemin de module est 'model.layers.0.self_attn.q_proj'."""
    import torch.nn as nn

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(in_dim, out_dim, bias=False)

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

    return Toy()


def test_lora_source_matches_peft_merge_and_unload(tmp_path):
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    model = _toy_lora_model()
    base_weight = model.model.layers[0].self_attn.q_proj.weight.data.clone()
    peft_model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"]))
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_A" in name:
                param.copy_(torch.arange(param.numel(), dtype=param.dtype).reshape(param.shape) * 0.01)
            elif "lora_B" in name:
                param.copy_(torch.arange(param.numel(), dtype=param.dtype).reshape(param.shape) * 0.02)
    adapter_dir = tmp_path / "adapter"
    peft_model.save_pretrained(str(adapter_dir))
    merged = peft_model.merge_and_unload()
    expected_delta = merged.model.layers[0].self_attn.q_proj.weight.data - base_weight

    source = LoraAdapterSource(adapter_dir)
    module = "model.layers.0.self_attn.q_proj"
    assert source.has(module)
    delta = source.delta(module)
    assert torch.allclose(delta, expected_delta, atol=1e-5)
    assert source.scale == pytest.approx(8 / 4)


def test_lora_source_used_by_merge(tmp_path):
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    model = _toy_lora_model()
    peft_model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"]))
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.full_like(param, 0.5))
    adapter_dir = tmp_path / "adapter"
    peft_model.save_pretrained(str(adapter_dir))

    spec = {"model.layers.0.self_attn.q_proj.weight": (6, 4)}
    base = build_checkpoint(tmp_path, spec, seed=1, name="base")
    out = tmp_path / "out"
    source = LoraAdapterSource(adapter_dir)
    result = merge_checkpoint(config("transfer", base, out, alpha=1.0, tau_a=source))
    assert result.n_fused == 1
    assert result.manifest["tau_a_source"]["kind"] == "lora-adapter"
    # aucun .npy ecrit par la bibliotheque
    assert not list(out.glob("*.npy"))


# --------------------------------------------------------------------------- #
# Verification post-ecriture
# --------------------------------------------------------------------------- #

def test_post_write_verification_reports_alpha_eff(tmp_path):
    base, post = pair(tmp_path)
    tau = CheckpointPairSource(post, base)
    out = tmp_path / "out"
    result = merge_checkpoint(config("transfer", base, out, alpha=0.3, tau_a=tau))
    verification = result.manifest["post_write_verification"]
    assert verification["ok"]
    assert verification["effective_global"] == pytest.approx(0.3, abs=1e-3)


def test_verify_merge_detects_wrong_alpha(tmp_path):
    base, post = pair(tmp_path, base_name="db", post_name="dp")
    tau = CheckpointPairSource(post, base)
    out = tmp_path / "out"
    # fusionne a 0.8, verifie comme si c'etait 0.3
    merge_checkpoint(config("ablation-dp-plus-tau-a", post, out, alpha=0.8, tau_a=tau))
    cfg = config("ablation-dp-plus-tau-a", post, out, alpha=0.3, tau_a=tau, verify=False)
    from a2d import compute_effective_coefficients
    report = compute_effective_coefficients(out, post, cfg)
    assert not report["ok"]
    assert report["families"]["__all__"]["effective_coefficient"] == pytest.approx(0.8, abs=1e-2)


def test_verify_merge_cli_ok_and_ko(tmp_path):
    base, post = pair(tmp_path, base_name="db", post_name="dp")
    out = tmp_path / "out"
    merge_cmd = [sys.executable, str(SCRIPTS / "merge_a2d.py"), "--mode", "ablation-dp-plus-tau-a",
                 "--diffusion", str(post), "--tau-a-post", str(post), "--tau-a-base", str(base),
                 "--alpha", "0.3", "--output", str(out), "--mapping", "identity"]
    done = subprocess.run(merge_cmd, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    verify_cmd = [sys.executable, str(SCRIPTS / "verify_merge.py"), "--merged", str(out),
                  "--reference", str(post), "--mode", "ablation-dp-plus-tau-a",
                  "--tau-a-post", str(post), "--tau-a-base", str(base), "--alpha", "0.3"]
    ok = subprocess.run(verify_cmd, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    bad = subprocess.run(verify_cmd[:-1] + ["0.8"], capture_output=True, text=True)
    assert bad.returncode == 1, bad.stdout + bad.stderr
    assert "ECHEC" in bad.stdout


# --------------------------------------------------------------------------- #
# Surete
# --------------------------------------------------------------------------- #

def test_refuses_to_overwrite(tmp_path):
    base, post = pair(tmp_path)
    tau = CheckpointPairSource(post, base)
    out = tmp_path / "out"
    out.mkdir()
    (out / "model-00001-of-00001.safetensors").write_bytes(b"previous")
    from a2d import MergeRefused
    with pytest.raises(MergeRefused):
        merge_checkpoint(config("transfer", base, out, alpha=0.3, tau_a=tau))


def test_phantom_target_is_an_error(tmp_path):
    # source au nommage Gemma 4 ('model.language_model.layers.0...') appliquee a un
    # receveur au nommage Qwen : le mapping annonce 'model.decoder.layers.0...',
    # absent du receveur => ERREUR (pas de fusion silencieuse).
    ar_base = build_checkpoint(tmp_path, DG_AR, seed=41, name="ar_base")
    ar_post = build_checkpoint(tmp_path, DG_AR, seed=42, name="ar_post")
    recipient = build_checkpoint(tmp_path, QWEN, seed=43, name="qwen_like")
    tau = CheckpointPairSource(ar_post, ar_base)
    with pytest.raises(MergeError):
        merge_checkpoint(config("transfer", recipient, tmp_path / "out", alpha=0.3, tau_a=tau,
                                mapping=get_mapping("gemma4-diffusiongemma")))


def test_manifest_records_provenance_and_norms(tmp_path):
    base, post = pair(tmp_path)
    tau = CheckpointPairSource(post, base)
    out = tmp_path / "out"
    result = merge_checkpoint(config("transfer", base, out, alpha=0.3, tau_a=tau))
    manifest = result.manifest
    assert manifest["alpha"] == 0.3
    assert manifest["mode"] == "transfer"
    assert manifest["tau_a_source"]["index_post_trained"]["weight_map_sha256"]
    assert manifest["git_commit"]
    assert manifest["torch_version"]
    families = manifest["families"]
    assert families["attn"]["norm_tau_A"] > 0
    assert families["attn"]["ratio_tau_over_W_global"] > 0
    assert families["attn"]["ratio_tau_over_W_median"] > 0
    assert families["embed"]["norm_tau_A"] > 0
    on_disk = json.loads((out / "merge_config.json").read_text())
    assert on_disk["alpha"] == 0.3
    assert on_disk["post_write_verification"]["ok"]


def test_cli_merge_without_tau_d_degrades_to_transfer(tmp_path):
    ar_base, ar_post, diff_base, diff_post = ar_and_diffusion(tmp_path)
    out = tmp_path / "out"
    cmd = [sys.executable, str(SCRIPTS / "merge_a2d.py"), "--mode", "merge",
           "--diffusion", str(diff_post), "--diffusion-base", str(diff_base),
           "--tau-a-post", str(ar_post), "--tau-a-base", str(ar_base),
           "--alpha", "0.4", "--output", str(out)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "degradation en A2D-Transfer" in done.stdout
    manifest = json.loads((out / "merge_config.json").read_text())
    assert manifest["mode"] == "transfer"
    assert manifest["alpha"] == 0.4
    src, dst = tensors_of(diff_base), tensors_of(out)
    ar_b, ar_p = tensors_of(ar_base), tensors_of(ar_post)
    for name in QWEN:
        expected = src[name] + 0.4 * (ar_p[name] - ar_b[name])
        assert torch.allclose(dst[name].to(torch.float32), expected, atol=1e-2), name


def test_cli_merge_without_tau_d_and_without_alpha_is_refused(tmp_path):
    ar_base, ar_post, diff_base, diff_post = ar_and_diffusion(tmp_path)
    cmd = [sys.executable, str(SCRIPTS / "merge_a2d.py"), "--mode", "merge",
           "--diffusion", str(diff_post), "--diffusion-base", str(diff_base),
           "--tau-a-post", str(ar_post), "--tau-a-base", str(ar_base),
           "--lambda", "0.6", "--output", str(tmp_path / "out")]
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 2, done.stdout + done.stderr
    assert "exige un --alpha explicite" in done.stdout


def test_cli_warns_above_coefficient_threshold(tmp_path):
    ar_base, ar_post, diff_base, diff_post = ar_and_diffusion(tmp_path)
    cmd = [sys.executable, str(SCRIPTS / "merge_a2d.py"), "--mode", "transfer",
           "--diffusion-base", str(diff_base),
           "--tau-a-post", str(ar_post), "--tau-a-base", str(ar_base),
           "--alpha", "2.0", "--output", str(tmp_path / "out")]
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "ATTENTION" in done.stdout and "1.5" in done.stdout


def test_cli_transfer_without_alpha_is_refused(tmp_path):
    """F2 (CLI) : un transfert sans coefficient explicite est refuse (exit 2)."""
    base, post = pair(tmp_path)
    cmd = [sys.executable, str(SCRIPTS / "merge_a2d.py"), "--mode", "transfer",
           "--diffusion-base", str(base),
           "--tau-a-post", str(post), "--tau-a-base", str(base),
           "--output", str(tmp_path / "out")]
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 2, done.stdout + done.stderr
    assert "coefficient nul refuse" in done.stdout
    assert not list((tmp_path / "out").glob("*.safetensors"))


def test_cli_allow_empty_coefficient_is_accepted_and_declared(tmp_path):
    """F2 (CLI) : --allow-empty-coefficient assume le no-op et le declare."""
    base, post = pair(tmp_path)
    out = tmp_path / "out"
    cmd = [sys.executable, str(SCRIPTS / "merge_a2d.py"), "--mode", "transfer",
           "--diffusion-base", str(base),
           "--tau-a-post", str(post), "--tau-a-base", str(base),
           "--allow-empty-coefficient", "--output", str(out)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "BYPASSED" in done.stdout
    manifest = json.loads((out / "merge_config.json").read_text())
    assert manifest["coefficient_guard"]["status"] == "bypassed"
    assert any("--allow-empty-coefficient" in w for w in manifest["warnings"])


def test_cli_no_verify_is_traced_in_the_manifest(tmp_path):
    """F3 : --no-verify laisse une trace explicite dans le manifeste."""
    base, post = pair(tmp_path)
    out = tmp_path / "out"
    cmd = [sys.executable, str(SCRIPTS / "merge_a2d.py"), "--mode", "transfer",
           "--diffusion-base", str(base),
           "--tau-a-post", str(post), "--tau-a-base", str(base),
           "--alpha", "0.5", "--no-verify", "--output", str(out)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    manifest = json.loads((out / "merge_config.json").read_text())
    assert manifest["post_write_verification"] == {"skipped": True,
                                                    "reason": "--no-verify"}
    assert "SKIPPEE" in done.stdout


def test_legacy_npy_source_still_readable(tmp_path):
    import numpy as np
    tau_dir = tmp_path / "tau"
    tau_dir.mkdir()
    np.save(tau_dir / "model_layers_0_self_attn_q_proj_weight.npy",
            np.full((4, 4), 0.5, dtype=np.float32))
    source = LegacyNpySource(tau_dir)
    assert source.has("model.layers.0.self_attn.q_proj")
    assert float(source.delta("model.layers.0.self_attn.q_proj")[0, 0]) == 0.5
