"""Moteur de fusion A2D (transfert, merge, ablations).

Formules (A2D §2.4, PDF p. 6) :

    tau_A = A_p - A_b                          (§2.1)
    tau_D = D_p - D_b                          (§2.1)
    D_Transfer(alpha) = D_b + alpha * tau_A                    (eq. 1)
    D_Merge(lambda)  = D_b + (1-lambda)*tau_D + lambda*tau_A
                     = D_p + lambda * (tau_A - tau_D)           (eq. 2)

Modes de la bibliotheque :

    transfer                D_b + alpha * tau_A
    merge                   D_p + lambda * (tau_A - tau_D)
    ablation-dp-plus-tau-a  D_p + alpha * tau_A        (Table 9 du papier)
    ablation-dp-minus-tau-d D_p - lambda * tau_D        (Table 10 du papier)

Proprietes garanties :

* tau n'est JAMAIS persiste sur disque : il est recalcule a la volee,
  tenseur par tenseur, depuis les sources (3-4 handles `safe_open`), en FP32.
* arithmetique FP32 sequentielle, UN SEUL cast BF16 (A2D App. A.3, PDF p. 20).
* ecriture atomique (fichier temporaire + `os.replace`).
* verification POST-ECRITURE obligatoire : relecture des tenseurs ecrits,
  calcul de alpha_eff / lambda_eff, echec dur au-dela de la tolerance.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .naming import (FAMILY_EMBED, IDENTITY, NameMapping, family_of, module_of)
from .sources import VectorSource, load_weight_map

MODES = ("transfer", "merge", "ablation-dp-plus-tau-a", "ablation-dp-minus-tau-d")

COEFFICIENT_WARNING_THRESHOLD = 1.5


class MergeError(RuntimeError):
    """Erreur metier du merge (exit 1)."""


class MergeRefused(MergeError):
    """Refus d'ecraser / etat incoherent (exit 2)."""


class MissingVectorError(MergeError):
    """Vecteur de tache manquant pour le mode demande."""


class ShapeMismatchError(MergeError):
    """Forme incompatible entre le vecteur et le tenseur receveur."""


@dataclass
class MergeConfig:
    mode: str = "transfer"
    recipient: Path = Path(".")
    output: Path = Path(".")
    alpha: float = 0.0
    lam: float = 0.0
    mapping: NameMapping = IDENTITY
    tau_a: Optional[VectorSource] = None
    tau_d: Optional[VectorSource] = None
    dtype: torch.dtype = torch.bfloat16
    on_shape_mismatch: str = "error"
    overwrite: bool = False
    tol_abs: float = 0.02
    verify: bool = True
    copy_sidecars: bool = True
    extra_manifest: dict = field(default_factory=dict)


@dataclass
class MergeResult:
    manifest: dict
    verification: dict
    n_fused: int = 0
    n_unchanged: int = 0


def mode_coefficients(mode: str, alpha: float, lam: float) -> Tuple[float, float]:
    """Coefficients (c_a, c_d) appliques au receveur : out = W + c_a*tau_A + c_d*tau_D."""
    if mode == "transfer":
        return alpha, 0.0
    if mode == "merge":
        return lam, -lam
    if mode == "ablation-dp-plus-tau-a":
        return alpha, 0.0
    if mode == "ablation-dp-minus-tau-d":
        return 0.0, -lam
    raise ValueError(f"mode inconnu : {mode!r} (attendu : {MODES})")


def effective_coefficient(mode: str, alpha: float, lam: float) -> Tuple[str, float]:
    """(nom du coefficient, valeur) qui doit etre retrouve par la verification."""
    if mode in ("transfer", "ablation-dp-plus-tau-a"):
        return "alpha", alpha
    return "lambda", lam


def _unscaled_norm_squared(stats: dict, mode: str) -> float:
    """Norme au carre du vecteur NON mis a l'echelle associe au coefficient."""
    if mode in ("transfer", "ablation-dp-plus-tau-a"):
        return stats["n_a"]
    if mode == "ablation-dp-minus-tau-d":
        return stats["n_d"]
    return stats["n_a"] + stats["n_d"] - 2.0 * stats["dot"]


def _new_stats() -> dict:
    return {"n_a": 0.0, "n_d": 0.0, "dot": 0.0, "w": 0.0, "resid": 0.0, "exact": 0.0,
            "num": 0.0, "n_tensors": 0, "n_a_tensors": 0, "n_d_tensors": 0}


def _sq_norm(tensor: torch.Tensor) -> float:
    """Norme au carre, accumulatee en float64.

    `torch.norm` en float32 se trompe de ~0,75 % sur un tenseur de 68 M elements
    (mesure) : les normes de famille du manifeste et les bornes de tolerance de
    la verification post-ecriture meritent l'exactitude float64.
    """
    return float(torch.linalg.vector_norm(tensor, dtype=torch.float64).item()) ** 2


def _dot(a: torch.Tensor, b: torch.Tensor) -> float:
    """Produit scalaire, accumule en float64."""
    return float((a * b).sum(dtype=torch.float64).item())


def validate_config(cfg: MergeConfig) -> List[str]:
    """Valide la configuration ; retourne la liste des avertissements."""
    if cfg.mode not in MODES:
        raise ValueError(f"mode inconnu : {cfg.mode!r} (attendu : {MODES})")
    warnings: List[str] = []
    if cfg.alpha < 0 or cfg.lam < 0:
        raise ValueError("alpha et lambda doivent etre >= 0 (le papier n'impose "
                         "pas de plafond ; cf. Table 6 : alpha jusqu'a 1.4)")
    for name, value in (("alpha", cfg.alpha), ("lambda", cfg.lam)):
        if value > COEFFICIENT_WARNING_THRESHOLD:
            warnings.append(
                f"{name}={value} > {COEFFICIENT_WARNING_THRESHOLD} : au-dela de cette "
                f"valeur le merge degrade generalement le modele (cf. A2D Fig. 8)")
    if cfg.mode in ("transfer", "ablation-dp-plus-tau-a", "merge") and cfg.tau_a is None:
        raise MissingVectorError("tau_A absent : ce mode exige un vecteur AR")
    if cfg.mode == "merge" and cfg.tau_d is None:
        raise MissingVectorError(
            "tau_D absent : l'A2D-Merge (D_p + lambda*(tau_A - tau_D)) est "
            "impossible sans le vecteur de post-training diffusion ; utiliser le "
            "mode transfer (D_b + alpha*tau_A)")
    if cfg.mode == "ablation-dp-minus-tau-d" and cfg.tau_d is None:
        raise MissingVectorError("tau_D absent : ce mode exige le vecteur diffusion")
    return warnings


def build_plan(weight_map: Dict[str, str], cfg: MergeConfig) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, int]]:
    """Plan de fusion : {tenseur_cible: (module_source, famille)} + compteurs.

    Deux sens de parcours :
      * source enumerable (paire de checkpoints, adaptateur LoRA) : on part
        des modules de la source, on calcule la cible via le mapping, et un
        nom cible fantome (absent du receveur) est une ERREUR.
      * source non enumerable (`.npy` legacy) : on part des tenseurs du
        receveur et on cherche l'equivalent source via le mapping inverse.
    """
    plan: Dict[str, Tuple[str, str]] = {}
    counters = {"no_counterpart": 0, "not_fusable": 0, "no_tau": 0, "phantom_target": 0}
    recipient_modules = {module_of(t): t for t in weight_map}
    source_modules = cfg.tau_a.modules() if cfg.tau_a is not None else None

    if source_modules:
        for source_module in sorted(source_modules):
            target_module = cfg.mapping.target(source_module)
            if target_module is None:
                counters["no_counterpart"] += 1
                continue
            if target_module not in recipient_modules:
                counters["phantom_target"] += 1
                raise MergeError(
                    f"nom cible fantome : le mapping '{cfg.mapping.id}' transforme "
                    f"'{source_module}' en '{target_module}', absent du checkpoint "
                    f"receveur. Modeles de familles incompatibles ?")
            family = family_of(target_module)
            if not cfg.mapping.is_fusable(family):
                counters["not_fusable"] += 1
                continue
            if cfg.tau_a is None or not cfg.tau_a.has(source_module):
                counters["no_tau"] += 1
                continue
            plan[recipient_modules[target_module]] = (source_module, family)
        return plan, counters

    for tensor_name in weight_map:
        module = module_of(tensor_name)
        source_module = cfg.mapping.source(module)
        if source_module is None:
            counters["no_counterpart"] += 1
            continue
        family = family_of(module)
        if not cfg.mapping.is_fusable(family):
            counters["not_fusable"] += 1
            continue
        if cfg.tau_a is None or not cfg.tau_a.has(source_module):
            counters["no_tau"] += 1
            continue
        plan[tensor_name] = (source_module, family)
    return plan, counters


def _align_delta(delta: torch.Tensor, weight: torch.Tensor, tensor_name: str,
                 family: str, cfg: MergeConfig, notes: List[str]) -> Optional[torch.Tensor]:
    """Aligne un vecteur sur la forme du tenseur receveur (embeddings : lignes).

    Retourne None si la forme est incompatible et que le mode `skip` est
    demande (le tenseur reste alors inchange).
    """
    if delta.shape == weight.shape:
        return delta
    if (family == FAMILY_EMBED and cfg.mapping.embedding_policy == "rows"
            and delta.dim() == 2 and weight.dim() == 2
            and delta.shape[1] == weight.shape[1]):
        rows = min(delta.shape[0], weight.shape[0])
        aligned = torch.zeros_like(weight)
        aligned[:rows] = delta[:rows]
        notes.append(
            f"{tensor_name}: embedding {tuple(delta.shape)} -> {tuple(weight.shape)} ; "
            f"tau applique sur les {rows} premieres lignes, tau=0 sur les "
            f"{weight.shape[0] - rows} lignes restantes")
        return aligned
    message = (f"{tensor_name}: forme tau={tuple(delta.shape)} incompatible avec "
               f"le receveur {tuple(weight.shape)}")
    if cfg.on_shape_mismatch == "error":
        raise ShapeMismatchError(message)
    notes.append(f"SHAPE WARNING: {message} — tenseur laisse inchange")
    return None


def merge_checkpoint(cfg: MergeConfig, log=print) -> MergeResult:
    """Execute la fusion et renvoie le manifeste + la verification."""
    warnings = validate_config(cfg)
    for warning in warnings:
        log(f"ATTENTION: {warning}")

    recipient = Path(cfg.recipient)
    output = Path(cfg.output)
    weight_map = load_weight_map(recipient)

    output.mkdir(parents=True, exist_ok=True)
    existing = [f for f in os.listdir(output) if f.endswith(".safetensors")]
    if existing and not cfg.overwrite:
        raise MergeRefused(
            f"{output} contient deja {len(existing)} shards. Relancer avec "
            f"--overwrite (nettoyage des shards obsoletes) ou vers un dossier "
            f"vide. Pas de merge partiel.")
    for stale in ("model.safetensors.index.json", "merge_config.json"):
        stale_path = output / stale
        if stale_path.exists():
            stale_path.unlink()

    plan, counters = build_plan(weight_map, cfg)
    c_a, c_d = mode_coefficients(cfg.mode, cfg.alpha, cfg.lam)
    coeff_name, coeff_value = effective_coefficient(cfg.mode, cfg.alpha, cfg.lam)
    log(f"Plan de fusion : {len(plan)} tenseurs sur {len(weight_map)} "
        f"(mode={cfg.mode}, {coeff_name}={coeff_value})")

    notes: List[str] = []
    stats: Dict[str, dict] = defaultdict(_new_stats)
    ratios: Dict[str, List[float]] = defaultdict(list)

    by_file: Dict[str, List[str]] = defaultdict(list)
    for tensor_name, fname in weight_map.items():
        by_file[fname].append(tensor_name)

    total_written = 0
    for n_file, (fname, tensor_names) in enumerate(sorted(by_file.items()), 1):
        out_tensors = {}
        with safe_open(str(recipient / fname), framework="pt") as handle:
            for tensor_name in sorted(tensor_names):
                weight = handle.get_tensor(tensor_name).to(torch.float32)
                if tensor_name not in plan:
                    out_tensors[tensor_name] = weight.to(cfg.dtype)
                    continue
                source_module, family = plan[tensor_name]
                module = module_of(tensor_name)
                delta_a = cfg.tau_a.delta(source_module)
                delta_a = _align_delta(delta_a, weight, tensor_name, family, cfg, notes)
                if delta_a is None:
                    # forme incompatible et --on-shape-mismatch=skip : tenseur inchange
                    plan.pop(tensor_name)
                    out_tensors[tensor_name] = weight.to(cfg.dtype)
                    continue
                delta_d = None
                if cfg.tau_d is not None and cfg.tau_d.has(module):
                    delta_d = cfg.tau_d.delta(module)
                    delta_d = _align_delta(delta_d, weight, tensor_name, family, cfg, notes)
                    if delta_d is None:
                        plan.pop(tensor_name)
                        out_tensors[tensor_name] = weight.to(cfg.dtype)
                        continue
                expected = weight + c_a * delta_a
                if delta_d is not None:
                    expected = expected + c_d * delta_d
                casted = expected.to(cfg.dtype)
                # UN SEUL cast : on mesure l'ecart BF16 pour la verification
                residual = _sq_norm(expected - casted.to(torch.float32))
                out_tensors[tensor_name] = casted

                st = stats[family]
                st["n_tensors"] += 1
                st["w"] += _sq_norm(weight)
                st["resid"] += residual
                st["n_a"] += _sq_norm(delta_a)
                st["n_a_tensors"] += 1
                if delta_d is not None:
                    st["n_d"] += _sq_norm(delta_d)
                    st["n_d_tensors"] += 1
                    st["dot"] += _dot(delta_a, delta_d)
                applied = c_a * delta_a
                if delta_d is not None:
                    applied = applied + c_d * delta_d
                w_norm = math.sqrt(_sq_norm(weight))
                if w_norm > 0:
                    ratios[family].append(math.sqrt(_sq_norm(applied)) / w_norm)
        _atomic_save(out_tensors, output / fname)
        total_written += len(tensor_names)
        log(f"    [{fname}] {len(tensor_names)} tenseurs — total {total_written}/{len(weight_map)}")

    # Nettoyage des shards obsoletes
    for entry in os.listdir(output):
        if entry.endswith(".safetensors") and entry not in by_file:
            (output / entry).unlink()
            log(f"  Nettoyage shard obsolete : {entry}")

    total_size = sum((output / fname).stat().st_size for fname in by_file)
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    _atomic_write_json(output / "model.safetensors.index.json", index)

    if cfg.copy_sidecars:
        _copy_sidecars(recipient, output, log)

    manifest = _build_manifest(cfg, plan, counters, stats, ratios, notes, warnings, weight_map)

    verification: Dict[str, object] = {}
    if cfg.verify:
        verification = verify_written(cfg, weight_map, plan, stats, log)
        manifest["post_write_verification"] = verification
        if not verification.get("ok", False):
            _atomic_write_json(output / "merge_config.json", manifest)
            raise MergeError(
                "verification post-ecriture FAILED : "
                + json.dumps(verification.get("failures", []), ensure_ascii=False))

    _atomic_write_json(output / "merge_config.json", manifest)
    return MergeResult(manifest=manifest, verification=verification,
                       n_fused=len(plan), n_unchanged=len(weight_map) - len(plan))


def _atomic_save(tensors: Dict[str, torch.Tensor], path: Path) -> None:
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".tmp-")
    os.close(fd)
    try:
        save_file(tensors, tmp_path)
        os.replace(tmp_path, str(path))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".tmp-")
    os.close(fd)
    try:
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, str(path))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _copy_sidecars(recipient: Path, output: Path, log) -> None:
    import sys
    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from _common import list_source_sidecar_files
    for entry in list_source_sidecar_files(str(recipient)):
        src = recipient / entry
        dst = output / entry
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    readme_src = recipient / "README.md"
    readme_dst = output / "README.md"
    if readme_dst.exists():
        log("  README.md de destination conserve (non ecrase).")
    elif readme_src.exists():
        shutil.copy2(readme_src, readme_dst)
        log("  README.md source copie (destination sans model card).")


def _family_summary(stats: Dict[str, dict], ratios: Dict[str, List[float]]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for family, st in sorted(stats.items()):
        n_a = math.sqrt(st["n_a"])
        n_d = math.sqrt(st["n_d"])
        w = math.sqrt(st["w"])
        cos = None
        if n_a > 0 and n_d > 0:
            cos = st["dot"] / (n_a * n_d)
        med = None
        values = ratios.get(family) or []
        if values:
            med = sorted(values)[len(values) // 2]
        out[family] = {
            "n_tensors": st["n_tensors"],
            "norm_tau_A": n_a,
            "norm_tau_D": n_d,
            "cos_tau_A_tau_D": cos,
            "norm_W": w,
            "ratio_tau_over_W_global": (math.sqrt(max(st["n_a"], st["n_d"])) / w) if w > 0 else None,
            "ratio_tau_over_W_median": med,
        }
    return out


def _build_manifest(cfg, plan, counters, stats, ratios, notes, warnings, weight_map) -> dict:
    import sys
    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from _common import git_commit
    import safetensors
    coeff_name, coeff_value = effective_coefficient(cfg.mode, cfg.alpha, cfg.lam)
    manifest = {
        "method": "A2D (arXiv:2610.08108) — " + {
            "transfer": "A2D-Transfer : D_b + alpha*tau_A (eq. 1)",
            "merge": "A2D-Merge : D_p + lambda*(tau_A - tau_D) (eq. 2)",
            "ablation-dp-plus-tau-a": "ablation D_p + alpha*tau_A (Table 9)",
            "ablation-dp-minus-tau-d": "ablation D_p - lambda*tau_D (Table 10)",
        }[cfg.mode],
        "mode": cfg.mode,
        "alpha": cfg.alpha,
        "lambda": cfg.lam,
        "effective_coefficient": {"name": coeff_name, "value": coeff_value},
        "recipient": str(Path(cfg.recipient).resolve()),
        "output": str(Path(cfg.output).resolve()),
        "mapping": cfg.mapping.describe(),
        "tau_a_source": cfg.tau_a.describe() if cfg.tau_a else None,
        "tau_d_source": cfg.tau_d.describe() if cfg.tau_d else None,
        "n_mapped": len(plan),
        "n_unchanged": len(weight_map) - len(plan),
        "n_tensors": len(weight_map),
        "plan_counters": counters,
        "families": _family_summary(stats, ratios),
        "notes": notes,
        "warnings": warnings,
        "dtype": str(cfg.dtype).replace("torch.", ""),
        "date": datetime.datetime.now().astimezone().isoformat(),
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "safetensors_version": safetensors.__version__,
    }
    manifest.update(cfg.extra_manifest)
    return manifest


def verify_written(cfg: MergeConfig, weight_map, plan, stats, log=print) -> dict:
    """Verification POST-ECRITURE : relit les tenseurs ecrits, calcule alpha_eff.

    L'ecart tolere est borne par le residu BF16 mesure a l'ecriture
    (|alpha_eff - alpha| <= ||eps||/||tau|| + tol_abs, inegalite triangulaire).
    """
    recipient = Path(cfg.recipient)
    output = Path(cfg.output)
    coeff_name, coeff_value = effective_coefficient(cfg.mode, cfg.alpha, cfg.lam)
    c_a, c_d = mode_coefficients(cfg.mode, cfg.alpha, cfg.lam)

    by_file: Dict[str, List[str]] = defaultdict(list)
    for tensor_name, fname in weight_map.items():
        by_file[fname].append(tensor_name)

    num: Dict[str, float] = defaultdict(float)
    for fname, tensor_names in sorted(by_file.items()):
        with safe_open(str(recipient / fname), framework="pt") as ref_handle, \
                safe_open(str(output / fname), framework="pt") as out_handle:
            for tensor_name in tensor_names:
                if tensor_name not in plan:
                    continue
                _, family = plan[tensor_name]
                out_t = out_handle.get_tensor(tensor_name).to(torch.float32)
                ref_t = ref_handle.get_tensor(tensor_name).to(torch.float32)
                num[family] += _sq_norm(out_t - ref_t)

    families: Dict[str, dict] = {}
    failures: List[str] = []
    total_num = 0.0
    total_den = 0.0
    total_resid = 0.0
    for family, st in sorted(stats.items()):
        den2 = _unscaled_norm_squared(st, cfg.mode)
        if den2 <= 0:
            continue
        den = math.sqrt(den2)
        eff = math.sqrt(num[family]) / den
        bound = math.sqrt(st["resid"]) / den
        ok = abs(eff - coeff_value) <= cfg.tol_abs + bound
        families[family] = {
            "effective_coefficient": eff,
            "requested": coeff_value,
            "tolerance": cfg.tol_abs,
            "bf16_bound": bound,
            "deviation": abs(eff - coeff_value),
            "ok": ok,
        }
        if not ok:
            failures.append(
                f"{family}: {coeff_name}_eff={eff:.4f} (demande {coeff_value}) "
                f"ecart {abs(eff - coeff_value):.4f} > tol {cfg.tol_abs} + "
                f"residu BF16 {bound:.4f}")
        total_num += num[family]
        total_den += den2
        total_resid += st["resid"]

    global_eff = None
    if total_den > 0:
        global_eff = math.sqrt(total_num) / math.sqrt(total_den)
        bound = math.sqrt(total_resid) / math.sqrt(total_den)
        ok = abs(global_eff - coeff_value) <= cfg.tol_abs + bound
        families["__all__"] = {
            "effective_coefficient": global_eff,
            "requested": coeff_value,
            "tolerance": cfg.tol_abs,
            "bf16_bound": bound,
            "deviation": abs(global_eff - coeff_value),
            "ok": ok,
        }
        if not ok:
            failures.append(
                f"global: {coeff_name}_eff={global_eff:.4f} (demande {coeff_value}) "
                f"ecart {abs(global_eff - coeff_value):.4f} > tol {cfg.tol_abs} + "
                f"residu BF16 {bound:.4f}")
    log(f"  Verification post-ecriture : {coeff_name}_eff global = "
        f"{global_eff if global_eff is None else round(global_eff, 4)} "
        f"(demande {coeff_value}) — {'OK' if not failures else 'ECHEC'}")
    return {
        "ok": not failures,
        "coefficient": coeff_name,
        "requested": coeff_value,
        "effective_global": global_eff,
        "families": families,
        "failures": failures,
    }


def compute_effective_coefficients(merged_dir, reference_dir, cfg: MergeConfig,
                                    log=print) -> dict:
    """Recalcule alpha_eff / lambda_eff sur un checkpoint deja fusionne.

    Utilise par `scripts/verify_merge.py` : relit le checkpoint fusionne, le
    receveur, et recalcule les vecteurs depuis les sources.
    """
    merged = Path(merged_dir)
    reference = Path(reference_dir)
    weight_map = load_weight_map(merged)
    plan, counters = build_plan(weight_map, cfg)
    coeff_name, coeff_value = effective_coefficient(cfg.mode, cfg.alpha, cfg.lam)

    by_file: Dict[str, List[str]] = defaultdict(list)
    for tensor_name, fname in weight_map.items():
        by_file[fname].append(tensor_name)

    families: Dict[str, dict] = defaultdict(lambda: _new_stats())
    c_a, c_d = mode_coefficients(cfg.mode, cfg.alpha, cfg.lam)
    for fname, tensor_names in sorted(by_file.items()):
        with safe_open(str(reference / fname), framework="pt") as ref_handle, \
                safe_open(str(merged / fname), framework="pt") as out_handle:
            for tensor_name in tensor_names:
                if tensor_name not in plan:
                    continue
                source_module, family = plan[tensor_name]
                module = module_of(tensor_name)
                out_t = out_handle.get_tensor(tensor_name).to(torch.float32)
                ref_t = ref_handle.get_tensor(tensor_name).to(torch.float32)
                delta_a = cfg.tau_a.delta(source_module)
                if delta_a.shape != ref_t.shape:
                    raise ShapeMismatchError(
                        f"{tensor_name}: forme tau={tuple(delta_a.shape)} != "
                        f"receveur {tuple(ref_t.shape)}")
                delta_d = None
                if cfg.tau_d is not None and cfg.tau_d.has(module):
                    delta_d = cfg.tau_d.delta(module)
                st = families[family]
                st["n_tensors"] += 1
                st["num"] += _sq_norm(out_t - ref_t)
                st["n_a"] += _sq_norm(delta_a)
                if delta_d is not None:
                    st["n_d"] += _sq_norm(delta_d)
                    st["dot"] += _dot(delta_a, delta_d)
                expected = ref_t + c_a * delta_a
                if delta_d is not None:
                    expected = expected + c_d * delta_d
                rounded = expected.to(cfg.dtype).to(torch.float32)
                # residu BF16 pur (arrondi) : c'est lui qui borne la tolerance
                st["resid"] += _sq_norm(expected - rounded)
                # deviation de la valeur ecrite par rapport a la valeur attendue
                st["exact"] += _sq_norm(out_t - rounded)

    results: Dict[str, dict] = {}
    failures: List[str] = []
    for family, st in sorted(families.items()):
        den2 = _unscaled_norm_squared(st, cfg.mode)
        if den2 <= 0:
            continue
        den = math.sqrt(den2)
        eff = math.sqrt(st["num"]) / den
        bound = math.sqrt(st["resid"]) / den
        exact = math.sqrt(st.get("exact", 0.0)) / den
        ok = abs(eff - coeff_value) <= cfg.tol_abs + bound
        results[family] = {
            "effective_coefficient": eff,
            "requested": coeff_value,
            "deviation": abs(eff - coeff_value),
            "bf16_bound": bound,
            "exactness_deviation": exact,
            "ok": ok,
            "n_tensors": st["n_tensors"],
        }
        if not ok:
            failures.append(
                f"{family}: {coeff_name}_eff={eff:.4f} (demande {coeff_value}) "
                f"ecart {abs(eff - coeff_value):.4f} > tol {cfg.tol_abs} + "
                f"residu BF16 {bound:.4f}")
    total_num = sum(st["num"] for st in families.values())
    total_den = sum(_unscaled_norm_squared(st, cfg.mode) for st in families.values())
    total_resid = sum(st["resid"] for st in families.values())
    if total_den > 0:
        eff = math.sqrt(total_num) / math.sqrt(total_den)
        bound = math.sqrt(total_resid) / math.sqrt(total_den)
        ok = abs(eff - coeff_value) <= cfg.tol_abs + bound
        results["__all__"] = {
            "effective_coefficient": eff,
            "requested": coeff_value,
            "deviation": abs(eff - coeff_value),
            "bf16_bound": bound,
            "exactness_deviation": math.sqrt(
                sum(st.get("exact", 0.0) for st in families.values())) / math.sqrt(total_den),
            "ok": ok,
            "n_tensors": sum(st["n_tensors"] for st in families.values()),
        }
        if not ok:
            failures.append(
                f"global: {coeff_name}_eff={eff:.4f} (demande {coeff_value}) "
                f"ecart {abs(eff - coeff_value):.4f} > tol {cfg.tol_abs} + "
                f"residu BF16 {bound:.4f}")
    return {
        "ok": not failures,
        "coefficient": coeff_name,
        "requested": coeff_value,
        "families": results,
        "failures": failures,
        "n_fused": len(plan),
        "plan_counters": counters,
    }
