#!/usr/bin/env python3
"""Verification POST HOC d'un checkpoint fusionne A2D.

Relit le checkpoint fusionne, le checkpoint receveur, recalcule les vecteurs
depuis les sources, et compare le resultat au coefficient annonce.

    python modules/a2d/cli/verify_merge.py \
        --merged checkpoints/dream_transfer_a09 \
        --reference checkpoints/dream_7b_base \
        --tau-a-post checkpoints/qwen25_7b_it --tau-a-base checkpoints/qwen25_7b \
        --mode transfer --alpha 0.9 --mapping identity

Rapporte alpha_eff (ou lambda_eff) par famille de tenseurs et echoue si
l'ecart au coefficient demande depasse la tolerance (bornee par le residu
BF16 mesure).

Cet outil aurait detecte le build livre le 2026-10-19 : fusionne a alpha=0.8
alors que son manifeste annoncait alpha=0.3.
"""

import argparse
import json

from academiia.a2d import (MODE_ALIASES, MODES, CheckpointPairSource, LegacyNpySource,
                           LoraAdapterSource, MergeConfig, MergeError,
                           MissingVectorError, compute_effective_coefficients,
                           get_mapping, normalize_mode, verify_scaled_adapter)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verifie un checkpoint fusionne A2D (alpha_eff / lambda_eff par famille)")
    parser.add_argument("--merged", required=True, help="Checkpoint fusionne a verifier")
    parser.add_argument("--reference", required=True,
                        help="Checkpoint receveur du merge (D_b en transfert, D_p en merge)")
    parser.add_argument("--mode", choices=list(MODES) + list(MODE_ALIASES),
                        default="transfer")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.0)
    parser.add_argument("--mapping", default="identity",
                        help="Mapping de noms : identity ou gemma4-diffusiongemma")
    parser.add_argument("--router-policy", choices=["merge", "skip"], default="merge")
    parser.add_argument("--tau-a-post")
    parser.add_argument("--tau-a-base")
    parser.add_argument("--tau-a-adapter")
    parser.add_argument("--tau-a-dir", help="LEGACY : dossier de .npy tau_A (modules/a2d/legacy/scripts/compute_tau_a.py)")
    parser.add_argument("--tau-d-post")
    parser.add_argument("--tau-d-base")
    parser.add_argument("--tau-d-adapter")
    parser.add_argument("--tau-d-dir", help="LEGACY : dossier de .npy tau_D")
    parser.add_argument("--merged-adapter",
                        help="Mode merge-dg-scaled : dossier de l'adaptateur produit "
                             "a verifier contre --tau-d-adapter (V5/V6)")
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--coverage", choices=["strict", "zero-fill"], default=None,
                        help="Politique de couverture des sources (meme semantique que "
                             "merge_a2d.py). Exige si une source est un adaptateur LoRA "
                             "et que le mode combine les deux sources.")
    parser.add_argument("--json", action="store_true", help="Sortie JSON (machine)")
    return parser


def _fmt(value) -> str:
    """Coefficient signe formate (ou 'n/a' si non defini)."""
    return "n/a" if value is None else f"{value:.4f}"


def build_tau_source(post, base, adapter, legacy_dir, label):
    if adapter:
        return LoraAdapterSource(adapter)
    if legacy_dir:
        return LegacyNpySource(legacy_dir)
    if post and base:
        return CheckpointPairSource(post, base)
    return None


def main() -> int:
    args = build_parser().parse_args()
    mode = normalize_mode(args.mode)
    tau_a = build_tau_source(args.tau_a_post, args.tau_a_base, args.tau_a_adapter,
                             args.tau_a_dir, "a")
    tau_d = build_tau_source(args.tau_d_post, args.tau_d_base, args.tau_d_adapter,
                             args.tau_d_dir, "d")
    if tau_a is None and mode in ("transfer", "ablation-dp-plus-tau-a", "merge",
                                  "merge-dg-scaled"):
        print("ERREUR: aucune source tau_A (--tau-a-post/--tau-a-base, --tau-a-adapter "
              "ou --tau-a-dir).")
        return 2
    if tau_d is None and mode in ("merge", "ablation-dp-minus-tau-d"):
        print("ERREUR: aucune source tau_D (--tau-d-post/--tau-d-base, --tau-d-adapter "
              "ou --tau-d-dir), exigee par le mode "
              f"'{mode}'.")
        return 2
    if args.merged_adapter and tau_d is None:
        print("ERREUR: --merged-adapter exige la source d'origine "
              "(--tau-d-adapter) pour comparer les deltas.")
        return 2
    if args.merged_adapter and not args.tau_d_adapter:
        print("ERREUR: --merged-adapter exige --tau-d-adapter (la source LoRA "
              "d'origine), pas une paire de checkpoints.")
        return 2

    cfg = MergeConfig(mode=mode, recipient=args.reference, output=args.merged,
                      alpha=args.alpha, lam=args.lam,
                      mapping=get_mapping(args.mapping, args.router_policy),
                      tau_a=tau_a, tau_d=tau_d, tol_abs=args.tolerance, verify=False,
                      coverage=args.coverage, write_scaled_adapter=False)
    try:
        report = compute_effective_coefficients(args.merged, args.reference, cfg)
    except MissingVectorError as exc:
        print(f"ERREUR: {exc}")
        return 2
    except MergeError as exc:
        print(f"ERREUR: {exc}")
        return 1
    except FileNotFoundError as exc:
        print(f"ERREUR: {exc}")
        return 2

    adapter_report = None
    if args.merged_adapter:
        adapter_report = verify_scaled_adapter(
            args.tau_d_adapter, args.merged_adapter, factor=1.0 - args.lam)
        report["adapter_verification"] = adapter_report
        if not adapter_report["ok"]:
            report["ok"] = False
            report.setdefault("failures", []).extend(
                "adaptateur: " + failure for failure in adapter_report["failures"])

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        coeff = report["coefficient"]
        print(f"Verification de {args.merged}")
        print(f"  mode={mode}  {coeff} demande = {report['requested']}")
        print(f"  tenseurs fusionnes detectes : {report['n_fused']}")
        print(f"  coefficients signes : c_a_eff={_fmt(report.get('c_a_eff'))} "
              f"c_d_eff={_fmt(report.get('c_d_eff'))} "
              f"(demande c_a={report['requested_coefficients']['a']}, "
              f"c_d={report['requested_coefficients']['d']})")
        print(f"  ecart maximal a round_bf16(expected) : "
              f"{report.get('max_ulp_deviation')} ULP "
              f"(limite {report.get('max_ulp_limit')})")
        print(f"  {'famille':12s} {'n':>5s} {'c_a_eff':>10s} {'c_d_eff':>10s} "
              f"{'ecart':>8s}  verdict")
        for family, stats in report["families"].items():
            deviation = stats["deviation"]
            print(f"  {family:12s} {stats['n_tensors']:5d} "
                  f"{_fmt(stats['c_a_eff']):>10s} {_fmt(stats['c_d_eff']):>10s} "
                  f"{('n/a' if deviation is None else f'{deviation:.4f}'):>8s}  "
                  f"{'OK' if stats['ok'] else 'ECHEC'}")
        if adapter_report is not None:
            print(f"  adaptateur : err_L2_max={adapter_report['max_relative_l2_error']:.3e} "
                  f"cos_min={adapter_report['min_cosine']} "
                  f"modules={adapter_report['n_modules_scaled']} "
                  f"-> {'OK' if adapter_report['ok'] else 'ECHEC'}")
        for failure in report["failures"]:
            print(f"  ECHEC: {failure}")
        print(f"  VERDICT: {'OK' if report['ok'] else 'ECHEC'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
