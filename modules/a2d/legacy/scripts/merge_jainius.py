#!/usr/bin/env python3
"""DEPRECATED — wrapper autour de la bibliotheque A2D (`modules/a2d/a2d/`).

Ce script est conserve pour compatibilite (le checkpoint `checkpoints/Jainius`
et sa docaute ont ete produits par cette interface). Il n'implique plus la
logique de fusion : tout passe par `modules/a2d/a2d/merge.py`.

  * Le calcul de tau se fait TOUJOURS a partir des `.npy` produits par
    `modules/a2d/legacy/scripts/compute_tau_a.py` (chemin legacy). Le pipeline A2D conforme
    (`modules/a2d/cli/merge_a2d.py`) calcule tau a la volee et ne persiste rien.
  * Le plafond alpha dans [0,1] est conserve COTE LEGACY (le papier n'en
    impose pas : cf. Table 6, alpha jusqu'a 1.4). La bibliotheque, elle,
    accepte alpha >= 0 sans plafond.
  * `modules/a2d/legacy/scripts/compute_tau_a.py` est un chemin legacy : il n'est pas supprime.

Usage (identique a l'historique) :
    python modules/a2d/legacy/scripts/merge_jainius.py \
        --diffusiongemma checkpoints/diffusiongemma_it \
        --tau-dir _artefact/tau_a \
        --output checkpoints/Jainius \
        [--alpha 0.3] [--include-norms-embed]
"""

import argparse
import json
import os

import torch

from academiia.a2d import (FAMILY_EMBED, FAMILY_NORMS, GEMMA4_TO_DIFFUSIONGEMMA,
                           LegacyNpySource, MergeConfig, MergeError, MergeRefused,
                           NameMapping, family_of, merge_checkpoint, module_of)
from academiia.a2d.sources import load_weight_map

LEGACY_MAPPING_ID = "gemma4-diffusiongemma"


def legacy_mapping(include_norms_embed: bool) -> NameMapping:
    """Mapping Gemma 4 -> DiffusionGemma du chemin historique.

    Par defaut : embeddings, normalisations, tete de sortie et vision sont
    inchanges (A2D App. A.2) ; le routeur est fusionne (famille declaree).
    `--include-norms-embed` ajoute les normes et les embeddings.
    """
    base = GEMMA4_TO_DIFFUSIONGEMMA
    families = set(base.fusible_families)
    if include_norms_embed:
        families |= {FAMILY_NORMS, FAMILY_EMBED}
    return NameMapping(id=base.id, description=base.description, rename=base.rename,
                       inverse=base.inverse, fusible_families=frozenset(families),
                       router_policy=base.router_policy,
                       embedding_policy=base.embedding_policy)


def count_skipped(recipient: str, mapping: NameMapping) -> int:
    """Nombre de tenseurs du receveur ecartes par la politique norms/embed."""
    skipped = 0
    for tensor_name in load_weight_map(recipient):
        module = module_of(tensor_name)
        if mapping.source(module) is None:
            continue
        if not mapping.is_fusable(family_of(module)):
            skipped += 1
    return skipped


def main():
    parser = argparse.ArgumentParser(description="[DEPRECATED] Merge: Jainius = D_p + alpha * tau_A")
    parser.add_argument("--diffusiongemma", required=True, help="Path to DiffusionGemma IT checkpoint (D_p)")
    parser.add_argument("--tau-dir", default="_artefact/tau_a", help="Directory with tau_A numpy files")
    parser.add_argument("--output", required=True, help="Output directory for Jainius checkpoint")
    parser.add_argument("--alpha", type=float, default=0.3, help="Interpolation weight (default: 0.3, 0.8 deprecie)")
    parser.add_argument("--include-norms-embed", action="store_true",
                        help="INCLUT embeddings/normalisations dans le merge "
                             "(defaut : exclus, conforme A2D App. A.2)")
    parser.add_argument("--allow-missing-tau-manifest", action="store_true",
                        help="Autorise un tau sans tau_manifest.json (provenance non verifiable)")
    parser.add_argument("--on-shape-mismatch", choices=["error", "warn"], default="error",
                        help="Comportement si tau_A et D_p ont des shapes differentes (defaut: error)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Autorise la reutilisation d'un dossier de sortie existant (nettoie les shards obsoletes)")
    args = parser.parse_args()

    if not 0 <= args.alpha <= 1:
        print(f"ERREUR: --alpha={args.alpha} hors [0,1].")
        raise SystemExit(2)

    # Controle du manifest tau (tracabilite : pas de tau perime/incomplet silencieux)
    manifest_path = os.path.join(args.tau_dir, "tau_manifest.json")
    npy = sorted(f for f in os.listdir(args.tau_dir) if f.endswith(".npy")) \
        if os.path.isdir(args.tau_dir) else []
    tau_manifest = None
    if not os.path.exists(manifest_path):
        if not args.allow_missing_tau_manifest:
            print(f"ERREUR: {manifest_path} absent : provenance du tau non verifiable. "
                  f"Recalculer tau avec modules/a2d/legacy/scripts/compute_tau_a.py, ou passer "
                  f"--allow-missing-tau-manifest.")
            raise SystemExit(2)
        print("ATTENTION: tau_manifest.json absent et --allow-missing-tau-manifest "
              "fourni : mapping base sur l'existence des .npy, provenance non verifiee.")
    else:
        with open(manifest_path) as f:
            tau_manifest = json.load(f)
        declared = tau_manifest.get("n_tensors")
        if declared is not None and declared != len(npy):
            print(f"ERREUR: manifest incoherent : n_tensors={declared} mais {len(npy)} "
                  f"fichiers .npy presents dans {args.tau_dir}")
            raise SystemExit(2)
        filt = tau_manifest.get("filter")
        if filt not in (None, "all", "decoder"):
            print(f"ERREUR: manifest : filter='{filt}' invalide (attendu 'all' ou 'decoder')")
            raise SystemExit(2)

    mapping = legacy_mapping(args.include_norms_embed)
    skipped = count_skipped(args.diffusiongemma, mapping)

    source = LegacyNpySource(args.tau_dir)
    cfg = MergeConfig(
        mode="ablation-dp-plus-tau-a",
        recipient=args.diffusiongemma,
        output=args.output,
        alpha=args.alpha,
        lam=0.0,
        mapping=mapping,
        tau_a=source,
        tau_d=None,
        dtype=torch.bfloat16,
        on_shape_mismatch="error" if args.on_shape_mismatch == "error" else "skip",
        overwrite=args.overwrite,
        # Le chemin legacy conserve son comportement observable : alpha=0 y
        # produit une copie de D_p. Le garde-fou « coefficient nul » de la
        # bibliotheque est donc explicitement contourne sur ce seul cas.
        allow_empty_coefficient=(args.alpha == 0.0),
        extra_manifest={
            "method": "Dp + alpha*tau_A (ablation inspiree A2D, pas de tau_D)",
            "deprecated": True,
            "deprecation_note": ("wrapper : la logique de fusion vit dans modules/a2d/a2d/ ; "
                                 "le pipeline conforme est modules/a2d/cli/merge_a2d.py "
                                 "(tau calcule a la volee, jamais persiste)"),
            "include_norms_embed": args.include_norms_embed,
            "n_skipped_norms_embed": skipped,
            "diffusiongemma": os.path.abspath(args.diffusiongemma),
            "tau_dir": os.path.abspath(args.tau_dir),
            "tau_manifest": tau_manifest,
        },
    )

    from academiia.a2d.merge import build_plan
    plan, _counters = build_plan(load_weight_map(args.diffusiongemma), cfg)
    print(f"Mapped tensors (Gemma4 -> DiffusionGemma): {len(plan)}"
          + (" (variante AVEC norms/embed)" if args.include_norms_embed
             else f" ({skipped} norms/embed exclus)"))

    try:
        result = merge_checkpoint(cfg)
    except MergeRefused as exc:
        print(f"ERREUR: {exc}")
        raise SystemExit(2)
    except MergeError as exc:
        print(f"ERREUR: {exc}")
        raise SystemExit(1)

    manifest = result.manifest
    manifest["shape_errors"] = [note for note in manifest.get("notes", [])
                                if note.startswith("SHAPE WARNING")]
    with open(os.path.join(args.output, "merge_config.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nJainius checkpoint saved to: {args.output}")
    print(f"  Total tensors: {manifest['n_tensors']}")
    print(f"  Mapped (merge applique): {manifest['n_mapped']}")
    print(f"  Unchanged (DiffusionGemma-only): {manifest['n_unchanged']}")
    print("  Manifest: merge_config.json")


if __name__ == "__main__":
    main()
