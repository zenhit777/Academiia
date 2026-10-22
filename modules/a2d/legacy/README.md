# `legacy/` — pipeline Jainius (déprécié, conservé pour l'historique)

Ce dossier regroupe le **pipeline legacy** du projet : le build `checkpoints/Jainius`
(`D_p + α·τ_A` appliqué à DiffusionGemma-IT) et les documents d'audit de la période
2026-10-16 → 2026-10-19. Il est **hors du périmètre scientifique** du papier A2D : c'est
une ablation, sans sélection de α sur ancre, et **sans gain mesuré** vs `D_p`
(`modules/a2d/legacy/eval/RESULTS.md`, `infra/vllm/KNOWN_ISSUES.md`).

> **Ne pas réintroduire ces scripts dans `modules/a2d/`.** Le pipeline **conforme** est
> `modules/a2d/cli/merge_a2d.py` + `modules/a2d/a2d/` ; ses modes et garanties sont décrits dans
> `README.md` et `CHECKLIST.md`. Les scripts ci-dessous ne sont conservés que pour
> rejouer le build historique et pour la traçabilité des audits.

## Contenu

| Fichier | Rôle | Statut |
|---|---|---|
| `modules/a2d/legacy/scripts/compute_tau_a.py` | `τ_A = A_p − A_b` persisté en `.npy` (Gemma 4 → DiffusionGemma) | **legacy** — bloqué ENOMEM sur VM 30 GiB |
| `modules/a2d/legacy/scripts/merge_jainius.py` | wrapper déprécié de la bibliothèque (`Jainius = D_p + α·τ_A`, α ∈ [0,1]) | **legacy** |
| `modules/a2d/legacy/scripts/measure_effect.py` | métrologie de l'arrondi BF16 (ratio, cosinus, fidélité de norme) | **legacy** |
| `modules/a2d/legacy/docs/AUDIT.md` | audit interne du 2026-10-16 (faiblesses corrigées/ouvertes) | archive, non normative |
| `modules/a2d/legacy/docs/GUIDE_RESTANT.md` | guide de publication Jainius du 2026-10-19 | archive **obsolète** (bandeau d'origine conservé) |

Les scripts ont été déplacés le **2026-10-21** (J8, phase C), puis le **2026-10-22**
(J9, phase 2) dans `modules/a2d/legacy/scripts/`. Ils importent désormais le
paquet installé (`academiia.a2d`, `academiia.core`) — plus aucun aménagement de
`sys.path` ; leur interface (argumentaire, codes de sortie, manifeste) est
**inchangée**.

## Rejouer le pipeline legacy

```bash
# 1. tau_A (necessite >= 50 Go de RAM : safe_open pre-fault le shard AR de 46,5 GiB)
python modules/a2d/legacy/scripts/compute_tau_a.py \
    --base checkpoints/gemma4_base --posttrained checkpoints/gemma4_it \
    --output-dir _artefact/tau_a

# 2. effet de l'arrondi BF16 (CPU)
python modules/a2d/legacy/scripts/measure_effect.py \
    --diffusiongemma checkpoints/diffusiongemma_it --tau-dir _artefact/tau_a \
    --alphas 0.05,0.1,0.2,0.3,0.5,0.8

# 3. build Jainius
python modules/a2d/legacy/scripts/merge_jainius.py \
    --diffusiongemma checkpoints/diffusiongemma_it --tau-dir _artefact/tau_a \
    --output checkpoints/Jainius --alpha 0.3

# 4. verification d'integrite
python modules/a2d/cli/verify_checkpoint.py --checkpoint checkpoints/Jainius
```

`modules/a2d/cli/verify_checkpoint.py` reste dans `modules/a2d/cli/` : il vérifie l'intégrité HF stricte de
n'importe quel checkpoint (y compris les fusions conformes) et n'est pas spécifique au legacy.

## Ce qui n'est pas ici

* `modules/a2d/eval/eval_compare.py` (évaluation vLLM FP8, 103 prompts) reste dans `modules/a2d/eval/` : il est
  utilisé par `eval/README.md` et `infra/vllm/` pour les contrôles de non-régression, même s'il
  n'est pas conforme au protocole du papier.
* `modules/a2d/legacy/scripts/smoke_int8.py` (test INT8) et `modules/a2d/cli/verify_checkpoint.py` restent dans
  `modules/a2d/` : ce sont des outils d'infrastructure, pas le pipeline Jainius.
* Les audits de mission (`modules/a2d/docs/reports/AUDIT_J6*.md`, `AUDIT_J7.md`) restent dans
  `modules/a2d/docs/reports/` : ils portent sur le pipeline conforme.
