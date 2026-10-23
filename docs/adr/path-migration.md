# Table de migration des chemins (J9, PHASE 2.0)

Statut : **appliquée** par les commits de déplacement de J9 (`git mv`, contenu
inchangé). Cette table est la référence des anciens→nouveaux chemins ; les
rapports historiques (`modules/a2d/docs/reports/**`) la citent au lieu d'être
réécrits.

## 1. Règles de classement

| Classe | Signification | Cible |
|---|---|---|
| `core` | code **générique** : statistiques (bootstrap, Wilson), empreintes, E/S safetensors/HF | `academiia/core/` |
| `a2d` | méthode A2D : bibliothèque, CLI, harnais, évaluation, docs, legacy | `modules/a2d/` |
| `infra` | service d'inférence vLLM (compose, smoke, diagnostic) | `infra/vllm/` |
| `legacy` | pipeline Jainius déprécié (hors papier) | `modules/a2d/legacy/` |

Règle de dépendance (ADR-001) : un module importe `academiia.core`, jamais un
autre module ; `academiia.core` n'importe aucun module.

## 2. Table ancien → nouveau

| Ancien chemin | Classe | Nouveau chemin |
|---|---|---|
| `docs/A2D_SPEC.md` | a2d | `modules/a2d/docs/A2D_SPEC.md` |
| `docs/AUDIT_A2D_vs_PAPIER.md` | a2d | `modules/a2d/docs/AUDIT_A2D_vs_PAPIER.md` |
| `docs/STATUS.md` | a2d | `modules/a2d/STATUS.md` (un `docs/STATUS.md` court est recree) |
| `docs/VP_ADAPTERS.md` | a2d | `modules/a2d/docs/VP_ADAPTERS.md` |
| `docs/design/EVAL_BUDGET.md` | a2d | `modules/a2d/docs/design/EVAL_BUDGET.md` |
| `docs/design/G1B_NEMOTRON3B.md` | a2d | `modules/a2d/docs/design/G1B_NEMOTRON3B.md` |
| `docs/design/G1_PROTOCOL.md` | a2d | `modules/a2d/docs/design/G1_PROTOCOL.md` |
| `docs/design/G2_adapter_scaling.md` | a2d | `modules/a2d/docs/design/G2_adapter_scaling.md` |
| `docs/reports/AUDIT_J6.md` | a2d | `modules/a2d/docs/reports/AUDIT_J6.md` |
| `docs/reports/AUDIT_J6BIS.md` | a2d | `modules/a2d/docs/reports/AUDIT_J6BIS.md` |
| `docs/reports/AUDIT_J6TER.md` | a2d | `modules/a2d/docs/reports/AUDIT_J6TER.md` |
| `docs/reports/AUDIT_J7.md` | a2d | `modules/a2d/docs/reports/AUDIT_J7.md` |
| `docs/reports/G1_BILAN.md` | a2d | `modules/a2d/docs/reports/G1_BILAN.md` |
| `docs/reports/G1_BILAN_1024.md` | a2d | `modules/a2d/docs/reports/G1_BILAN_1024.md` |
| `docs/reports/J6.md` | a2d | `modules/a2d/docs/reports/J6.md` |
| `docs/reports/J6bis.md` | a2d | `modules/a2d/docs/reports/J6bis.md` |
| `docs/reports/J7.md` | a2d | `modules/a2d/docs/reports/J7.md` |
| `docs/reports/OUT_OF_WORKSPACE.md` | a2d | `modules/a2d/docs/reports/OUT_OF_WORKSPACE.md` |
| `docs/reports/REVIEW_N2_J6.md` | a2d | `modules/a2d/docs/reports/REVIEW_N2_J6.md` |
| `docs/reports/REVIEW_N2_J6QUATER.md` | a2d | `modules/a2d/docs/reports/REVIEW_N2_J6QUATER.md` |
| `eval/README.md` | legacy | `modules/a2d/legacy/eval/README.md` |
| `eval/RESULTS.md` | legacy | `modules/a2d/legacy/eval/RESULTS.md` |
| `eval/dream_a2d/G1_VERDICT.json` | a2d | `modules/a2d/eval/dream_a2d/G1_VERDICT.json` |
| `eval/dream_a2d/ITEM_FINGERPRINTS.json` | a2d | `modules/a2d/eval/dream_a2d/ITEM_FINGERPRINTS.json` |
| `eval/dream_a2d/RESTRUCTURE_MAP.json` | a2d | `modules/a2d/eval/dream_a2d/RESTRUCTURE_MAP.json` |
| `eval/dream_a2d/SUMMARY.json` | a2d | `modules/a2d/eval/dream_a2d/SUMMARY.json` |
| `eval/dream_a2d/SUMMARY.md` | a2d | `modules/a2d/eval/dream_a2d/SUMMARY.md` |
| `eval/dream_a2d/TRUNCATION_512.json` | a2d | `modules/a2d/eval/dream_a2d/TRUNCATION_512.json` |
| `eval/dream_a2d/dream_7b_base/gsm8k_cot_a2d/results_2026-10-20T00-53-03.608851.json` | a2d | `modules/a2d/eval/dream_a2d/dream_7b_base/gsm8k_cot_a2d/results_2026-10-20T00-53-03.608851.json` |
| `eval/dream_a2d/dream_7b_base/gsm8k_cot_a2d/samples_gsm8k_cot_a2d_2026-10-20T00-53-03.608851.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_7b_base/gsm8k_cot_a2d/samples_gsm8k_cot_a2d_2026-10-20T00-53-03.608851.jsonl` |
| `eval/dream_a2d/dream_7b_base/ifeval_a2d/results_2026-10-20T06-26-54.453067.json` | a2d | `modules/a2d/eval/dream_a2d/dream_7b_base/ifeval_a2d/results_2026-10-20T06-26-54.453067.json` |
| `eval/dream_a2d/dream_7b_base/ifeval_a2d/samples_ifeval_a2d_2026-10-20T06-26-54.453067.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_7b_base/ifeval_a2d/samples_ifeval_a2d_2026-10-20T06-26-54.453067.jsonl` |
| `eval/dream_a2d/dream_7b_base/minerva_math_algebra/results_2026-10-20T03-25-11.907605.json` | a2d | `modules/a2d/eval/dream_a2d/dream_7b_base/minerva_math_algebra/results_2026-10-20T03-25-11.907605.json` |
| `eval/dream_a2d/dream_7b_base/minerva_math_algebra/samples_minerva_math_algebra_2026-10-20T03-25-11.907605.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_7b_base/minerva_math_algebra/samples_minerva_math_algebra_2026-10-20T03-25-11.907605.jsonl` |
| `eval/dream_a2d/dream_7b_base/minerva_math_algebra/samples_minerva_math_counting_and_prob_2026-10-20T03-25-11.907605.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_7b_base/minerva_math_algebra/samples_minerva_math_counting_and_prob_2026-10-20T03-25-11.907605.jsonl` |
| `eval/dream_a2d/dream_7b_base/minerva_math_algebra/samples_minerva_math_geometry_2026-10-20T03-25-11.907605.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_7b_base/minerva_math_algebra/samples_minerva_math_geometry_2026-10-20T03-25-11.907605.jsonl` |
| `eval/dream_a2d/dream_7b_base/minerva_math_algebra/samples_minerva_math_intermediate_algebra_2026-10-20T03-25-11.907605.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_7b_base/minerva_math_algebra/samples_minerva_math_intermediate_algebra_2026-10-20T03-25-11.907605.jsonl` |
| `eval/dream_a2d/dream_7b_base/minerva_math_algebra/samples_minerva_math_num_theory_2026-10-20T03-25-11.907605.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_7b_base/minerva_math_algebra/samples_minerva_math_num_theory_2026-10-20T03-25-11.907605.jsonl` |
| `eval/dream_a2d/dream_7b_base/minerva_math_algebra/samples_minerva_math_prealgebra_2026-10-20T03-25-11.907605.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_7b_base/minerva_math_algebra/samples_minerva_math_prealgebra_2026-10-20T03-25-11.907605.jsonl` |
| `eval/dream_a2d/dream_7b_base/minerva_math_algebra/samples_minerva_math_precalc_2026-10-20T03-25-11.907605.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_7b_base/minerva_math_algebra/samples_minerva_math_precalc_2026-10-20T03-25-11.907605.jsonl` |
| `eval/dream_a2d/dream_7b_instruct/gsm8k_cot_a2d/results_2026-10-20T01-44-38.332314.json` | a2d | `modules/a2d/eval/dream_a2d/dream_7b_instruct/gsm8k_cot_a2d/results_2026-10-20T01-44-38.332314.json` |
| `eval/dream_a2d/dream_7b_instruct/gsm8k_cot_a2d/samples_gsm8k_cot_a2d_2026-10-20T01-44-38.332314.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_7b_instruct/gsm8k_cot_a2d/samples_gsm8k_cot_a2d_2026-10-20T01-44-38.332314.jsonl` |
| `eval/dream_a2d/dream_7b_instruct/ifeval_a2d/results_2026-10-20T07-10-39.426121.json` | a2d | `modules/a2d/eval/dream_a2d/dream_7b_instruct/ifeval_a2d/results_2026-10-20T07-10-39.426121.json` |
| `eval/dream_a2d/dream_7b_instruct/ifeval_a2d/samples_ifeval_a2d_2026-10-20T07-10-39.426121.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_7b_instruct/ifeval_a2d/samples_ifeval_a2d_2026-10-20T07-10-39.426121.jsonl` |
| `eval/dream_a2d/dream_7b_instruct/minerva_math_algebra/results_2026-10-20T05-05-04.209520.json` | a2d | `modules/a2d/eval/dream_a2d/dream_7b_instruct/minerva_math_algebra/results_2026-10-20T05-05-04.209520.json` |
| `eval/dream_a2d/dream_7b_instruct/minerva_math_algebra/samples_minerva_math_algebra_2026-10-20T05-05-04.209520.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_7b_instruct/minerva_math_algebra/samples_minerva_math_algebra_2026-10-20T05-05-04.209520.jsonl` |
| `eval/dream_a2d/dream_7b_instruct/minerva_math_algebra/samples_minerva_math_counting_and_prob_2026-10-20T05-05-04.209520.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_7b_instruct/minerva_math_algebra/samples_minerva_math_counting_and_prob_2026-10-20T05-05-04.209520.jsonl` |
| `eval/dream_a2d/dream_7b_instruct/minerva_math_algebra/samples_minerva_math_geometry_2026-10-20T05-05-04.209520.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_7b_instruct/minerva_math_algebra/samples_minerva_math_geometry_2026-10-20T05-05-04.209520.jsonl` |
| `eval/dream_a2d/dream_7b_instruct/minerva_math_algebra/samples_minerva_math_intermediate_algebra_2026-10-20T05-05-04.209520.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_7b_instruct/minerva_math_algebra/samples_minerva_math_intermediate_algebra_2026-10-20T05-05-04.209520.jsonl` |
| `eval/dream_a2d/dream_7b_instruct/minerva_math_algebra/samples_minerva_math_num_theory_2026-10-20T05-05-04.209520.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_7b_instruct/minerva_math_algebra/samples_minerva_math_num_theory_2026-10-20T05-05-04.209520.jsonl` |
| `eval/dream_a2d/dream_7b_instruct/minerva_math_algebra/samples_minerva_math_prealgebra_2026-10-20T05-05-04.209520.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_7b_instruct/minerva_math_algebra/samples_minerva_math_prealgebra_2026-10-20T05-05-04.209520.jsonl` |
| `eval/dream_a2d/dream_7b_instruct/minerva_math_algebra/samples_minerva_math_precalc_2026-10-20T05-05-04.209520.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_7b_instruct/minerva_math_algebra/samples_minerva_math_precalc_2026-10-20T05-05-04.209520.jsonl` |
| `eval/dream_a2d/dream_merge_l06/gsm8k_cot_a2d/results_2026-10-21T07-48-06.462751.json` | a2d | `modules/a2d/eval/dream_a2d/dream_merge_l06/gsm8k_cot_a2d/results_2026-10-21T07-48-06.462751.json` |
| `eval/dream_a2d/dream_merge_l06/gsm8k_cot_a2d/samples_gsm8k_cot_a2d_2026-10-21T07-48-06.462751.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_merge_l06/gsm8k_cot_a2d/samples_gsm8k_cot_a2d_2026-10-21T07-48-06.462751.jsonl` |
| `eval/dream_a2d/dream_merge_l06/ifeval_a2d/results_2026-10-21T10-14-30.044694.json` | a2d | `modules/a2d/eval/dream_a2d/dream_merge_l06/ifeval_a2d/results_2026-10-21T10-14-30.044694.json` |
| `eval/dream_a2d/dream_merge_l06/ifeval_a2d/samples_ifeval_a2d_2026-10-21T10-14-30.044694.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_merge_l06/ifeval_a2d/samples_ifeval_a2d_2026-10-21T10-14-30.044694.jsonl` |
| `eval/dream_a2d/dream_merge_l06/minerva_math_algebra/results_2026-10-21T09-28-59.487500.json` | a2d | `modules/a2d/eval/dream_a2d/dream_merge_l06/minerva_math_algebra/results_2026-10-21T09-28-59.487500.json` |
| `eval/dream_a2d/dream_merge_l06/minerva_math_algebra/samples_minerva_math_algebra_2026-10-21T09-28-59.487500.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_merge_l06/minerva_math_algebra/samples_minerva_math_algebra_2026-10-21T09-28-59.487500.jsonl` |
| `eval/dream_a2d/dream_merge_l06/minerva_math_algebra/samples_minerva_math_counting_and_prob_2026-10-21T09-28-59.487500.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_merge_l06/minerva_math_algebra/samples_minerva_math_counting_and_prob_2026-10-21T09-28-59.487500.jsonl` |
| `eval/dream_a2d/dream_merge_l06/minerva_math_algebra/samples_minerva_math_geometry_2026-10-21T09-28-59.487500.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_merge_l06/minerva_math_algebra/samples_minerva_math_geometry_2026-10-21T09-28-59.487500.jsonl` |
| `eval/dream_a2d/dream_merge_l06/minerva_math_algebra/samples_minerva_math_intermediate_algebra_2026-10-21T09-28-59.487500.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_merge_l06/minerva_math_algebra/samples_minerva_math_intermediate_algebra_2026-10-21T09-28-59.487500.jsonl` |
| `eval/dream_a2d/dream_merge_l06/minerva_math_algebra/samples_minerva_math_num_theory_2026-10-21T09-28-59.487500.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_merge_l06/minerva_math_algebra/samples_minerva_math_num_theory_2026-10-21T09-28-59.487500.jsonl` |
| `eval/dream_a2d/dream_merge_l06/minerva_math_algebra/samples_minerva_math_prealgebra_2026-10-21T09-28-59.487500.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_merge_l06/minerva_math_algebra/samples_minerva_math_prealgebra_2026-10-21T09-28-59.487500.jsonl` |
| `eval/dream_a2d/dream_merge_l06/minerva_math_algebra/samples_minerva_math_precalc_2026-10-21T09-28-59.487500.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_merge_l06/minerva_math_algebra/samples_minerva_math_precalc_2026-10-21T09-28-59.487500.jsonl` |
| `eval/dream_a2d/dream_transfer_a09/gsm8k_cot_a2d/results_2026-10-21T04-25-17.733232.json` | a2d | `modules/a2d/eval/dream_a2d/dream_transfer_a09/gsm8k_cot_a2d/results_2026-10-21T04-25-17.733232.json` |
| `eval/dream_a2d/dream_transfer_a09/gsm8k_cot_a2d/samples_gsm8k_cot_a2d_2026-10-21T04-25-17.733232.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_transfer_a09/gsm8k_cot_a2d/samples_gsm8k_cot_a2d_2026-10-21T04-25-17.733232.jsonl` |
| `eval/dream_a2d/dream_transfer_a09/ifeval_a2d/results_2026-10-21T06-56-45.324729.json` | a2d | `modules/a2d/eval/dream_a2d/dream_transfer_a09/ifeval_a2d/results_2026-10-21T06-56-45.324729.json` |
| `eval/dream_a2d/dream_transfer_a09/ifeval_a2d/samples_ifeval_a2d_2026-10-21T06-56-45.324729.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_transfer_a09/ifeval_a2d/samples_ifeval_a2d_2026-10-21T06-56-45.324729.jsonl` |
| `eval/dream_a2d/dream_transfer_a09/minerva_math_algebra/results_2026-10-21T06-08-25.127810.json` | a2d | `modules/a2d/eval/dream_a2d/dream_transfer_a09/minerva_math_algebra/results_2026-10-21T06-08-25.127810.json` |
| `eval/dream_a2d/dream_transfer_a09/minerva_math_algebra/samples_minerva_math_algebra_2026-10-21T06-08-25.127810.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_transfer_a09/minerva_math_algebra/samples_minerva_math_algebra_2026-10-21T06-08-25.127810.jsonl` |
| `eval/dream_a2d/dream_transfer_a09/minerva_math_algebra/samples_minerva_math_counting_and_prob_2026-10-21T06-08-25.127810.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_transfer_a09/minerva_math_algebra/samples_minerva_math_counting_and_prob_2026-10-21T06-08-25.127810.jsonl` |
| `eval/dream_a2d/dream_transfer_a09/minerva_math_algebra/samples_minerva_math_geometry_2026-10-21T06-08-25.127810.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_transfer_a09/minerva_math_algebra/samples_minerva_math_geometry_2026-10-21T06-08-25.127810.jsonl` |
| `eval/dream_a2d/dream_transfer_a09/minerva_math_algebra/samples_minerva_math_intermediate_algebra_2026-10-21T06-08-25.127810.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_transfer_a09/minerva_math_algebra/samples_minerva_math_intermediate_algebra_2026-10-21T06-08-25.127810.jsonl` |
| `eval/dream_a2d/dream_transfer_a09/minerva_math_algebra/samples_minerva_math_num_theory_2026-10-21T06-08-25.127810.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_transfer_a09/minerva_math_algebra/samples_minerva_math_num_theory_2026-10-21T06-08-25.127810.jsonl` |
| `eval/dream_a2d/dream_transfer_a09/minerva_math_algebra/samples_minerva_math_prealgebra_2026-10-21T06-08-25.127810.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_transfer_a09/minerva_math_algebra/samples_minerva_math_prealgebra_2026-10-21T06-08-25.127810.jsonl` |
| `eval/dream_a2d/dream_transfer_a09/minerva_math_algebra/samples_minerva_math_precalc_2026-10-21T06-08-25.127810.jsonl` | a2d | `modules/a2d/eval/dream_a2d/dream_transfer_a09/minerva_math_algebra/samples_minerva_math_precalc_2026-10-21T06-08-25.127810.jsonl` |
| `eval/dream_a2d_seed1/ITEM_FINGERPRINTS.json` | a2d | `modules/a2d/eval/dream_a2d_seed1/ITEM_FINGERPRINTS.json` |
| `eval/dream_a2d_seed1/dream_7b_instruct/gsm8k_cot_a2d/results_2026-10-21T02-12-04.380082.json` | a2d | `modules/a2d/eval/dream_a2d_seed1/dream_7b_instruct/gsm8k_cot_a2d/results_2026-10-21T02-12-04.380082.json` |
| `eval/dream_a2d_seed1/dream_7b_instruct/gsm8k_cot_a2d/samples_gsm8k_cot_a2d_2026-10-21T02-12-04.380082.jsonl` | a2d | `modules/a2d/eval/dream_a2d_seed1/dream_7b_instruct/gsm8k_cot_a2d/samples_gsm8k_cot_a2d_2026-10-21T02-12-04.380082.jsonl` |
| `eval/dream_a2d_trunc/ATTEMPT_1024.json` | a2d | `modules/a2d/eval/dream_a2d_trunc/ATTEMPT_1024.json` |
| `eval/dream_a2d_trunc/ATTEMPT_1024.log` | a2d | `modules/a2d/eval/dream_a2d_trunc/ATTEMPT_1024.log` |
| `eval/prompts.jsonl` | legacy | `modules/a2d/legacy/eval/prompts.jsonl` |
| `eval/results_Dp.json` | legacy | `modules/a2d/legacy/eval/results_Dp.json` |
| `eval/results_jainius_a03.json` | legacy | `modules/a2d/legacy/eval/results_jainius_a03.json` |
| `eval/smoke_diffusiongemma_it_20261019_170756.json` | legacy | `modules/a2d/legacy/eval/smoke_diffusiongemma_it_20261019_170756.json` |
| `eval/smoke_jainius_20261019_164553.json` | legacy | `modules/a2d/legacy/eval/smoke_jainius_20261019_164553.json` |
| `eval/smoke_jainius_20261019_170010.json` | legacy | `modules/a2d/legacy/eval/smoke_jainius_20261019_170010.json` |
| `eval/smoke_jainius_20261019_182115.json` | legacy | `modules/a2d/legacy/eval/smoke_jainius_20261019_182115.json` |
| `eval/smoke_jainius_a03_20261019_175036.json` | legacy | `modules/a2d/legacy/eval/smoke_jainius_a03_20261019_175036.json` |
| `eval/smoke_jainius_a03_20261019_175057.json` | legacy | `modules/a2d/legacy/eval/smoke_jainius_a03_20261019_175057.json` |
| `eval/smoke_jainius_a03_20261019_175059.json` | legacy | `modules/a2d/legacy/eval/smoke_jainius_a03_20261019_175059.json` |
| `eval/smoke_jainius_a03_20261019_175101.json` | legacy | `modules/a2d/legacy/eval/smoke_jainius_a03_20261019_175101.json` |
| `eval/smoke_jainius_a03_ne_20261019_181207.json` | legacy | `modules/a2d/legacy/eval/smoke_jainius_a03_ne_20261019_181207.json` |
| `eval/smoke_jainius_a03_ne_20261019_181209.json` | legacy | `modules/a2d/legacy/eval/smoke_jainius_a03_ne_20261019_181209.json` |
| `eval/smoke_jainius_a03_ne_20261019_181212.json` | legacy | `modules/a2d/legacy/eval/smoke_jainius_a03_ne_20261019_181212.json` |
| `harness/README.md` | a2d | `modules/a2d/harness/README.md` |
| `harness/dream/README.md` | a2d | `modules/a2d/harness/dream/README.md` |
| `harness/dream/requirements-dream.txt` | a2d | `modules/a2d/harness/dream/requirements-dream.txt` |
| `harness/dream/run_evals.sh` | a2d | `modules/a2d/harness/dream/run_evals.sh` |
| `harness/dream/setup.sh` | a2d | `modules/a2d/harness/dream/setup.sh` |
| `harness/dream/summarize.py` | a2d | `modules/a2d/harness/dream/summarize.py` |
| `harness/dream/tasks/gsm8k_cot_a2d.yaml` | a2d | `modules/a2d/harness/dream/tasks/gsm8k_cot_a2d.yaml` |
| `harness/dream/tasks/ifeval_a2d.yaml` | a2d | `modules/a2d/harness/dream/tasks/ifeval_a2d.yaml` |
| `harness/dream/tasks/instructions.py` | a2d | `modules/a2d/harness/dream/tasks/instructions.py` |
| `harness/dream/tasks/instructions_registry.py` | a2d | `modules/a2d/harness/dream/tasks/instructions_registry.py` |
| `harness/dream/tasks/instructions_util.py` | a2d | `modules/a2d/harness/dream/tasks/instructions_util.py` |
| `harness/dream/tasks/math_trunc_a2d.jsonl` | a2d | `modules/a2d/harness/dream/tasks/math_trunc_a2d.jsonl` |
| `harness/dream/tasks/math_trunc_a2d.jsonl.manifest.json` | a2d | `modules/a2d/harness/dream/tasks/math_trunc_a2d.jsonl.manifest.json` |
| `harness/dream/tasks/math_trunc_a2d.yaml` | a2d | `modules/a2d/harness/dream/tasks/math_trunc_a2d.yaml` |
| `harness/dream/tasks/utils.py` | a2d | `modules/a2d/harness/dream/tasks/utils.py` |
| `harness/dream/tasks/utils_math_trunc.py` | a2d | `modules/a2d/harness/dream/tasks/utils_math_trunc.py` |
| `legacy/README.md` | legacy | `modules/a2d/legacy/README.md` |
| `legacy/docs/AUDIT.md` | legacy | `modules/a2d/legacy/docs/AUDIT.md` |
| `legacy/docs/GUIDE_RESTANT.md` | legacy | `modules/a2d/legacy/docs/GUIDE_RESTANT.md` |
| `legacy/scripts/compute_tau_a.py` | legacy | `modules/a2d/legacy/scripts/compute_tau_a.py` |
| `legacy/scripts/measure_effect.py` | legacy | `modules/a2d/legacy/scripts/measure_effect.py` |
| `legacy/scripts/merge_jainius.py` | legacy | `modules/a2d/legacy/scripts/merge_jainius.py` |
| `scripts/_common.py` | core + a2d | `academiia/core/common.py` + `modules/a2d/a2d/mapping.py` |
| `scripts/a2d/__init__.py` | a2d | `modules/a2d/a2d/__init__.py` |
| `scripts/a2d/adapters.py` | a2d | `modules/a2d/a2d/adapters.py` |
| `scripts/a2d/merge.py` | a2d | `modules/a2d/a2d/merge.py` |
| `scripts/a2d/naming.py` | a2d | `modules/a2d/a2d/naming.py` |
| `scripts/a2d/sources.py` | a2d | `modules/a2d/a2d/sources.py` |
| `scripts/download_models.py` | a2d | `modules/a2d/cli/download_models.py` |
| `scripts/eval_compare.py` | a2d | `modules/a2d/eval/eval_compare.py` |
| `scripts/eval_item_fingerprint.py` | a2d | `modules/a2d/eval/eval_item_fingerprint.py` |
| `scripts/g1_verdict.py` | a2d | `modules/a2d/eval/g1_verdict.py` |
| `scripts/make_math_trunc_subset.py` | a2d | `modules/a2d/eval/make_math_trunc_subset.py` |
| `scripts/measure_geometry.py` | a2d | `modules/a2d/cli/measure_geometry.py` |
| `scripts/measure_math_truncation.py` | a2d | `modules/a2d/eval/measure_math_truncation.py` |
| `scripts/merge_a2d.py` | a2d | `modules/a2d/cli/merge_a2d.py` |
| `scripts/paired_bootstrap.py` | a2d (+ stats dans core) | `modules/a2d/eval/paired_bootstrap.py` |
| `scripts/restructure_eval_dirs.py` | a2d | `modules/a2d/eval/restructure_eval_dirs.py` |
| `scripts/seed_noise_floor.py` | a2d | `modules/a2d/eval/seed_noise_floor.py` |
| `scripts/smoke_int8.py` | legacy | `modules/a2d/legacy/scripts/smoke_int8.py` |
| `scripts/tests/test_a2d_library.py` | a2d | `modules/a2d/tests/test_a2d_library.py` |
| `scripts/tests/test_a2d_mutation.py` | a2d | `modules/a2d/tests/test_a2d_mutation.py` |
| `scripts/tests/test_eval_compare.py` | a2d | `modules/a2d/tests/test_eval_compare.py` |
| `scripts/tests/test_g1_verdict.py` | a2d | `modules/a2d/tests/test_g1_verdict.py` |
| `scripts/tests/test_pipeline.py` | a2d | `modules/a2d/tests/test_pipeline.py` |
| `scripts/verify_checkpoint.py` | a2d | `modules/a2d/cli/verify_checkpoint.py` |
| `scripts/verify_merge.py` | a2d | `modules/a2d/cli/verify_merge.py` |
| `vLLM/.env.template` | infra | `infra/vllm/.env.template` |
| `vLLM/KNOWN_ISSUES.md` | infra | `infra/vllm/KNOWN_ISSUES.md` |
| `vLLM/RAPPORT_J4BIS-BIS.md` | infra | `infra/vllm/RAPPORT_J4BIS-BIS.md` |
| `vLLM/README.md` | infra | `infra/vllm/README.md` |
| `vLLM/diagnose_image.sh` | infra | `infra/vllm/diagnose_image.sh` |
| `vLLM/docker-compose.yml` | infra | `infra/vllm/docker-compose.yml` |
| `vLLM/scripts/tests/test_compose.py` | infra | `infra/vllm/scripts/tests/test_compose.py` |
| `vLLM/smoke.sh` | infra | `infra/vllm/smoke.sh` |
| `vLLM/smoke_chat.py` | infra | `infra/vllm/smoke_chat.py` |

## 3. Cas particuliers

* `scripts/_common.py` est **scindé** : les helpers génériques (`sha256_file`,
  `git_commit`, `list_source_sidecar_files`, `index_fingerprint`) vont dans
  `academiia/core/common.py` ; le mapping Gemma 4 ↔ DiffusionGemma et les
  helpers de τ legacy (`family_of`, `tau_filename`, `check_tau_manifest`) vont
  dans `modules/a2d/a2d/mapping.py`.
* `scripts/paired_bootstrap.py` reste un script d'évaluation, mais `wilson` et
  `bootstrap_ci95` sont **déplacés** dans `academiia/core/stats.py` (mêmes
  algorithme et graine : le verdict G1 doit rester numériquement identique).
* `eval/` est scindé : les résultats A2D (`dream_a2d*`) vont dans
  `modules/a2d/eval/`, les preuves J5 du build Jainius dans
  `modules/a2d/legacy/eval/` (elles documentent une ablation hors papier).
* `scripts/smoke_int8.py` cible `checkpoints/Jainius` / `diffusiongemma_it` :
  outil legacy, classé `legacy`.
* Les données de `eval/**` et `docs/reports/**` sont déplacées **sans**
  modification de contenu : leurs chemins internes restent une provenance
  historique (exception documentée pour le critère 4c).

## 4. À TRANCHER

1. **`harness/README.md`** (convention des harnais) est classé `a2d` faute de
   second harnais. À remonter à la racine dès qu'un `harness/nemotron/` ou
   `harness/diffusiongemma/` existera.
2. **`scripts/download_models.py`** est classé `a2d` (les checkpoints servent G1/G2)
   mais pourrait devenir un outil `infra` s'il sert un jour un module non-A2D.
3. **Preuves J5** (`modules/a2d/legacy/eval/`) : conservées pour la traçabilité.
   Décider si elles restent versionnées ou sont archivées hors dépôt.
4. **Caches du harnais** : `harness/dream/{.venv,.fork,.cache}` (≈ 8,4 Go, ignorés
   par git) sont **restés à l'ancien emplacement** : seuls les fichiers
   versionnés du harnais ont été déplacés vers `modules/a2d/harness/dream/`.
   Le venv contient des chemins absolus et serait de toute façon à
   reconstruire. Suppression après reconstruction au nouveau chemin —
   **accord utilisateur requis**.
5. **`modules/dopd/`** : module **d-OPD** (arXiv 2609.35362 ; dev sur
   Qwen3-0.6B). **Créé en J12** (squelette : paquet `academiia.dopd`, cible
   future-aware NumPy/CPU, aucun entraînement). *(nom OPDLM -> d-OPD en J11)*
6. **Distribution** : `academiia` est un paquet local (`pip install -e .`),
   API non stabilisée, usage personnel — pas de publication PyPI.

## 5. Exceptions au balayage des anciens chemins (critère 4c)

Le critère « aucun ancien chemin dans les fichiers non historiques » admet ces
**exceptions explicites**, toutes étant des documents **historiques** ou des
mentions **méta** :

| Emplacement | Nature de l'occurrence |
|---|---|
| `modules/a2d/docs/reports/**` | rapports J6–J8 : bandeau de migration en tête, chemins de l'époque conservés |
| `modules/a2d/legacy/docs/**` | archives Jainius (AUDIT.md, GUIDE_RESTANT.md) : non réécrites |
| `docs/adr/path-migration.md` | la table elle-même (colonne « ancien chemin ») |
| `docs/adr/ADR-001-disposition.md` | description de la disposition **antérieure** |
| `modules/a2d/docs/AUDIT_A2D_vs_PAPIER.md` | audit daté : le document annonce lui-même que ses chemins sont ceux de l'époque |
| `modules/a2d/docs/VP_ADAPTERS.md` | chemins du **dépôt publié** des adaptateurs médicaux (`scripts/train.py`, …) : sans rapport avec ce dépôt |
| `docs/KNOWN_ISSUES.md` | entrées historiques explicitement conservées (annonce en tête de section) — fichier **déplacé de la racine en J11** |

Hors de ces exceptions, le balayage ne trouve **aucune** occurrence d'ancien
chemin dans les fichiers versionnés.

## 6. Déplacements J11 (stabilisation post-J10)

| Ancien chemin | Nouveau chemin | Nature |
|---|---|---|
| `CHECKLIST.md` | `docs/CHECKLIST.md` | document transversal (portes G1/G2/G3) : sorti de la racine pour que la racine ne contienne que l'entrée dans le pipeline |
| `KNOWN_ISSUES.md` | `docs/KNOWN_ISSUES.md` | idem (anomalies connues) |

Les liens relatifs de tous les documents **actifs** ont été mis à jour ; les
rapports historiques (`modules/a2d/docs/reports/**`) et les archives legacy
(`modules/a2d/legacy/docs/**`) gardent leurs chemins d'époque (bandeau de
migration).

## 7. Création J12 (module d-OPD)

Aucun chemin existant n'est déplacé. J12 **crée** :

| Nouveau chemin | Nature |
|---|---|
| `modules/dopd/dopd/` | paquet `academiia.dopd` (cible future-aware, configuration) |
| `modules/dopd/docs/DOPD_SPEC.md` | brief de conception d-OPD |
| `modules/dopd/docs/reports/J12.md` | rapport de mission J12 |
| `modules/dopd/tests/` | tests CPU du squelette |

Le paquet est exposé par `pyproject.toml` (`academiia.dopd` →
`modules/dopd/dopd`). Il n'importe pas `academiia.a2d` (ADR-001).

## 8. Ajouts J13 (interfaces d-OPD + harness)

Aucun chemin existant n'est déplacé. J13 **crée** :

| Nouveau chemin | Nature |
|---|---|
| `modules/dopd/dopd/interfaces.py` | contrats ABC : `TeacherScorer`, `RolloutRecorder`, `TaskVerifier`, `LineageWriter` |
| `modules/dopd/harness/smoke_qwen3.py` | script de smoke **hors paquet** (torch/transformers autorisés ici) |
| `modules/dopd/tests/test_dopd_interfaces.py` | tests CPU des contrats |
| `modules/dopd/docs/reports/J13.md` | rapport de mission J13 |

`modules/dopd/harness/` est **hors** du paquet importable `academiia.dopd` :
il n'est pas déclaré dans `pyproject.toml` et ne peut donc pas être importé
comme `academiia.dopd.harness`. C'est ce qui autorise `torch` dans les scripts
sans violer ADR-001.
