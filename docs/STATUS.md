# STATUS — index des modules

Document **court** : il pointe vers l'état détaillé de chaque module. Il ne
remplace ni `docs/CHECKLIST.md` (portes G1/G2/G3) ni les rapports de mission.

| Module | Paquet importable | État | Détail |
|---|---|---|---|
| `academiia/core/` | `academiia.core` | **stable** — code générique (stats bootstrap/Wilson, empreintes, E/S safetensors). Aucune dépendance vers un module. | `academiia/core/` |
| `modules/a2d/` | `academiia.a2d` | **validé mécaniquement** — méthode A2D (transfert, merge, ablations, vérification post hoc). G1 **non franchie** (verdict inchangé, aucun seuil rejoué). | `modules/a2d/STATUS.md` |
| `modules/a2d/legacy/` | — | **déprécié** — pipeline Jainius (hors papier), conservé pour l'historique. | `modules/a2d/legacy/README.md` |
| `infra/vllm/` | — | **service d'inférence** (compose FP8 épinglé, smoke, diagnostic). | `infra/vllm/README.md` |
| `modules/dopd/` | `academiia.dopd` | **porte Annexe D franchie avec réserves de lecture (J14), renforcée J15, auditée J16, `TeacherScorer` réel livré J17, **GO J18 non certifié** (audit corrigé : écart critique, mesure à reconstruire) — module **d-OPD** (arXiv 2609.35362) : cible future-aware (NumPy/CPU), configuration, **contrats d'interface** (professeur, rollout, vérificateur, lignée ADR-002), **noyau du diagnostic Annexe D** (énumération exacte, KL, bootstrap, décision de porte), **`TeacherScorer` réel** Qwen3-0.6B (isolé sous `harness/`, ADR-001 respecté). Diagnostic multi-prompt sur Qwen3-0.6B **exécuté et versionné** : `ΔKL = +0,1623` (J14) puis multi-prompt `ΔKL = +0,0802`, IC 95 % `[+0,0638 ; +0,0962]`. Réduction relative **8,58 %** (multi-prompt), médiane nulle et queue nuisible de 13,3 % : décision **GO conditionnel**. **J17** : cache de préfixe exact dans la tolérance C3 ; écart BF16 0,395069 imputé à la précision du protocole (pas au cache), écart FP32 ≤ 6,2e-05. **C2 échoué en BF16 sur états réels** (seuil 0,35) ; décision J18 **GO** par le run complet (C1–C10 déclarés tenus) : effet future-aware mesuré en FP32 (professeur gelé), porte Annexe D déclarée **ouverte pour un `StudentPolicy` réduit (≤ 0,6B) en J19**. **Audit J18 (corrigé le 2026-10-27)** : le GO de J18 **n'est pas certifié** — dans la géométrie J18 la cible future-aware coïncide avec la posterior exacte (4,5e-10 près), donc `future_kl ≡ 0` et `ΔKL ≡ KL(exact‖causal)` **par construction** : C1–C4 sont des identités, non des observations. **J19 ne doit pas s'appuyer sur ce GO** ; la mesure doit être reconstruite (futur échantillonné séparément, convention J15) puis re-lancer. **J18R phase 1 (2026-10-27)** : harnais de décision `modules/dopd/harness/run_j18r_decision.py` + tests T1–T8 livrés, CI verte (315 passed, 1 skipped) ; **aucun run, aucune décision** — phase 2 bloquée faute d'acceptation écrite du PRE §11, le harnais refuse d'exécuter sans `--confirm-accepted` et n'utilise pas torch sur la voie NumPy. Aucun entraînement, aucun rollout réel, aucun `StudentPolicy`. **Revue N2** : 7 constats corrigés (dépendances bornées, empreintes ADR-002, tests sensibles au préfixe/positions/masque, échec dur hors dépôt) ; sensibilité prouvée par injection de fautes. **Réserve de gouvernance** : push non autorisé sur `origin/main` (de `08d62bd` à `50464eb`) ; les commits suivants sont locaux. | `modules/dopd/docs/DOPD_SPEC.md`, `J17.md`, `AUDIT_J18.md`, `reports/J18R_PRE.md`, `reports/J18R.md` |

## Règles de navigation

- La bibliothèque A2D s'importe comme `academiia.a2d`, le code générique comme
  `academiia.core` : `pip install -e .` puis `import academiia.a2d`.
- Le squelette d-OPD s'importe comme `academiia.dopd` ; il n'expose aucun CLI et
  n'importe pas `academiia.a2d`. Son harnais (`modules/dopd/harness/`) est hors
  du paquet importable : c'est le seul endroit où `torch` est autorisé pour
  d-OPD.
- Règle de dépendance : un module importe `academiia.core`, **jamais** un autre
  module (ADR-001, `docs/adr/ADR-001-disposition.md`).
- Les anciens chemins sont listés dans `docs/adr/path-migration.md`.
- L'historique (rapports J6–J8) n'est **pas** réécrit : il renvoie à la table
  de migration.
