# STATUS — index des modules

Document **court** : il pointe vers l'état détaillé de chaque module. Il ne
remplace ni `docs/CHECKLIST.md` (portes G1/G2/G3) ni les rapports de mission.

| Module | Paquet importable | État | Détail |
|---|---|---|---|
| `academiia/core/` | `academiia.core` | **stable** — code générique (stats bootstrap/Wilson, empreintes, E/S safetensors). Aucune dépendance vers un module. | `academiia/core/` |
| `modules/a2d/` | `academiia.a2d` | **validé mécaniquement** — méthode A2D (transfert, merge, ablations, vérification post hoc). G1 **non franchie** (verdict inchangé, aucun seuil rejoué). | `modules/a2d/STATUS.md` |
| `modules/a2d/legacy/` | — | **déprécié** — pipeline Jainius (hors papier), conservé pour l'historique. | `modules/a2d/legacy/README.md` |
| `infra/vllm/` | — | **service d'inférence** (compose FP8 épinglé, smoke, diagnostic). | `infra/vllm/README.md` |
| `modules/dopd/` | `academiia.dopd` | **porte Annexe D franchie avec réserves de lecture (J14), renforcée J15, auditée J16, `TeacherScorer` réel livré J17** — module **d-OPD** (arXiv 2609.35362) : cible future-aware (NumPy/CPU), configuration, **contrats d'interface** (professeur, rollout, vérificateur, lignée ADR-002), **noyau du diagnostic Annexe D** (énumération exacte, KL, bootstrap, décision de porte), **`TeacherScorer` réel** Qwen3-0.6B (isolé sous `harness/`, ADR-001 respecté). Diagnostic multi-prompt sur Qwen3-0.6B **exécuté et versionné** : `ΔKL = +0,1623` (J14) puis multi-prompt `ΔKL = +0,0802`, IC 95 % `[+0,0638 ; +0,0962]`. Réduction relative **8,58 %** (multi-prompt), médiane nulle et queue nuisible de 13,3 % : décision **GO conditionnel**. **J17** : cache de préfixe exact (écart FP32 1,1e-05), latence 53,2 ms à 16 branches ; **13/14** critères pré-enregistrés tenus, C2 échoué en BF16 sur états réels (0,395 > 0,35) sans atteindre les conditions de NO-GO ; décision J18 **GO conditionnel**. Aucun entraînement, aucun rollout réel, aucun `StudentPolicy`. | `modules/dopd/docs/DOPD_SPEC.md`, `J17.md` |

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
