# Review architecte N2 — mission J6 (verdict)

**Date** : 2026-10-20 · **Auteur de la review** : l'architecte (N2) · **Reçu par** : OhMyPi,
développeur, pour la mission J6bis (clôture corrective de J6).

> Ce document est le verdict N2. Il est reproduit **tel quel** depuis la review, sans reformulation
> ni interprétation. Les corrections correspondantes sont traitées dans `docs/reports/J6bis.md`.

## Verdict N2 sur J6 (à copier tel quel dans docs/reports/REVIEW_N2_J6.md)

J6 est REFUSÉ. G1 reste OUVERTE. Conservé : formules de transfert/merge sur entrées complètes, τ à
la volée, safe_open, FP32 + un cast BF16, écritures atomiques, géométrie Table 21 reproduite,
équivalence LoRA/PEFT, harnais et checkpoints de référence Dream. Confirmés par lecture du code :
BLK-01, BLK-02, MAJ-01 à MAJ-06 et MAJ-08 (MAJ-07 invalidé) de l'audit N1. Constats N2 additionnels :
- N2-1 : build_plan ne parcourt que les modules de τ_A ; un module présent seulement dans τ_D est ignoré.
- N2-2 : la vérification post-écriture compare une norme, pas une projection signée ; elle est
  structurellement trop faible (inversion de signe, mode erroné, couverture partielle passent).
- N2-3 : écritures hors workspace non déclarées dans le rapport (/home/sjteb/dream-harness) et emploi de
  « validé » sans porte franchie. (Le constat N1 MAJ-07 « aucun push » est INVALIDÉ : le push a été fait
  par l'utilisateur après la livraison du rapport J6, qui était exact à sa date ; le reflog n'identifie
  pas l'auteur d'un push.)
- N2-4 : le critère C0.5 du prompt J6 (±3 pts, IFEval complet) était irréaliste en coût ; il est
  remplacé par le protocole de la section D.

---

## Correspondance exigence → livrable J6bis

| Constat | Exigence J6bis | Livrable |
|---|---|---|
| BLK-01, BLK-02, N2-1 | B1, B2 | plan sur l'union τ_A ∪ τ_D, politique `--coverage`, plan vide refusé |
| MAJ-01 | B5 | `verify_merge` appelle `validate_config` |
| MAJ-02, N2-2 | B4 | vérification v2 : projection signée (moindres carrés), Gram float64 |
| MAJ-03 | B3 | fonction unique `expected_tensor()` (alignement des lignes d'embedding inclus) |
| MAJ-04 | A2 | « MATH-500 » → « MATH-175 (sous-ensemble minerva_math, 175/5000) » |
| MAJ-05, N2-4 | A2, D2 | C0.5 = NON ATTEINT ; protocole G1 de la section D |
| MAJ-06 | A2 | évaluation ≠ validation de la bibliothèque de fusion |
| MAJ-07 | — | **invalidé** par N2-3 ; voir ERRATUM N2 en tête de `docs/reports/AUDIT_J6.md` |
| MAJ-08 | B7 | design `docs/design/G2_adapter_scaling.md` (aucun code) |
| N2-3 | C1–C7 | harnais versionné dans le dépôt, inventaire hors workspace documenté |
