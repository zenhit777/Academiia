# Review N2 externe — HEAD `16a0232` — 2026-10-21 — GO conditionnel pour J7

> **Document historique (J6–J8).** Les chemins cités dans ce rapport sont
> ceux de l'époque : la table de migration des chemins est
> [`docs/adr/path-migration.md`](../../../docs/adr/path-migration.md).

**Date** : 2026-10-21 · **Auteur de la review** : l'architecte (N2), audit externe · **HEAD
revu** : `16a0232` · **Reçu par** : OhMyPi, développeur, pour la mission **J7** (franchissement
de la porte G1).

> **Provenance du document — à lire avant de l'utiliser.**
> Le **texte intégral** de la review N2 externe **n'a pas été transmis** avec la mission J7 : il
> ne figure ni dans le dépôt (ni `docs/reports/`, ni `_artefact/audit_j6quater/`) ni dans le
> contexte de la mission, qui n'en donne que le **contenu** (absence de déviation silencieuse
> A2D, GO conditionnel pour J7, trois constats M1/M2/M3). Ce document est donc une
> **reconstitution** : il consigne le verdict, les constats et **les amendements effectivement
> intégrés** au dépôt, avec un emplacement réservé (§1) pour le verbatim. Il ne contient
> **aucune citation fabriquée** de l'auditeur. Si le texte original est fourni, il doit être
> collé tel quel en §1, sans modifier le reste du document.

## 1. Verdict N2 — verbatim

> **Emplacement réservé** : coller ici la review N2 externe **tel quel**, sans reformulation
> (même convention que `docs/reports/REVIEW_N2_J6.md`). En l'état, cette section est vide : le
> texte original n'était pas disponible au moment du versionnement (cf. encadré « Provenance »).

### 1.1 Verdict reconstitué

* **Aucune déviation silencieuse A2D** constatée sur HEAD `16a0232`.
* **GO conditionnel** pour la mission **J7** (fusions M2/M3 + évaluations G1).
* Condition : les trois amendements **M1**, **M2** et **M3** doivent être **intégrés avant les
  évaluations GPU coûteuses** (c'est-à-dire avant les Phases 2 et 3 de J7).

## 2. Les trois constats et leur intégration

| Constat | Contenu | Intégration dans le dépôt | État |
|---|---|---|---|
| **M1** | les critères C1–C4 portent sur des **estimations ponctuelles** trop fragiles au vu des IC | `docs/design/G1_PROTOCOL.md` §4.1 (tableau C1–C4) et §4.2 (définition de la borne basse) : C1/C2 deviennent une **double condition** (point estimate **et** borne basse de l'IC apparié 95 % > 0) ; C3/C4 sont requalifiés en **« non-régression non réfutée »** (borne basse ≥ −2 ; « établie » seulement si borne basse ≥ 0) | **intégré** (`590c37a`) |
| **M1 (bis)** | le seuil α > 1,5 était justifié par une **mauvaise citation** (Fig. 8 du papier) | `docs/A2D_SPEC.md` §2 et `scripts/a2d/merge.py` (commentaire + message d'avertissement) : le seuil est déclaré **choix de projet** (seuil d'avertissement arbitraire ; la Fig. 8 est le diagramme des succès communs) | **intégré** (`ffaf325`) |
| **M2** | le **bruit de décodage** (stochastique + dépendance au lot) n'est pas quantifié | Phase 1 de J7 : M1 rejoué à **graine alternative** (GSM8K n=300, lot 4, protocole figé) → **plancher de bruit** ; il est **rapporté avec chaque écart** comme référence d'interprétation (`docs/design/G1_PROTOCOL.md` §4.1) | **intégré au protocole**, mesure **en cours de J7** |
| **M3** | **traçabilité documentaire des verdicts** à renforcer | Cet audit est **versionné** (le présent document) ; le verdict G1 sera produit dans `docs/reports/J7.md` et reporté dans `CHECKLIST.md` ; les preuves (manifestes de fusion, empreintes d'items, IC appariés) sont versionnées avec eux | **intégré** |

## 3. Ce que la review a vérifié

| Élément | Résultat rapporté |
|---|---|
| Déviation silencieuse A2D (équations, coefficients, mappings, schéma numérique) | **aucune** |
| Suite de tests | **65 verts** (`python3 -m pytest scripts/tests -q`) |
| HEAD revu | `16a0232` (fin de J6quater) |
| Verdict pour J7 | **GO conditionnel** (condition : M1/M2/M3 intégrés avant les évaluations GPU) |

## 4. Suivi

Les amendements de la Phase 0 de J7 sont commits dans `docs/reports/J7.md` (journal des
commits). La mesure du plancher de bruit (M2) et le verdict G1 (Phase 4) y sont consignés.
