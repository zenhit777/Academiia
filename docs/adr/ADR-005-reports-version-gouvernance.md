# ADR-005 — Reports différés, version unique et gouvernance de mission

- **Statut** : accepté (J14, 2026-10-23)
- **Décide par** : mission J14, phases 2–3
- **Contexte** : ADR-001 (disposition), ADR-002 (provenance), ADR-003 (règle de deux), ADR-004 (import `torch` paresseux)
- **Portée** : documents du dépôt et politiques de mission ; aucune règle scientifique d-OPD ou A2D n'est modifiée

> **Vocabulaire.** Les références `M1`, `M2`, `M6` et `M7` de cet ADR sont
> les **constats d'audit externe** du 23 octobre 2026. Elles ne coïncident pas
> avec les noms de modèles `M0`–`M3` utilisés dans les rapports A2D (Dream).
> Aucun seuil, verdict ou résultat G1 n'est modifié par cet ADR.

## Contexte

L'audit externe du 23 octobre 2026 a relevé des écarts de gouvernance qui
n'affectent pas la justesse numérique du code mais bloquent une release :

1. **M1 — scripts et données mêlés.** Des scripts jetables et des données de
   campagne ont été proches, voire dans, le paquet distribuable. Le packaging
   actuel interdit explicitement l'embarquement des résultats de campagne et
   ne déclare que les paquets `academiia.core`, `academiia.a2d`,
   `academiia.a2d.eval` et `academiia.dopd`.
2. **M2 — monolithe `merge.py`.** `modules/a2d/a2d/merge.py` compte 1 369
   lignes. Il concentre le moteur de fusion, les plans de tenseurs et la
   vérification post-écriture.
3. **M6 — verdict G1 non rejouable hors harnais.** Le verdict est versionné
   (`modules/a2d/eval/dream_a2d/G1_VERDICT.json`) et reproductible depuis les
   échantillons, mais le réexécution complet du protocole de génération exige
   l'environnement `lm_eval` du harnais Dream.
4. **M7 — métadonnées de version.** Deux constantes de version coexistent :
   `version` dans `pyproject.toml` et `PACKAGE_VERSION` dans
   `modules/dopd/dopd/config.py`.
5. **Python 3.12.** Le plancher Python est passé de 3.10/3.11 à **3.12**
   dans la CI, `pyproject.toml`, `requirements.txt` et `AGENTS.md` (commits
   `b1302ac`, `58a65b9`).
6. **Lint hors périmètre.** La CI ne contrôlait que `academiia/`,
   `modules/a2d/` et `modules/dopd/` ; `tests/`, `tools/` et `infra/` en
   étaient exclus, et une violation E741 y subsistait.
7. **Boucle de consignation.** Plusieurs rapports listaient leurs propres
   commits de documentation, puis un nouveau commit recordait ce hash, ce qui
   créait une boucle sans information scientifique nouvelle.

## Décision

### 1. Reports différés (M1, M2, M6, M7)

Les quatre constats sont **reportés**, avec une condition de reprise
explicite. Aucun n'est traité dans J14 : tous touchent au cœur du pipeline
A2D ou à la chaîne de release, ce qui sort du périmètre de la porte
scientifique d-OPD.

| Réf. | Report | Condition de reprise | Propriétaire |
|---|---|---|---|
| **M1** | scripts jetables et données hors paquet ; wheel sans résultats | Au prochain changement de packaging ou d'entrée de données | pipeline A2D / packaging |
| **M2** | monolithe `merge.py` (1 369 lignes) | Mission « découpage `merge.py` » avec tests de non-régression numériques bit-à-bit | bibliothèque A2D |
| **M6** | verdict G1 rejouable seulement avec le harnais Dream complet | Mission « reproductibilité G1 » : environnement `lm_eval` épinglé ou verdict recalculable depuis les échantillons seuls | trajectoire A2D |
| **M7** | version dupliquée entre `pyproject.toml` et `PACKAGE_VERSION` | Mission release : source unique générée ou importée du `pyproject.toml` | packaging |

### 2. Python 3.12

**3.12 est le plancher unique du projet.** `requires-python = ">=3.12"`, la
matrice CI mono-version `["3.12"]`, `requirements.txt` (pointeur vers
`pyproject.toml`) et `AGENTS.md` le déclarent de façon cohérente. Une
migration vers une version plus récente est une décision de release
distincte ; aucune version antérieure n'est supportée.

### 3. Lint

* Le périmètre ruff **E/F couvre tout le dépôt** : `academiia/`, `modules/`,
  `tests/`, `tools/`, `infra/`. La violation E741 restante a été corrigée.
* **B/SIM ne sont pas réactivés en J14.** L'inventaire Ruff sur le dépôt
  courant relève 23 alertes B/SIM réparties sur 13 codes (`B007`, `B027`,
  `B904`, `B905`, `SIM102`, `SIM103`, `SIM105`, `SIM108`, `SIM110`, `SIM114`,
  `SIM115`, `SIM118`, `SIM210`, `SIM300`). Les réactivations `SIM114` et
  `SIM300` sont mécaniques ; les autres exigent un examen. Activer B/SIM
  exige donc une mission de correction dédiée, après examination code par
  code — pas un ajout aveugle à `select`.

### 4. Politique de rapport de mission

1. Un rapport de mission **ne liste pas ses propres commits de
   consignation**. Il cite la plage `HEAD_départ..HEAD_arrivée`, qui est
   vérifiable dans l'historique.
2. Les preuves de CI sont citées **par lien de run**, sans hash de commit de
   consignation supplémentaire.
3. Les désaccords avec un audit sont consignés dans un **journal d'audit
   séparé**, pas dans le rapport de mission lui-même.
4. Le rapport consigne les **faits scientifiques et de gouvernance** : critères
   pré-enregistrés, résultats, budget, décisions et vérifications.

### 5. Placement des documents de statut

`docs/CHECKLIST.md` et `docs/KNOWN_ISSUES.md` **restent globaux** : ils
portent les portes de trajectoire (G1/G2/G3) et les anomalies transverses qui
ne relèvent ni d'A2D ni de d-OPD. Leur contenu historique A2D/Jainius est
conservé tel quel pour ne pas réécrire l'historique ; d-OPD dispose déjà de
ses propres documents (`modules/dopd/docs/DOPD_SPEC.md`,
`modules/dopd/docs/reports/`). Un index global de portes par module sera
demandé par une mission de documentation dédiée si la densité des portes
augmente.

### 6. Règle 11 — push et tags d'archive

La règle 11 d'`AGENTS.md` est réécrite pour supprimer la contradiction
constatée par l'audit : le push est **interdit par défaut**, et **autorisé
mission par mission par accord écrit explicite** de l'utilisateur.

Décision retenue pour J14 (accord explicite de l'utilisateur, le 23 octobre
2026) : l'agent est autorisé à pousser sur `origin/main` et à gérer les tags
d'archive du dépôt, en utilisant la **clé personnelle** de l'utilisateur via le
client SSH disponible sous WSL2
(`/mnt/c/Windows/System32/OpenSSH/ssh.exe`). Aucune clé de déploiement
limitée au dépôt n'est configurée : l'accord porte donc sur la clé
personnelle, à périmètre de dépôt. Les commits poussés, les tags touchés et le
transport employé sont consignés dans le rapport de mission.

Résiduel accepté : la clé personnelle n'est pas restreinte au dépôt. La
création d'une clé de déploiement limitée reste recommandée et devra faire
l'objet d'un accord distinct.

## Conséquences

* Le lint CI porte désormais sur la totalité du code Python du dépôt.
* B/SIM restent exclus de la sélection, mais leur état est inventorié et
  leur activation est planifiée plutôt que perdue.
* Python 3.12 est la seule version supportée et testée.
* La règle 11 d'`AGENTS.md` et la pratique J14 sont en cohérence : le push
  reste une exception explicitement autorisée, jamais une pratique implicite.
* La version `0.1.0` reste une version de développement : aucune release
  `v0.1.0` n'est publiée tant que M1/M2/M6/M7 ne sont pas clos.
