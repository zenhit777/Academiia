# Audit N1 OpenCode — mission J6ter, 2026-10-20/21

> **Document historique (J6–J8).** Les chemins cités dans ce rapport sont
> ceux de l'époque : la table de migration des chemins est
> [`docs/adr/path-migration.md`](../../../docs/adr/path-migration.md).

**Date** : 2026-10-20 (campagne) / 2026-10-21 (versionnement) · **Auteur de l'audit** : OpenCode
(N1) · **Vérification** : architecte (N2) · **Cible auditée** : commit `ad4c888`
(HEAD du dépôt au moment de la campagne d'audit, enregistré dans les manifestes produits
par l'auditeur — §3).

> **Provenance de ce document — à lire avant de l'utiliser.**
> Le rapport N1 d'OpenCode (mission J6ter) **n'a pas été persisté** : il ne figure ni dans le
> dépôt (ni `docs/reports/`, ni `_artefact/audit_j6ter/`), ni dans le contexte de la mission
> J6quater. Ce fichier est donc une **reconstitution** : il consigne le **périmètre**, la
> **méthode**, les **artefacts vérifiables** et les **constats** de l'audit N1, reconstitués à
> partir (i) des artefacts produits par l'auditeur lui-même (`_artefact/audit_j6ter/`), (ii) des
> corrections commitées sous F1–F8, (iii) du verdict N2 rapporté dans la mission J6quater.
> Il ne contient **aucune citation verbatim** de l'auditeur et n'en fabrique pas. Si le texte
> original est fourni, il doit être collé tel quel en §2 (emplacement réservé) ; le reste du
> document est indépendant de cette insertion.
>
> **Numérotation** : les constats F1–F8 sont ceux de l'audit N1 de J6bis
> (`docs/reports/AUDIT_J6BIS.md`) ; l'audit N1 de J6ter **revient sur les corrections** J6ter
> correspondantes. La numérotation n'est pas réinitialisée.

## 1. Objet et périmètre de l'audit N1 (J6ter)

Reprise, par un auditeur indépendant (N1, OpenCode, lecture seule), des **corrections J6ter**
(F1–F8), c'est-à-dire :

* la **conformité A2D** du moteur de fusion (équations (1) et (2), coefficients, mappings,
  schéma numérique) sur le chemin Qwen2.5-7B → Dream-7B ;
* le **garde-fou « coefficient nul »** (F2) : refus par défaut d'un transfert `α=0` et d'un merge
  `λ=0`, option `--allow-empty-coefficient`, trace au manifeste ;
* la **trace de `--no-verify`** (F3) : `post_write_verification.skipped` + `reason` au manifeste ;
* les **ancrages chiffrés** du protocole G1 (F1), les **compteurs de tests** (F4), les
  **latences/projections** d'évaluation (F5), l'**inventaire hors dépôt** (F6), l'**aide CLI**
  `--lambda` (F7), la **provenance des empreintes d'items** (F8).

Hors périmètre : toute exécution GPU (aucun job de fusion ni d'évaluation) — le rapport porte
sur le **code**, les **manifestes** et la **documentation**.

## 2. Rapport de l'auditeur — verbatim

> **Emplacement réservé** : coller ici le rapport N1 d'OpenCode, tel quel, sans reformulation
> (même convention que `docs/reports/REVIEW_N2_J6.md`). En l'état, cette section est vide : le
> texte original n'était pas disponible au moment du versionnement (cf. encadré « Provenance »).

## 3. Méthode et artefacts produits par l'audit

L'auditeur a travaillé dans `_artefact/audit_j6ter/` (hors git), avec `TMPDIR` **redirigé dans le
dépôt** (`_artefact/audit_j6ter/pytest_tmp/`) — conforme à la règle 9 du `AGENTS.md`.

### 3.1 Scénarios CLI rejoués (4 fusions de jouet, 2026-10-20 20:03:46–48)

| Scénario | Dossier de sortie | Résultat observé | Trace |
|---|---|---|---|
| transfert **sans** `--alpha` | `_artefact/audit_j6ter/cli/transfer_no_alpha/` | **vide** ⇒ fusion **refusée** (aucun fichier écrit) | garde-fou F2, sortie 2 |
| merge **sans** `--lambda` | `_artefact/audit_j6ter/cli/merge_no_lambda/` | **vide** ⇒ fusion **refusée** | garde-fou F2, sortie 2 |
| transfert `α=0` **avec** `--allow-empty-coefficient` | `_artefact/audit_j6ter/cli/transfer_allow/` | fusion **écrite**, garde-fou **contourné et déclaré** | `merge_config.json` : `coefficient_guard.status = "bypassed"`, `allow_empty_coefficient = true`, `plan_has_nonzero_deltas = true`, warning explicite |
| transfert `α=0,9` **avec** `--no-verify` | `_artefact/audit_j6ter/cli/transfer_noverify/` | fusion **écrite**, vérification **skippée et tracée** | `merge_config.json` : `post_write_verification = {"skipped": true, "reason": "--no-verify"}` |

Les deux manifestes enregistrent `git_commit = ad4c888` : la campagne porte donc bien sur le HEAD
annoncé. Ils enregistrent aussi `method = "A2D (arXiv:2610.08108) — A2D-Transfer : D_b +
alpha*tau_A (eq. 1)"`, le mapping `identity` (Qwen2.5-7B → Dream-7B, routeur fusionné), le
`dtype = bfloat16`, la couverture `strict` et les empreintes de contenu des shards source.

### 3.2 Vérification du schéma numérique

Pour le scénario `transfer_allow` (α=0), la vérification v2 mesure `c_a_eff ≈ -4,2e-5`
(soit ≈ 0, comme attendu pour un coefficient nul) et conclut `ok = true` : elle est
**structurellement incapable** de voir passer un no-op (elle compare les coefficients effectifs
aux coefficients **demandés**, tous deux nuls) — c'est exactement ce qui fonde la décision
d'ajouter le garde-fou F2.

## 4. Constats et suite donnée

| Constat | Objet | État après J6ter | Commit |
|---|---|---|---|
| **F1** | ancrages +19,4 / +15,2 dans `docs/design/G1_PROTOCOL.md` | **conforme** | `88faf77` |
| **F2** | garde-fou « coefficient nul » (refus, option, trace) | **conforme** | `ca3deef` |
| **F3** | trace de `--no-verify` au manifeste | **conforme** | `a61d298` |
| **F4** | compteurs de tests (65 / 21 mutations) | **conforme** | `9f342ab` |
| **F5** | latences IFEval (~326 s/prompt) et projections | **conforme**, sauf **résidu** en §3 de `EVAL_BUDGET.md` → clos en J6quater | `dc64315` + `0dc55f8` |
| **F6** | inventaire hors dépôt (C5) | **conforme** | `619a6c6` |
| **F7** | aide CLI `--lambda` (seuil 1,0) | **conforme** | `42124bc` |
| **F8** | provenance des empreintes (`fork_git_hash`, `academiia_head_at_run`) | **conforme** | `9574473` + `ad4c888` |

**Résidu F5 (unique)** : la conclusion du §3 de `docs/design/EVAL_BUDGET.md` disait
« MATH-500 (5 000 problèmes) ne l'est pas non plus » — confusion entre le jeu MATH **complet**
(5 000, hors budget) et **MATH-500** (500 sur 5 000, jouable en ≈ 4 h 47 par modèle), en
contradiction avec le tableau du même paragraphe. Requalifié en J6quater (`0dc55f8`) : seul
GSM8K complet est jouable ; IFEval complet (541, ≈ 48 h 59 par modèle) et MATH complet (5 000,
≈ 47 h 51 par modèle) sont hors budget ; MATH-500 reste jouable mais est **hors du critère G1**
(le protocole compare sur MATH-175, 1 h 40 par modèle). Aucun chiffre validé n'a été modifié.

## 5. Verdict N2 (architecte)

Rapporté dans la mission J6quater : **aucune déviation silencieuse A2D** ; **F1–F8 conformes** ;
**65 tests verts** ; **HEAD `ad4c888`** validé. Le seul résidu documentaire restant était la
contradiction du §3 ci-dessus, close en J6quater. (Le verdict N2 lui-même n'est pas versionné :
seul `docs/reports/REVIEW_N2_J6.md` — la review N2 de **J6** — l'est.)

## 6. État de la bibliothèque au terme de J6ter/J6quater

* **Moteur A2D** : conforme (audit N1, review N2) ; garde-fou « coefficient nul » actif par
  défaut ; `--no-verify` tracé ; τ jamais persisté ; vérification post-écriture v2 signée.
* **Suite de tests** : `python3 -m pytest scripts/tests -q` → **65 passed**
  (29 `test_a2d_library.py` + 21 `test_a2d_mutation.py` + 9 `test_pipeline.py` +
  6 `test_eval_compare.py`).
* **Prêt pour J7** : les fusions M2/M3 (transfert et merge) et les évaluations G1 peuvent être
  lancées selon `docs/design/G1_PROTOCOL.md` ; **aucune fusion ni évaluation GPU n'a été
  exécutée** dans le cadre de J6ter/J6quater.

## 7. Références

* `docs/reports/AUDIT_J6BIS.md` — audit N1 de J6bis (F1–F9 d'origine).
* `docs/reports/J6bis.md` §7 — corrections post-audit N1, commits F1–F8.
* `docs/reports/REVIEW_N2_J6.md` — verdict N2 sur J6 (le seul N2 versionné).
* `docs/design/G1_PROTOCOL.md` — protocole de la porte G1 (M0–M3, critères C1–C4).
* `docs/design/EVAL_BUDGET.md` — coûts mesurés et projections (résidu F5 requalifié).
* Artefacts de l'audit N1 : `_artefact/audit_j6ter/` (hors git).
