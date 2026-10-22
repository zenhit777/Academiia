# Audit de la mission J7 — conformité aux directives et à la méthode A2D

> **Document historique (J6–J8).** Les chemins cités dans ce rapport sont
> ceux de l'époque : la table de migration des chemins est
> [`docs/adr/path-migration.md`](../../../docs/adr/path-migration.md).

**Date** : 2026-10-21 · **HEAD audité** : `027b952` · **Auditeur** : agent de codage
(OpenCode), sur demande de l'utilisateur · **Périmètre** : mission J7, Phases 0–4 ·
**Mode** : lecture des artefacts versionnés et recalculs ciblés, **sans ré-exécution GPU**.

> **Résumé.** Les quatre phases de J7 sont présentes, traçables et conformes aux
> directives. **Aucune déviation silencieuse** par rapport à A2D n'a été détectée :
> équations (1)/(2), coefficients 0,9/0,6 (Table 6), mapping identité, FP32 → un seul
> cast BF16, τ jamais persisté, vérification v2. Le verdict **G1 NON FRANCHIE** est
> correctement dérivé. Cinq écarts de forme/traçabilité ont été relevés ; quatre sont
> corrigés dans `28e8410`, le cinquième (traçabilité M3) est documenté sans changement ;
> le verbatim N2 reste une entrée manquante (constat 4).

## 1. Périmètre et méthode

### 1.1 Documents et artefacts audités

| Élément | Rôle |
|---|---|
| `docs/design/G1_PROTOCOL.md` | protocole pré-enregistré, critères C1–C4 amendés |
| `docs/reports/J7.md` | rapport de mission (Phases 0–4, verdict) |
| `docs/reports/REVIEW_N2_J6QUATER.md` | revue N2 externe (reconstitution) |
| `docs/A2D_SPEC.md` | spécification A2D (équations, Table 6, cibles) |
| `scripts/a2d/`, `scripts/merge_a2d.py`, `scripts/verify_merge.py` | moteur de fusion et vérification |
| `scripts/g1_verdict.py`, `scripts/seed_noise_floor.py`, `scripts/paired_bootstrap.py` | verdict et statistiques |
| `checkpoints/dream_transfer_a09/merge_config.json` | manifeste M2 |
| `checkpoints/dream_merge_l06/merge_config.json` | manifeste M3 |
| `eval/dream_a2d/**` | 12 runs (4 modèles × 3 tâches) |
| `eval/dream_a2d/G1_VERDICT.json`, `SUMMARY.md`, `ITEM_FINGERPRINTS.json` | verdict, scores, empreintes |

### 1.2 Ce qui n'a pas été refait

- **Aucune évaluation GPU.** Les 6 h 41 de la Phase 3 et les 48 min 43 de la Phase 1
  ne sont pas relancées : les résultats sont audités sur les fichiers `results_*.json`,
  les réponses brutes et les manifestes.
- **M3 n'a pas été revérifié indépendamment.** La vérification `verify_merge` de M2 a
  été rejouée sur CPU/disque ; celle de M3 a été interrompue. Le manifeste M3 porte la
  même vérification v2 (`ok=true`, `max_ulp=0`), ce qui suffit à la conclusion.
- **Le PDF n'a pas été relu en natif** : le texte a été extrait dans
  `_artefact/j7_audit/a2d_paper.txt` (PyMuPDF) pour vérifier les équations et la Table 6.

### 1.3 Recalculs indépendants

| Contrôle | Résultat |
|---|---|
| Recalcul des 4 IC appariés (10 000 rééchantillonnages, graine 0) | **identique** au JSON versionné |
| Rejeu de `g1_verdict.py` | sortie **identique** à `G1_VERDICT.json` (code 1 = verdict négatif, pas une erreur) |
| `verify_merge` M2 | `ok=true`, 339/339, `c_a_eff=0,9106`, `max_ulp=0`, toutes familles OK |
| Configs des 12 runs | `model_args` identiques **sauf `pretrained`** ; `task_hashes`, `system_instruction_sha`, `versions`, `n_samples` identiques |
| `pytest scripts/tests -q` (TMP redirigé dans `_artefact/`) | **73 passed** |
| `git status --short` | vide ; **aucun push** |

Les fichiers d'audit non versionnés sont dans `_artefact/j7_audit/`
(`a2d_paper.txt`, `verify_m2.json`, `g1_verdict_replay.json`).

## 2. Verdict global

| Question | Réponse |
|---|---|
| Les directives J7 ont-elles été suivies ? | **Oui**, sur les quatre phases. |
| Y a-t-il une déviation silencieuse A2D ? | **Non détectée.** |
| Le verdict G1 est-il correctement dérivé ? | **Oui** : `G1 NON FRANCHIE`, C2 seul atteint. |
| Les seuils ont-ils été modifiés après les résultats ? | **Non** (voir §7.3). |
| Reste-t-il des problèmes ? | **Cinq écarts de forme/traçabilité**, aucun bloquant. |

## 3. Phase 0 — amendements documentaires

| Exigence | Preuve | Statut |
|---|---|---|
| C1/C2 en **double condition** (point estimate **et** borne basse de l'IC > 0) | `G1_PROTOCOL.md` §4.1, lignes 133-152 ; commit `590c37a` | conforme |
| C3/C4 requalifiés en **non-régression non réfutée** (borne basse ≥ −2 ; « établie » si ≥ 0) | `G1_PROTOCOL.md` §4.1, lignes 153-161 | conforme |
| Citation **Fig. 8** corrigée (seuil α > 1,5 = choix de projet) | `A2D_SPEC.md` §2 ; `scripts/a2d/merge.py` ; commit `ffaf325` | conforme |
| Revue N2 versionnée | `docs/reports/REVIEW_N2_J6QUATER.md` ; commit `a29ef26` | conforme, avec la réserve du verbatim (§9, constat 4) |
| Mission J7 ouverte avec M1/M2/M3 intégrés | `docs/reports/J7.md` ; commit `1766f6a` | conforme |
| `CHECKLIST.md` / `KNOWN_ISSUES.md` à jour | état G1, bruit, défauts corrigés | conforme après `28e8410` |

## 4. Phase 1 — plancher de bruit de décodage

**Dispositif** : M1 (`checkpoints/dream_7b_instruct`) rejoué sur GSM8K n=300, lot 4,
protocole identique, **seule** la graine torch change (`0,1234,1234,1234` →
`0,1234,4321,1234`).

| Contrôle | Résultat |
|---|---|
| Différence de config | **uniquement** `torch_seed` 1234 → 4321 |
| Empreintes d'items / prompts | identiques |
| Réponses brutes identiques | **300/300 (100,0 %)** |
| Précision `flexible-extract` | **84,00 %** dans les deux runs |
| Écart apparié | **+0,00 pt**, IC **[+0,00 ; +0,00]** |
| Discordantes (McNemar) | B seul 0, A seul 0 |

**Interprétation** : à `temperature=0,1`, `top_p=0,9`, `alg=entropy` et lot 4, la
distribution est assez piquée pour que `torch.multinomial` ne change aucune sortie.
La graine est bien câblée (`evaluator.py` → `torch.manual_seed` ; `gen_utils.py` →
`torch.multinomial`), mais elle n'a aucun effet observable dans ce régime.

**Limites, déjà documentées dans J7 §5** : une seule tâche (GSM8K, 256 tokens) et un
seul modèle (M1). Les planchers de MATH (512 tokens) et d'IFEval (1 280 tokens) ne sont
pas mesurés. La dépendance au **lot** n'est pas balayée : elle est **contrôlée** en
gardant `batch_size=4` pour les 12 runs, pas mesurée.

## 5. Phase 2 — fusions M2 et M3

| | **M2** `dream_transfer_a09` | **M3** `dream_merge_l06` |
|---|---|---|
| Mode / équation | transfert, eq. (1) `D_b + ατ_A` | merge, eq. (2) `D_p + λ(τ_A − τ_D)` |
| Coefficient demandé | α = **0,9** | λ = **0,6** |
| Destinataire | `dream_7b_base` | `dream_7b_instruct` |
| Couverture | `A-only` **339/339** | `both` **339/339** |
| Tenseurs fusionnés / inchangés | **339 / 0** | **339 / 0** |
| `post_write_verification` | **ok=true**, version **2**, `max_ulp=0` | **ok=true**, version **2**, `max_ulp=0` |
| Coefficients effectifs | `c_a=0,9106` | `c_a=0,6004`, `c_d=−0,6027` |
| `__metadata__` des shards | `{'format': 'pt'}` | `{'format': 'pt'}` |
| Smoke `from_pretrained` + GSM8K n=8 | **rc=0**, 8/8, 0,625 | **rc=0**, 8/8, 0,750 |
| `git_commit` du manifeste | `4f1b7d6` | `4f1b7d6` |

**Défaut trouvé et corrigé avant les évaluations** : les shards n'avaient pas de bloc
`__metadata__` ; `transformers.load_state_dict` échouait (`'NoneType' object has no
attribute 'get'`). La vérification **mécanique** passait — seul le **smoke de
chargement** a révélé le défaut. Corrigé en `4f1b7d6` (bloc repris du shard source,
test de non-régression), puis les deux fusions ont été refaites. Les manifestes
portent `git_commit=4f1b7d6`, donc les checkpoints évalués sont bien les checkpoints
corrigés.

## 6. Phase 3 — évaluations G1

### 6.1 Protocole et provenance

| Élément | Valeur vérifiée |
|---|---|
| Modèles | M0 `dream_7b_base`, M1 `dream_7b_instruct`, M2 `dream_transfer_a09`, M3 `dream_merge_l06` |
| Tâches | GSM8K n=300, MATH-175 n=175, IFEval n=8 |
| Décodage | `temperature=0,1`, `top_p=0,9`, `alg=entropy`, `diffusion_steps=max_new_tokens`, `bfloat16`, 0-shot |
| Budgets | 256 / 512 / 1 280 tokens |
| Lot / graines | `batch_size=4` ; `0,1234,1234,1234` |
| Différence entre les 4 modèles | **uniquement** le chemin `pretrained` dans `model_args` |
| Empreintes | `items_sha256` et `prompts_sha256` identiques sur les 4 modèles |
| Code de génération | `fork_git_hash=31f94a6` pour les **12** runs |
| Tâches / prompts système / versions | `task_hashes`, `system_instruction_sha`, `versions`, `n_samples` identiques par tâche |

### 6.2 Scores et intervalles

| Comparaison | Écart | IC 95 % apparié | Critère |
|---|---:|---|---|
| **C1** Transfer − Base, MATH-175 | **+9,71** | **[+2,86 ; +16,57]** | NON ATTEINT (0,29 pt sous +10 ; IC exclut 0) |
| **C2** Transfer − Base, GSM8K | **+14,33** | **[+9,33 ; +19,67]** | **ATTEINT** (papier : +15,2) |
| **C3** Merge − Instruct, MATH-175 | **−2,29** | **[−9,14 ; +4,57]** | NON ÉTABLI (borne basse < −2) |
| **C4** Merge − Instruct, GSM8K | **+1,67** | **[−2,33 ; +5,67]** | NON ÉTABLI (borne basse < −2) |

Scores bruts : GSM8K **68,67 / 84,00 / 83,00 / 85,67** ; MATH-175 **24,00 / 36,57 /
33,71 / 34,29** ; IFEval n=8 **57,14 / 60,71 / 57,14 / 71,88** (M0/M1/M2/M3).

### 6.3 Réserves de périmètre, déjà documentées

- **MATH-175** est un sous-ensemble de `minerva_math` (175 problèmes sur 5 000), pas
  MATH-500 : les cibles de la Table 16 ne sont pas comparables directement.
- **IFEval n=8** est un contrôle de fonctionnement, pas une mesure ; il n'entre dans
  aucun critère.
- Les seuils C1/C2 (10 et 8) sont une **frange basse** des gains de transfert du
  papier (+19,4 sur MATH-500, +15,2 sur GSM8K), pas une reproduction des chiffres.

## 7. Phase 4 — verdict

### 7.1 Application mécanique

`g1_verdict.py` applique les critères amendés : C1/C2 = point estimate **et** borne
basse > 0 ; C3/C4 = borne basse ≥ −2. Le rejeu produit exactement
`eval/dream_a2d/G1_VERDICT.json` : **C2 atteint**, **C1 non atteint**, **C3/C4 non
établis**, donc **G1 NON FRANCHIE** (règle §1 : un seul point en échec ⇒ porte ouverte).

### 7.2 Défaut d'unités corrigé avant publication

Le premier outil comparait des **proportions** (0,1433) à des **seuils en points** (8),
ce qui classait à tort C2 « non atteint » et C3/C4 « non réfutés ». Corrigé en `484b152`,
verrouillé par 7 tests. Le verdict faux n'a jamais été publié : le rapport (`a0dc0eb`) et
le JSON (`027b952`) sont postérieurs au correctif. Le rejeu du code final confirme le
verdict versionné.

### 7.3 Absence de post-hoc

| Date | Événement |
|---|---|
| 2026-10-21 01:14:56 | amendement des critères (`590c37a`) |
| 01:16:12 | ouverture formelle de J7 (`1766f6a`) |
| 02:52:21 | correctif `__metadata__` (`4f1b7d6`) |
| 02:56 / 03:06 | fusions M2 / M3 |
| 03:30 → 10:14 | évaluations GPU (6 h 41) |
| pendant (04:06:05) | alignement du pas 7 (`c849b5d`) — **aucun critère modifié** |
| pendant (08:39:24) | outil de verdict (`0d22e52`) — applique les critères déjà figés |
| 10:19:23 | correctif d'unités (`484b152`) |
| 10:20 / 10:22 | rapport (`a0dc0eb`) et verdict (`027b952`) |

Les critères et les seuils sont figés **avant** la première évaluation. L'outil de
verdict a été écrit pendant la fenêtre d'évaluation, mais il ne choisit aucun seuil :
il applique §4.1. Le correctif d'unités est un correctif de calcul, pas un changement
de critère.

## 8. Conformité A2D

| Élément | Vérification |
|---|---|
| **Équation (1)** `D_b + α·τ_A` | M2 : destinataire `dream_7b_base`, `c_a=0,9`, 339 tenseurs A-only. |
| **Équation (2)** `D_b+(1−λ)τ_D+λτ_A = D_p+λ(τ_A−τ_D)` | M3 : destinataire `dream_7b_instruct`, `c_a=+0,6`, `c_d=−0,6`. |
| **Coefficients Table 6** | α=0,9, λ=0,6 ; non réajustés sur la machine. |
| **τ_A / τ_D** | Qwen-Instruct − Qwen-Base ; Dream-Instruct − Dream-Base. Calculés à la volée en FP32, **jamais persistés**. |
| **Mapping identité** | 339/339, embeddings et tête inclus, **0 tenseur inchangé**. |
| **Schéma numérique** | FP32 séquentiel, **un seul cast BF16**, `max_ulp=0`. |
| **Provenance** | Hashes d'index et empreintes de contenu des shards sources dans les manifestes ; `git_commit=4f1b7d6`. |
| **Vérification v2** | Coefficients signés par famille, ULP, `ok=true`. |

**Écarts de coefficients effectifs.** M2 : `0,9106` (demandé 0,9) ; M3 : `attn`
`−0,6739`, `norms` `0,7392/−0,9896` (demandé 0,6/−0,6). Ce sont des **coefficients de
régression** globaux, dominés par les familles à faibles deltas où l'arrondi BF16 est
du même ordre que le delta. Le contrôle fort est `max_ulp=0` : chaque élément est
exactement l'arrondi BF16 de l'expression FP32 demandée. Ce n'est pas un changement de
formule.

**Protocole d'évaluation.** Température, `top_p`, `alg`, `diffusion_steps =
max_new_tokens`, dtype, 0-shot et extracteurs sont conformes à l'App. A.1/A.3 du
papier. Les choix locaux (`batch_size=4`, graines, MATH-175, IFEval n=8) sont
documentés et **tenus constants** sur les quatre modèles ; ils ne sont pas des
déviations silencieuses.

**Conclusion** : aucune déviation silencieuse détectée. Les deux défauts trouvés
(`__metadata__`, unités) ont été détectés, documentés et corrigés avant la
publication : le premier avant les évaluations (`4f1b7d6`), le second avant le verdict
(`484b152`) ; ils ne sont pas silencieux.

## 9. Écarts de forme et de traçabilité

| # | Constat | Gravité | Suite |
|---|---|---|---|
| **1** | `CHECKLIST.md` et `KNOWN_ISSUES.md` disaient « la non-régression de −2 pts ne peut pas être exclue », ce qui **inverse le sens** (C3/C4 sont NON ÉTABLIS). | moyenne (editorial) | corrigé dans `28e8410` : « la dégradation de 2 points ne peut pas être exclue ». |
| **2** | `J7.md` §10 affirmait « aucune écriture hors du dépôt », alors que les `pytest` de J7 ont utilisé le `TMPDIR` par défaut (`/tmp/pytest-of-sjteb`, 13 095 569 o). | faible (processus) | corrigé dans `28e8410` : violation déclarée, renvoi à `OUT_OF_WORKSPACE.md` §1.3. |
| **3** | La docstring de `seed_noise_floor.py` affirmait qu'une autre graine « ne redonne pas les mêmes réponses », contredit par la mesure (300/300 identiques). | faible | corrigé dans `28e8410` : outil de mesure, résultat J7 en référence. |
| **4** | `REVIEW_N2_J6QUATER.md` est une **reconstitution** ; le verbatim N2 n'a pas été fourni (encadré « Provenance », §1 vide). | faible (entrée manquante) | renvoi ajouté dans `J7.md` §2.4 (`28e8410`) ; coller le verbatim en §1 s'il est fourni. |
| **5** | La sortie `verify_merge` de M3 n'est pas archivée ; le label « non établie » diffère du mot « régression » du protocole pour `lo < −2`. | faible (traçabilité/sémantique) | documenté ici ; le manifeste v2 suffit. Aligner le vocabulaire lors d'une prochaine révision du protocole, **pas dans J7** (ce serait post hoc). |

**Note de précision (pas un écart)** : la famille `norms` a des deltas si faibles que
l'arrondi BF16 domine le coefficient de régression effectif. `max_ulp=0` prouve que
l'arithmétique est correcte ; c'est une limite de représentabilité BF16, déjà signalée
dans J7 §6.2.

## 10. Preuves chiffrées

| Preuve | Valeur |
|---|---|
| M2 : couverture / ULP / `c_a` | 339/339, `max_ulp=0`, `0,9106` |
| M3 : couverture / ULP / `c_a` / `c_d` | 339/339, `max_ulp=0`, `0,6004`, `−0,6027` |
| Bruit de décodage | 300/300 réponses identiques, **+0,00 pt** [0,00 ; 0,00] |
| C1 / C2 | +9,71 [2,86 ; 16,57] / **+14,33 [9,33 ; 19,67]** |
| C3 / C4 | −2,29 [−9,14 ; 4,57] / +1,67 [−2,33 ; 5,67] |
| Empreintes | 12 runs, `fork_git_hash=31f94a6`, items/prompts identiques |
| Tests | **73 passed** |
| Git | HEAD audité `027b952` ; corrections `28e8410` ; **aucun push** |

## 11. Conclusion et recommandations

1. **J7 est conforme aux directives** et **la méthode A2D n'a pas été altérée
   silencieusement**. Le verdict **G1 NON FRANCHIE** est correct, reproductible et
   dérivé de seuils pré-enregistrés.
2. **C1 manque de 0,29 pt** : c'est sous la résolution de la mesure (±7 pts sur 175
   items). Le protocole le dit déjà (§8.4 de J7). **Ne pas modifier le seuil** ; si G1
   doit être rouverte, il faut un **nouveau protocole pré-enregistré** (n plus grand
   ou formulation TOST).
3. **C3/C4** : le merge est statistiquement indiscernable de son témoin, ce qui est
   cohérent avec les gains de merge modestes du papier. Le label « non établie » est
   plus prudent que « régression » ; l'aligner un jour, sans toucher au verdict.
4. **Les corrections de `28e8410` sont documentaires** : aucun checkpoint, aucun
   résultat, aucun seuil n'a changé.
5. **Ne pas relancer M3** : le manifeste v2 suffit ; une vérification indépendante
   supplémentaire coûterait ~20 min de CPU/disque sans changer la conclusion.
6. **Aucun push** : les commits restent locaux, conformément à la règle 11 de
   `AGENTS.md`.

## Annexe A — Commandes de reproduction (CPU/disque uniquement)

```bash
# Verdict (nécessite le venv du harnais pour math_verify)
harness/dream/.venv/bin/python scripts/g1_verdict.py --root eval/dream_a2d

# Vérification indépendante de M2
python scripts/verify_merge.py --merged checkpoints/dream_transfer_a09 \
    --reference checkpoints/dream_7b_base --mode transfer \
    --tau-a-post checkpoints/qwen25_7b_it --tau-a-base checkpoints/qwen25_7b \
    --alpha 0.9 --mapping identity

# Tests, avec TMP redirigé dans le dépôt
TMPDIR="$PWD/_artefact/tmp" TEMP="$PWD/_artefact/tmp" TMP="$PWD/_artefact/tmp" \
    python -m pytest scripts/tests -q
```

> **Note opérationnelle (2026-10-21, agent J7 — post-audit, sans modification des
> constats de l'audit)** : la commande « Tests » ci-dessus **échoue sur cette machine**.
> Le dépôt est sur **DrvFs** (`/mnt/d`), qui ne supporte pas `O_TMPFILE` :
> `tempfile.TemporaryFile()` retombe sur un fichier nommé et la capture pytest échoue en
> `FileNotFoundError` (`_pytest/capture.py:594`, au `truncate()` de fin de session) —
> **après** l'exécution des tests, donc sans ligne « passed », et avec un code de sortie
> non nul. Recette **vérifiée** qui fonctionne dans le dépôt :
> `TMPDIR="$PWD/_artefact/tmp" TEMP="$PWD/_artefact/tmp" TMP="$PWD/_artefact/tmp" \
> python3 -m pytest scripts/tests -q --capture=sys` → **73 passed** (le nombre rapporté
> par l'audit est donc confirmé) ; `-s` convient aussi. `--capture=fd` échoue comme la
> capture par défaut sur DrvFs ; le `TMPDIR` par défaut (`/tmp`, ext4) fonctionne mais
> écrit hors dépôt.

## Annexe B — Limites de l'audit

- Pas de ré-exécution GPU : l'audit vérifie la **traçabilité** et la **statistique**,
  pas la reproductibilité matérielle des générations.
- M3 non revérifié indépendamment ; le manifeste v2 est la preuve.
- Verbatim N2 indisponible : la revue est une reconstitution honnête, pas le texte
  original.
- IFEval n=8 et MATH-175 sont des périmètres réduits, documentés ; ils ne permettent
  pas de conclure sur les jeux complets.
- L'audit ne constitue pas une nouvelle mesure de qualité : il ne remplace ni J7 ni
  une évaluation indépendante.
