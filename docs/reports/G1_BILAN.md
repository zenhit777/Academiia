# Bilan de la porte G1 — clôture scientifique (J8, phase A)

**Date** : 2026-10-21 · **Statut** : **G1 NON FRANCHIE — porte laissée OUVERTE** ·
**Périmètre** : consolidation de J7 + analyse de troncature MATH (nouvelle mesure J8) ·
**HEAD de départ** : `bbcf626` · **Rapport J7** : `docs/reports/J7.md` (non réécrit).

> Ce document **ne réécrit pas** J7 et **ne rejoue aucun seuil**. Il rassemble les preuves
> déjà versionnées, ajoute une mesure qui manquait (la troncature des générations MATH au
> plafond de 512 tokens) et formule la conclusion scientifique de la porte. Les critères
> pré-enregistrés (`docs/design/G1_PROTOCOL.md` §4.1, amendement M1, HEAD `16a0232`) sont
> appliqués **tels quels**.

## 1. Verdict

> ## **G1 NON FRANCHIE.**
>
> Trois des quatre critères d'écart ne sont pas atteints (C1 pour 0,29 point, C3 et C4 sur
> la borne basse). Le point 3 de la règle de franchissement (§1 du protocole) échoue ; les
> points 1, 2 et 4 sont satisfaits. La porte reste **ouverte**, aucun seuil n'est modifié.

| Critère | Comparaison | Mesure | Écart apparié | IC bootstrap 95 % | Verdict |
|---|---|---|---:|---|---|
| **C1** | M2 − M0, MATH-175 | 24,00 % → 33,71 % | **+9,71** | [+2,86 ; +16,57] | **NON ATTEINT** — 0,29 pt sous le seuil de +10 ; IC excluant 0 |
| **C2** | M2 − M0, GSM8K | 68,67 % → 83,00 % | **+14,33** | [+9,33 ; +19,67] | **ATTEINT** — ≥ +8 et borne basse > 0 |
| **C3** | M3 − M1, MATH-175 | 36,57 % → 34,29 % | **−2,29** | [−9,14 ; +4,57] | **non-régression NON ÉTABLIE** — borne basse < −2 |
| **C4** | M3 − M1, GSM8K | 84,00 % → 85,67 % | **+1,67** | [−2,33 ; +5,67] | **non-régression NON ÉTABLIE** — borne basse marginalement < −2 |

Modèles : **M0** `dream_7b_base` (D_b), **M1** `dream_7b_instruct` (D_p), **M2**
`dream_transfer_a09` (transfert, α = 0,9), **M3** `dream_merge_l06` (merge, λ = 0,6).
Source des chiffres : `eval/dream_a2d/G1_VERDICT.json`, recalculé par `scripts/g1_verdict.py`
à partir des échantillons versionnés.

**Les trois acquis positifs, à ne pas confondre avec le verdict :**

1. **C2 est atteint et l'effet est robuste** : +14,33 points sur GSM8K, IC [+9,33 ; +19,67],
   pour un seuil de +8. Le papier rapporte **+15,2** pour la même paire
   (Qwen2.5-7B → Dream-7B, Fig. 2a) : l'ordre de grandeur est reproduit.
2. **La mécanique A2D est validée sur une famille réelle** : deux fusions produites,
   vérifiées (`post_write_verification` v2, `max_ulp = 0`, `verify_merge` VERDICT OK),
   chargeables (`from_pretrained`), évaluées sur les **mêmes items** (empreintes identiques
   sur les 4 modèles) avec un **plancher de bruit de décodage nul** au lot 4.
3. **Aucun effondrement** : IFEval n=8, M2 = 57,14 % (identique à M0), M3 = 71,88 %
   (au-dessus de M1). Contrôle non bloquant, mais sans signal de dégradation.

**Ce que le verdict ne dit pas** : il ne dit pas que le transfert est sans effet — l'IC de C1
exclut 0 — il dit que l'effet mesuré sur MATH-175 à 512 tokens n'atteint pas le seuil
pré-enregistré de +10 points. La section 4 montre que ce résultat est **dilué par le plafond
de génération**, sans que cela autorise à changer le verdict.

## 2. Ce que J7 a établi (rappel, sans réécriture)

| Point de la règle (§1 du protocole) | État | Preuve |
|---|---|---|
| 1. mêmes items, même protocole, identité prouvée | **✔** | `eval/dream_a2d/ITEM_FINGERPRINTS.json` : `items_sha256` / `prompts_sha256` identiques sur les 4 modèles ; lot 4, graines `0,1234,1234,1234` |
| 2. chaque fusion a passé la vérification v2 | **✔** | manifestes `checkpoints/dream_transfer_a09/merge_config.json` et `checkpoints/dream_merge_l06/merge_config.json` : `ok = true`, `version = 2`, `max_ulp = 0.0` ; `verify_merge` VERDICT OK par famille |
| 3. critères d'écart atteints | **✘** | C1, C3, C4 non atteints (§1) |
| 4. IC appariés rapportés | **✔** | `eval/dream_a2d/G1_VERDICT.json` (10 000 ré-échantillonnages, graine 0, percentile 2,5 %) |

Détails, y compris le défaut de chargeabilité corrigé avant évaluation (bloc `__metadata__`
absent des shards fusionnés, commit `4f1b7d6`) : `docs/reports/J7.md` §5–§8.

## 3. Mesure nouvelle — troncature des générations MATH à 512 tokens

### 3.1 Définition opérationnelle

Une génération est **tronquée sans réponse** si les deux conditions sont réunies :

* la réponse compte **≥ 512 tokens** (le plafond `max_new_tokens` du protocole MATH) ;
* elle **ne contient aucun marqueur de réponse finale** (`\boxed{...}` ou « answer is »).

Le second critère est indispensable : une génération qui atteint le plafond **après** avoir
écrit sa réponse reste notable. Mesure : `scripts/measure_math_truncation.py`
(tokenizer `checkpoints/dream_7b_instruct`), sortie `eval/dream_a2d/TRUNCATION_512.json`.

### 3.2 Résultats

| Modèle | Générations au plafond (≥ 512 tok) | … dont réponse présente | **tronquées sans réponse** | part de MATH-175 |
|---|---:|---:|---:|---:|
| **M0** `dream_7b_base` | 50 | 0 | **50** | 28,6 % |
| **M1** `dream_7b_instruct` | 4 | 1 | **3** | 1,7 % |
| **M2** `dream_transfer_a09` | **82** | **15** | **67** | **38,3 %** |
| **M3** `dream_merge_l06` | 57 | 2 | **55** | 31,4 % |

Répartition par sujet pour M2 (le modèle le plus touché) :

| Sujet (25 items) | tronquées sans réponse |
|---|---:|
| algebra | 5 |
| counting_and_prob | 7 |
| geometry | 10 |
| intermediate_algebra | 16 |
| num_theory | 9 |
| prealgebra | 5 |
| precalc | 15 |

**Lecture.** Le plafond de 512 tokens n'est pas neutre : M1 termine presque toujours
(3 items sans réponse), M0 et M3 sont touchés sur ~30 % des items, **M2 sur 38 %**. Le
transfert rend le modèle **plus verbeux** — cohérent avec un raisonnement plus long — et le
protocole à 512 tokens le **pénalise** précisément là où il a le plus à dire. Cette mesure
explique un résultat autrement contre-intuitif : M2, qui gagne +14,33 points sur GSM8K
(256 tokens), ne gagne que +9,71 sur MATH-175 (512 tokens).

## 4. Où vit l'effet C1 — analyse appariée sur les 67 items tronqués de M2

Le sous-ensemble des 67 items tronqués sans réponse **de M2** (le modèle du critère C1) est
figé dans `harness/dream/tasks/math_trunc_a2d.jsonl` (construction reproductible :
`scripts/make_math_trunc_subset.py`, manifeste avec SHA-256 des fichiers sources). Les
réponses à 512 tokens sont **réutilisées telles quelles** — aucune ré-exécution.

| Modèle | Exactitude sur les 67 items | Exactitude sur les 108 autres |
|---|---:|---:|
| M0 `dream_7b_base` | 10,45 % (7/67) | 32,41 % (35/108) |
| M1 `dream_7b_instruct` | 19,40 % (13/67) | 47,22 % (51/108) |
| **M2** `dream_transfer_a09` | **7,46 % (5/67)** | **50,00 % (54/108)** |
| M3 `dream_merge_l06` | 16,42 % (11/67) | 45,37 % (49/108) |

Comparaisons appariées (bootstrap percentile, 20 000 ré-échantillonnages, graine 20261021) :

| Comparaison | Sur les 67 items tronqués | Sur les 108 autres |
|---|---|---|
| **M2 − M0** | **−2,99** [−11,94 ; +5,97] · 4/6 · p = 0,75 | **+17,59** [+9,26 ; +26,85] · 23/4 · **p = 0,0003** |
| **M2 − M1** | **−11,94** [−20,90 ; −2,99] · 1/9 · p = 0,021 | +2,78 [−5,56 ; +11,11] · 12/9 · p = 0,66 |

**Lecture — et sa limite.**

* Sur les items où M2 **peut terminer** (108/175), le transfert vaut **+17,59 points** contre
  la base, avec un IC qui exclut largement 0 : l'effet est **plus fort** que le +9,71 mesuré
  sur l'ensemble.
* Sur les 67 items qu'il **ne peut pas terminer**, M2 n'est pas meilleur que M0 (−2,99, IC
  contenant 0) et est **significativement moins bon que M1** (−11,94, IC excluant 0) : la
  verbosité supplémentaire y consomme le budget sans produire de réponse.
* Autrement dit, le +9,71 de C1 est la **moyenne diluée** d'un effet fort (+17,6) sur 62 %
  des items et d'un effet nul ou négatif sur les 38 % restants. **Cela n'annule pas le
  verdict** : le protocole pré-enregistré mesure MATH-175 à 512 tokens, et c'est cette
  mesure qui vaut pour C1. Mais cela identifie la cause principale de l'échec de 0,29 point :
  **le plafond de génération, pas l'absence d'effet du transfert**.

**Biais de sélection — analyse à qualifier d'EXPLORATOIRE.** La partition 67/108 n'est
**pas pré-enregistrée** : elle est **définie par le résultat de M2** (les items que M2 ne
termine pas à 512 tokens). Les items dits « terminés » sont donc, par construction, ceux
sur lesquels M2 produit une réponse finale : ils sont **mécaniquement plus faciles**, et le
contraste entre les deux sous-ensembles surestime nécessairement la part attribuable au
seul plafond de génération. Cette décomposition **décrit** un effet observé ; elle ne
l'identifie pas causalement et ne peut pas être lue comme une estimation non biaisée de
l'effet du transfert sur les items non tronqués. Elle ne modifie **aucun chiffre** du
présent bilan ni le verdict : C1 reste celui de MATH-175 à 512 tokens.

*Repère de résolution* : sur 175 items, **un seul item correct en plus** vaut +0,57 point ;
le seuil de +10 serait franchi par **60/175 au lieu de 59/175** (34,29 % → +10,29 points).
Distinguer 9,71 de 10,00 avec un IC de ±7 points est **sous la résolution de la mesure** —
c'est un problème de dimensionnement de C1, pas une preuve d'absence d'effet (voir §6).

## 5. Sensibilité à 1 024 tokens — diagnostic annulé sur coût mesuré

Pour tester directement l'hypothèse de troncature, un diagnostic a été lancé le 2026-10-21 à
13:48 (heure locale) sur **M0 et M2**, sur le **même sous-ensemble de 67 items**, avec
`max_new_tokens = 1024` et `diffusion_steps = 1024`. Tout le reste du protocole est inchangé
(`temperature=0,1`, `top_p=0,9`, `alg=entropy`, `bfloat16`, 0-shot, `--batch_size 4`,
graines `0,1234,1234,1234`, `math_verify`).

* Commande :
  `TASK=math_trunc MODEL_DIR=checkpoints/dream_7b_base,checkpoints/dream_transfer_a09 OUTDIR=eval/dream_a2d_trunc bash harness/dream/run_evals.sh`
* **Débit réellement mesuré : 313,93 s/item** (4/67 items en 20 min 55 s ; aucun item
  supplémentaire dans les 21 min suivantes), soit **5,84 h par modèle** et **11,7 h** pour
  M0 + M2 — 2,3× le plafond GPU de 5 h de la mission. Le diagnostic a été **arrêté à
  14:32**, avant toute écriture de résultat (lm-eval n'écrit qu'en fin de tâche).
* L'estimation de préparation (**108 s/item**, 4 h pour deux modèles) n'a **pas été
  reproduite** (facteur 2,9) : elle provenait d'un pilote dont le journal n'a pas été conservé,
  probablement lancé sans `diffusion_steps = 1024`. La mesure à 67 items fait foi.
* **M1 et M3 ne sont pas rejoués** (budget) : le diagnostic ne peut donc pas recalculer C1
  sur les 175 items, ni comparer M2 à M1 à 1 024 tokens.
* **Le diagnostic « lot 1 vs lot 4 » n'a pas été lancé** : il aurait doublé le coût GPU pour
  une question déjà documentée par le balayage D3 (`docs/design/EVAL_BUDGET.md` §4.1 : 0 à 5
  réponses identiques sur 16 selon la taille de lot, scores 0,4375 → 0,5000). Le lot 4 reste
  le protocole de tous les runs G1 ; changer de lot aurait rendu les comparaisons invalides.
  L'arbitrage est donc : **un seul diagnostic, celui qui teste l'hypothèse de troncature**.

Ce diagnostic était une **analyse de sensibilité**, pas un second test de G1. **Il n'a produit
aucun résultat** : l'hypothèse de troncature reste l'explication la plus plausible du déficit
de C1, mais elle n'est **pas démontrée** par une contre-mesure à budget supérieur. Le détail
des horodatages, de la projection et des conditions d'arrêt est dans
`docs/reports/G1_BILAN_1024.md` ; rien dans les sections 1 à 4 du présent bilan ne change.

## 6. Le merge M3 reste indiscernable de D_p

C3 (−2,29 [−9,14 ; +4,57]) et C4 (+1,67 [−2,33 ; +5,67]) ne permettent ni d'établir ni
d'exclure une dégradation de 2 points. Ce n'est **pas** un échec de la mécanique : le merge
est vérifié à 0 ULP et chargeable ; c'est une **absence de signal détectable** aux effectifs
de J7 (175 et 300 items).

C'est cohérent avec le papier, qui rapporte pour cette même paire des gains de merge bien
plus faibles que ceux du transfert : **+0,7** sur GSM8K, **+6,6** sur MATH-500, **+9,8** sur
IFEval. Avec des IC de ±7 points sur MATH-175, un effet de +0,7 à +6,6 points est **sous la
résolution** de la mesure. La conclusion correcte est donc « **merge non concluant sur cette
paire et ces effectifs** », pas « merge sans effet » ni « merge dégrade ».

## 7. Ce qui n'est pas conclu

* **Aucune équivalence** avec les cibles de la Table 16 : MATH-500 (500 items) et MATH-175
  (175 items) sont des périmètres différents ; un IC qui contient une cible n'est pas une
  preuve d'équivalence (cf. `docs/design/G1_PROTOCOL.md` §4.3).
* **Aucune généralisation** à DiffusionGemma (G2) ni à Nemotron (G1b) : une seule paire AR
  (Qwen2.5-7B → Dream-7B) est testée.
* **Aucune inférence sur le choix de τ** : seule la paire Qwen2.5-7B est utilisée.
* **Le diagnostic 1 024 tokens ne restaure pas C1** : il porte sur 67 items, deux modèles,
  et ne peut pas produire un écart sur MATH-175.
* **Le plancher de bruit nul** est mesuré sur GSM8K/256 tokens avec M1 ; il n'est pas mesuré
  sur MATH/512 tokens ni à 1 024 tokens. Les IC rapportés restent donc des IC
  d'**échantillonnage**, sans composante de bruit de décodage mesurée pour MATH.

## 8. Recommandations (aucune ne modifie le verdict)

1. **Si l'on veut rouvrir G1 sur C1**, le faire avec un protocole **pré-enregistré** :
   soit augmenter `n` sur MATH (le jeu complet coûte ~48 h par modèle, hors budget), soit
   reformuler C1 en **test d'équivalence (TOST)** sur une marge d'acceptabilité, soit
   **pré-enregistrer un budget de 1 024 tokens pour les quatre modèles** — au prix d'un
   protocole différent de celui du papier (App. A.1 : 512 tokens pour MATH).
2. **Ne pas réajuster α ou λ après ces résultats** : la sélection sur ancre (App. A.5) est
   hors périmètre de G1, et un ajustement post hoc invaliderait la comparaison.
3. **Passer à G2** : la bibliothèque est débloquée côté mode `merge-dg-scaled` (phase B de
   J8). L'évaluation G2 exige toutefois un GPU où le décodeur DiffusionGemma 26B et le juge
   27B tiennent ensemble ; sur 32 Go, cela impose une quantification et un protocole déclaré.
4. **Conserver M2 comme référence de transfert** : c'est le seul modèle qui reproduit, sur
   cette machine, l'ordre de grandeur du gain GSM8K du papier (+14,33 vs +15,2).

## 9. Artefacts et reproduction

| Artefact | Rôle |
|---|---|
| `docs/reports/J7.md` | résultats et verdict d'origine (inchangé) |
| `docs/design/G1_PROTOCOL.md` | critères pré-enregistrés (inchangés) |
| `eval/dream_a2d/G1_VERDICT.json` | écarts, IC, verdicts C1–C4 |
| `eval/dream_a2d/ITEM_FINGERPRINTS.json` | preuve d'identité des items (12 runs) |
| `eval/dream_a2d/TRUNCATION_512.json` | comptes de troncature par modèle et par sujet |
| `harness/dream/tasks/math_trunc_a2d.jsonl` (+ `.manifest.json`) | les 67 items du sous-ensemble |
| `harness/dream/tasks/math_trunc_a2d.yaml` + `utils_math_trunc.py` | tâche lm-eval du diagnostic |
| `scripts/measure_math_truncation.py` | mesure reproductible de la troncature |
| `scripts/make_math_trunc_subset.py` | construction reproductible du sous-ensemble |
| `eval/dream_a2d_trunc/` | résultats du diagnostic 1 024 tokens (M0, M2) |
| `docs/reports/G1_BILAN_1024.md` | addendum : diagnostic 1 024 tokens annulé sur coût mesuré |

```bash
# troncature (CPU, à partir des échantillons versionnés)
python scripts/measure_math_truncation.py \
    --samples-root eval/dream_a2d --tokenizer checkpoints/dream_7b_instruct \
    --budget 512 --output eval/dream_a2d/TRUNCATION_512.json

# sous-ensemble des 67 items tronqués de M2
python scripts/make_math_trunc_subset.py \
    --samples-root eval/dream_a2d --tokenizer checkpoints/dream_7b_instruct \
    --model dream_transfer_a09 --budget 512 \
    --output harness/dream/tasks/math_trunc_a2d.jsonl

# diagnostic 1 024 tokens (GPU, ~4 h)
TASK=math_trunc \
MODEL_DIR=checkpoints/dream_7b_base,checkpoints/dream_transfer_a09 \
OUTDIR=eval/dream_a2d_trunc BATCH=4 bash harness/dream/run_evals.sh
```

## 10. Conclusion

La porte G1 **reste non franchie**, et le restera tant que C1, C3 et C4 n'auront pas été
reformulés ou remesurés dans un protocole pré-enregistré. Les données produites sont
néanmoins **interprétables et cohérentes avec le papier** : transfert net sur GSM8K, transfert
réel mais sous le seuil sur MATH-175, merge indiscernable, aucun effondrement. L'échec de C1
pour 0,29 point est **un problème de résolution et de budget de génération**, documenté ici
(section 4) et non un argument contre la méthode A2D.
