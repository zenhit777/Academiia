# Budget RTX 5090 — chiffrage d'une voie d'entraînement d-OPD locale

**Date** : 2026-10-23 · **Mission** : J14 · **Machine** : 1 × NVIDIA GeForce
RTX 5090, 32 607 MiB (31,82 GiB) selon `nvidia-smi` (32 579,4 MiB selon les
propriétés CUDA de PyTorch), WSL2 · **Cible de référence** : Qwen3-0.6B

> **Portée.** Cette note chiffre un **budget**, pas un protocole d'entraînement.
> Aucun entraînement n'a été exécuté : toutes les mesures ci-dessous sont des
> **forwards sans gradient** (`torch.no_grad`), sans optimizer, sans
> modification de poids. Les projections de la §4 sont des **extrapolations**
> explicites, signalées comme telles ; elles ne valent pas protocole
> pré-enregistré.
>
> **Source des mesures** : harnais versionné
> `modules/dopd/harness/bench_budget_rtx5090.py`, résultat versionné
> `modules/dopd/eval/budget_rtx5090/bench_rtx5090.json` (graine `20261023`,
> commit producteur `a59afbf`, empreinte SHA-256 de la source dans le JSON :
> `b420b1a1…41d56`, **vérifiée identique** au fichier de ce commit).
>
> ```bash
> HF_HOME="$PWD/_artefact/hf-cache" python modules/dopd/harness/bench_budget_rtx5090.py \
>   --model Qwen/Qwen3-0.6B --seed 20261023 --local-files-only \
>   --output modules/dopd/eval/budget_rtx5090/bench_rtx5090.json
> ```
>
> Ces mesures sont des **débits**, pas des résultats scientifiques : la graine
> fixe la re-exécution, elle ne transforme pas la mesure en expérience.

---

## 1. Mesures (forward seul, Qwen3-0.6B, BF16, RTX 5090)

### 1.1 Empreinte et préremplissage

| Grandeur | Valeur |
|---|---|
| Paramètres | 596 049 920 |
| Poids BF16 en mémoire | **1 136,9 MiB** |
| Couches / `hidden_size` / têtes KV / `head_dim` | 28 / 1024 / 8 / 128 |
| Vocabulaire | 151 936 |
| Environnement | PyTorch 2.12.0+cu130, Python 3.12.3, CUDA sur RTX 5090 |

| Préremplissage (batch 1) | tok/s | Pic alloué |
|---|---:|---:|
| 512 tokens | 21 994 | 1 240 MiB |
| 2 048 tokens | 82 207 | 1 446 MiB |
| 4 096 tokens | 81 547 | 1 723 MiB |

### 1.2 Décodage avec cache KV

| Contexte | Lot 1 | Lot 8 | Lot 16 | Lot 32 | Lot 64 |
|---|---:|---:|---:|---:|---:|
| 1 024 tokens | 34,4 | 350,9 | 716,5 | **1 399,5** | **2 358,1** |
| 4 096 tokens | 45,9 | 353,0 | 632,2 | 779,7 | 54,8 *(effondrement)* |

tok/s. Le lot 64 à 4 096 de contexte s'effondre (54,8 tok/s) : le cache KV
(28,7 GiB pour 64 séquences) plus les activations **dépassent le budget**
(36,5 GiB alloués au pic, OOM au-delà).

### 1.3 Cache KV (théorique = mesuré)

| Contexte | Analytique par séquence | Lot 16 | Lot 32 | Lot 64 |
|---|---:|---:|---:|---:|
| 1 024 | 112 MiB | 1 792 MiB | 3 584 MiB | 7 168 MiB |
| 4 096 | 448 MiB | 7 168 MiB | 14 336 MiB | 28 672 MiB |

La mesure (mémoire allouée après préremplissage moins l'empreinte des poids)
donne 1 824,1 / 3 616,3 / 7 200,5 MiB pour les lots 16 / 32 / 64 à 1 024 de
contexte : l'écart à la formule analytique est un **overhead constant de
32 MiB** (indépendant de la taille du lot), pas un facteur par séquence. Le
cache par séquence vaut donc bien 112 MiB à 1 024 et 448 MiB à 4 096.

### 1.4 Coût du scoring professeur (lots de branches, contexte rejoué)

Un forward de `k` branches en parallèle, contexte complet rejoué
(identique à la méthode de `_suffix_loglikelihoods` du diagnostic) :

| Contexte | Branches | tok/s | s / forward | Pic alloué |
|---|---:|---:|---:|---:|
| 512 | 16 | 108 927 | **0,150** | 2 273 MiB |
| 1 024 | 16 | 98 755 | **0,332** | 3 378 MiB |
| 2 048 | 16 | 89 251 | 0,734 | 5 586 MiB |
| 4 096 | 16 | 82 521 | 1,588 | 10 003 MiB |
| 2 048 | 64 | 91 406 | 2,868 | 18 835 MiB |
| 4 096 | 64 | 3 884 | 135,0 | 36 501 MiB |
| 4 096 | 128 | — | **OOM** | — |

Le scoring est **linéaire en la longueur de contexte** et sature à ~90 000
tok/s jusqu'à 2 048 tokens ; au-delà de 4 096 tokens × 64 branches, la mémoire
devient le facteur limitant (36,5 GiB alloués, effondrement du débit).

Cette campagne mesure le **rejeu intégral** (sans cache de préfixe). Le
coût du chemin **avec** cache, qui est le mode d'implémentation du papier,
est mesuré séparément en §1.5 sur le job de scoring réel.

*Note d'artefact* : le pic **réservé** CUDA atteint 55 762 MiB au cours de la
campagne (les expériences OOM gonfient le pool de l'allocateur). Seule la
mémoire **allouée** (36 500,9 MiB au pic) est une contrainte physique ;
le réservé est un artefact de l'allocateur.

### 1.5 Coût du scoring avec cache de préfixe (mesuré, géométrie d-OPD)

C'est le poste dominant du budget local, et c'est le seul qui pouvait rester
une inférence. La campagne le mesure sur le **job réel** d-OPD (eq. 11–14) :

- le **préfixe causal** `(q, z, Â_i)` est partagé ;
- le **candidat** `v` diffère d'une branche à l'autre (token à la position
  masquée `i`) ;
- le **suffixe complété** `Ĉ_i` est partagé (rollout complété) ;
- le prior causal `r_i(v)` est le logit du dernier token du préfixe, le score
  futur `s_i(v)` est la somme des log-probabilités du suffixe sous `v`.

Deux modes, **même job** : avec cache (préremplissage une fois, cache
dupliqué, puis chaque branche ne calcule que `[v, Ĉ_i]`) et sans cache
(contexte complet rejoué par branche, comme dans `_suffix_loglikelihoods` du
diagnostic). Le coût du cache est **complet** : préremplissage + duplication
du cache sur les branches + passe de branche.

| `k` | Contexte | Suffixe | Prérempl. | Duplication | Branches | **Total cache** | Sans cache | Gain | Pic (cache / sans) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 256 | 3 | 29,6 ms | 0,7 ms | 30,8 ms | **61,1 ms** | 54,8 ms | 0,90× | 1 700 / 1 740 MiB |
| 16 | 512 | 3 | 33,1 ms | 0,8 ms | 25,9 ms | **59,8 ms** | 119,6 ms | 2,00× | 2 147 / 2 282 MiB |
| **16** | **1 024** | **3** | 23,0 ms | 1,4 ms | 26,8 ms | **51,2 ms** | 167,9 ms | **3,28×** | 3 098 / 3 386 MiB |
| 16 | 1 024 | 256 | 23,3 ms | 1,3 ms | 74,8 ms | 99,5 ms | 221,1 ms | 2,22× | 9 352 / 9 392 MiB |
| 16 | 4 096 | 3 | 50,4 ms | 5,0 ms | 61,9 ms | 117,3 ms | 815,4 ms | 6,95× | 8 859 / 10 014 MiB |
| 33 | 512 | 3 | 33,0 ms | 1,4 ms | 27,0 ms | **61,3 ms** | 171,8 ms | 2,80× | 3 228 / 3 508 MiB |
| **33** | **1 024** | **3** | 23,6 ms | 2,6 ms | 39,4 ms | **65,6 ms** | 368,9 ms | **5,62×** | 5 147 / 5 742 MiB |
| 33 | 1 024 | 256 | 23,2 ms | 2,6 ms | 145,2 ms | 171,0 ms | 475,8 ms | 2,78× | 18 068 / 18 154 MiB |
| 33 | 4 096 | 3 | 58,8 ms | 10,0 ms | 405,5 ms | 474,3 ms | 50 230,8 ms | 105,9× | 17 029 / 19 409 MiB |

**Lecture.** Trois enseignements, tous mesurés :

1. **Le cache ne paie qu'à partir d'un contexte moyen.** À 256 tokens il est
   **plus lent** que le rejeu (0,90×) : la duplication du cache et la passe de
   branche coûtent plus que le contexte court à recalculer. À 1 024 de
   contexte, il divise le coût par **3,3×** (`k = 16`) à **5,6×** (`k = 33`),
   pour une **mémoire équivalente** (3 098 contre 3 386 MiB : le cache ne
   coûte pas de mémoire, il coûte du temps de recalcul).
2. **Le coût est dominé par la passe de branche**, pas par le préremplissage :
   à `k = 16`, contexte 1 024, la passe de branche vaut 26,8 ms sur 51,2 ms
   de total. Le préremplissage (23,0 ms) et la duplication (1,4 ms) sont
   minoritaires.
3. **Le rejeu devient vite prohibitif.** À 4 096 de contexte avec 33
   branches, le rejeu du contexte complet prend **50,2 s** par job, contre
   **474 ms** avec cache (106×). C'est le mécanisme qui rend le rollout de
   4 000 tokens du papier hors de portée d'une carte unique.

**Contrôle de cohérence.** Le même job est rejoué en **FP32** : l'écart entre
les deux modes tombe à `3,1e-5` (score de suffixe) et `9e-6` (prior), contre
`0,04` à `2,1` en BF16. L'écart BF16 est donc un **artefact de précision**,
pas une erreur de calcul du chemin cache. Ce contrôle est rejoué par le
harnais et versionné dans le JSON
(`prefix_cache_scoring_fp32_control`).

**Limite de la mesure.** L'App. B du papier construit le cache **une fois par
trajectoire** (sur la séquence prompt-réponse complète) puis le **tronque**
à la position candidate ; le même cache est réutilisé pour toutes les
positions éligibles de l'exemple. La mesure ci-dessus **pré-remplit le
préfixe pour chaque job** : elle est donc **conservatrice**, et le gain réel
en entraînement est au moins celui mesuré.

---

## 2. Ce que le papier suppose

| Paramètre | Papier | Local |
|---|---|---|
| GPU | **8 × H100 80 Go** | **1 × RTX 5090 32 Go** |
| Passage d'entraînement | 61 816 prompts, **483 itérations**, ≤ 128 prompts/iteration, 1 rollout/prompt | — |
| Longueur de réponse max | 4 000 tokens (rampe cosinus 100 → 4 000 sur 100 itérations) | — |
| Bloc / pas de débruitage | `N = 4` (aussi 8, 16), 4 pas | — |
| Candidats | `k = 16` | — |
| Optimiseur | AdamW, ZeRO-3, états d'optimiseur sur CPU | — |
| Temps mural jusqu'à la performance d'OPDLM, échelle 0,6B | **13,15 h** (OPDLM : 18,39 h), soit ~105 h-GPU | — |

Le budget du papier est donc **~105 h-GPU H100 pour l'échelle 0,6B**, et non
13,15 h de carte unique.

---

## 3. Choix de configuration locale

### 3.1 Taille d'élève réaliste

| Élève | Poids BF16 | Gradients BF16 | AdamW FP32 (m, v) | Total optimiser+poids+grads | Verdict sur 32 GiB |
|---|---:|---:|---:|---:|---|
| **0,6B** | 1,1 | 1,1 | 4,7 | **6,9 GiB** | **faisable**, large marge |
| 1,7B | 3,3 | 3,3 | 13,0 | 19,6 GiB | faisable seulement avec `gradient_checkpointing` et caches courts (< 2 048) |
| 4B | 7,6 | 7,6 | 30,5 | 45,7 GiB | **impossible** sans déport CPU des états d'optimiseur |
| 8B (échelle du papier) | 15,3 | 15,3 | 61,0 | 91,6 GiB | **impossible** — c'est le cas du papier (8 × H100 80 Go, ZeRO-3) |

**Élève retenu : 0,6B.** Un élève 1,7B reste envisageable mais serre la
mémoire ; l'échelle 4B–8B est hors de portée d'une seule RTX 5090, quelle que
soit l'optimisation des états d'optimiseur.

### 3.2 Professeur

Le professeur AR doit coexister avec l'élève **entraîné** (poids + gradients +
états d'optimiseur) et les caches KV.

| Professeur | Poids BF16 | Total avec élève 0,6B entraînable | Verdict |
|---|---:|---:|---|
| **Qwen3-0.6B** | 1,1 | ~8 GiB | **retenu** |
| Qwen3-1.7B | 3,3 | ~10,2 GiB | possible, caches à 1 024 |
| Qwen3-4B | 7,6 | ~14,5 GiB | possible à contexte court, sans marge |
| Qwen3-8B | 15,3 | ~22,2 GiB | **impossible** : plus aucun budget pour les caches ni les activations |

### 3.3 Longueur de rollout

Le coût du scoring professeur croît **linéairement avec le contexte**
(§1.4 et §1.5) : à `k = 16`, le job passe de 51,2 ms (contexte 1 024) à
117,3 ms (contexte 4 096), soit **2,3×** pour 4× le contexte — et 474,3 ms
avec 33 branches.

| Longueur max | Conséquence |
|---|---|
| 4 000 (papier) | contexte 4 096 : 117 ms par job avec cache, 815 ms sans ; **incompatible** avec un nombre de prompts significatif sur une carte |
| **512** | contexte ≤ 1 024 ⇒ 51,2 ms par job (`k = 16`) ; cache KV 144 MiB/séquence |
| 256 | contexte ≤ 768 ; marge mémoire maximale |

**Longueur retenue : 256 à 512 nouveaux tokens**, avec plafond de contexte
1 024. C'est la principale réduction d'échelle, et elle est **dictée par la
mémoire et le scoring**, pas par un choix de commodité.

### 3.4 Itérations et taille de lot

| Contrainte | Valeur | Source |
|---|---|---|
| Débits de décodage soutenus | 1 399 tok/s (lot 32, 1 024) ; 2 358 tok/s (lot 64, 1 024) | §1.2 mesuré |
| Cache KV par séquence | 144 MiB (1 024) / 480 MiB (4 096) | §1.3 mesuré |
| Marge mémoire (élève + professeur entraînés) | ~24 GiB | §3.1, §3.2 |

| Lot de rollouts | Contexte 1 024 | Cache KV | Verdict |
|---:|---:|---:|---|
| 16 | 1 024 | 2,3 GiB | confortable |
| **32** | 1 024 | 4,5 GiB | **lot retenu** |
| 64 | 1 024 | 9,0 GiB | possible, gain marginal (+69 % de débit) |
| 32 | 4 096 | 18,0 GiB | possible, mais scoring 2,3× plus cher par position |

**Lot retenu : 32 rollouts simultanés**, contexte 1 024. Un lot de 64 ne
rapporte que 69 % de débit en plus pour le doublement du cache ; il sera
préféré si la pression mémoire est faible.

---

## 4. Projection d'un run significatif

**Hypothèses** (à confirmer par un protocole pré-enregistré, J15) :

* élève = professeur = Qwen3-0.6B ; `N = 4`, 4 pas de débruitage, `k = 16` ;
* rollout : 1 par prompt, 256 tokens générés, contexte ≤ 1 024 ;
* **un état de débruitage par bloc** (App. C.2 du papier), soit
  64 blocs scorés par rollout ;
* ~2,2 positions masquées par bloc (`N = 4`, rétention 0,45 — App. D.4 du
  papier, confirmant le chiffre du diagnostic) ;
* un **seul** rollout par prompt et par itération, comme le papier
  (128 prompts × 483 itérations = 61 824 ≈ 61 816 prompts) ;
* le scoring n'a lieu que sur les positions **éligibles** : le papier applique
  une porte de vérificateur et ne corrige que les positions ayant un futur
  visible ; on retient ici **100 %** des positions, hypothèse **majorante**.

Le nombre de positions masquées à scorer pour 8 000 prompts est donc
`8 000 × 64 × 2,2 ≈ 1,13 M`.

| Poste | Coût calculé | Base |
|---|---|---|
| Génération du rollout (élève) | 2,05 M tokens ÷ 1 363 tok/s ≈ **0,4 h** | mesuré §1.2 |
| Scoring professeur, **avec cache de préfixe** | 1,13 M × 51,2 ms ≈ **16,0 h** | **mesuré §1.5** (`k=16`, contexte 1 024, suffixe de bloc) |
| Scoring professeur, **avec cache, `k=33`** | 1,13 M × 65,6 ms ≈ **20,6 h** | **mesuré §1.5** (lot maximal App. B) |
| Scoring professeur, **sans cache** (contexte rejoué) | 1,13 M × 167,9 ms ≈ **52,7 h** | **mesuré §1.5** |
| Entraînement élève (forward + backward) | ~4 M passes de tokens ÷ ~7 000 tok/s effectifs ≈ **0,2 h** | **inférence** (forward mesuré, backward ×3 estimé) |
| **Total, avec cache de préfixe** | **≈ 16–21 h** | |
| **Total, sans cache de préfixe** | **≈ 53 h** | |

Le cache de préfixe (App. B du papier : préfixe causal partagé entre les
branches) n'est **pas** une optimisation optionnelle : sans lui, le scoring
professeur passe de 16 h à 53 h et devient le poste dominant à lui seul.

L'écart entre 16 h et 21 h tient au **lot de branches** : `k = 16` candidats
(51,2 ms par job) contre `2k + 1 = 33` branches avec la branche de référence
(65,6 ms). Le papier plafonne à 33 branches par job (App. B), c'est donc la
borne haute qui sera payée si l'union professeur/élève ne déborde pas.

Ces projections **surestiment** le coût réel sur deux points mesurables :

- le papier **réutilise le cache entre positions d'une même trajectoire**
  (App. B) alors que la mesure pré-remplit le préfixe à chaque job ;
- la porte de vérificateur **réduit** le nombre de positions scorées.

Ellipses : le coût de backward de l'élève (≈ 3× le forward) reste une
estimation, et la rétention des états de débruitage, le mask-token et le
découpage en blocs de `StudentPolicy` ne sont pas implémentés.

### 4.1 Extrapolation à l'échelle du papier

| Protocole | Prompts | Rollout | Temps mural projeté (1 × 5090) |
|---|---:|---:|---|
| **Réduit recommandé** | 8 000 | 256 tok | **16–21 h** (avec cache) / ~53 h (sans) |
| Réduit long | 20 000 | 256 tok | 40–52 h (avec cache) |
| 1 epoch papier | 61 816 | 256 tok | 124–160 h (avec cache) |
| 1 epoch papier, rollout 4 000 tok | 61 816 | 4 000 tok | **hors de portée** : voir ci-dessous |

Pour le dernier cas, les deux facteurs se cumulent : le nombre de blocs par
rollout passe de 64 à 1 000 (`4 000 / N = 4`) et le contexte de scoring de
1 024 à 4 096. Sur le seul poste de scoring, la mesure donne **117,3 ms**
par job avec cache (`k = 16`, contexte 4 096) contre **815,4 ms** sans : le
volume de jobs passe à `61 816 × 1 000 × 2,2 ≈ 136 M`, soit **4 430 h**
avec cache et **30 800 h** sans. C'est hors de portée d'une carte unique par
plusieurs ordres de grandeur, ce qui confirme que l'échelle du papier exige
la location ou les 8 × H100.

### 4.2 Comparaison avec 8 × H100

| | Papier (8 × H100 80 Go) | Local (1 × RTX 5090) | Ratio |
|---|---:|---:|---|
| GPU-heures pour atteindre la performance d'OPDLM, échelle 0,6B | 8 × 13,15 ≈ **105 h-GPU** | run réduit 8 000 prompts : **16–21 h-GPU** | ~5–7× moins |
| Prompts traités | 61 816 | 8 000 (13 %) | 7,7× moins |
| Longueur de rollout | 4 000 tok | 256 tok (6 %) | 15,6× moins |
| Élève | jusqu'à 8B | 0,6B | — |
| Mémoire GPU par nœud | 80 Go × 8, NVLink | 32 Go × 1, sans NVLink | — |

Le rapport « GPU-heures / prompt traité » est favorable au local parce que la
longueur de rollout est divisée par ~16 et parce qu'aucun état d'optimiseur
n'est déporté. En revanche, le protocole local **ne permet aucune comparaison
de performance** avec le papier : même le papier n'évalue pas 8 000 prompts
avec des rollouts de 256 tokens.

---

## 5. Verdict

1. **Une voie d'entraînement locale a du sens** à l'échelle **0,6B**, avec un
   protocole **réduit et explicitement déclaré** : 8 000 prompts, rollouts de
   256 tokens, lot 32, contexte 1 024, cache de préfixe partagé. Budget
   chiffré : **16–21 h** sur une RTX 5090, dont l'essentiel (≈ 16–21 h) est
   le scoring professeur. L'incertitude résiduelle est faible (± 25 %) et
   vient du **lot de branches** (`k = 16` mesuré, `2k + 1 = 33` mesuré), pas
   d'une extrapolation non mesurée.
2. **L'échelle 4B–8B est hors de portée** d'une seule RTX 5090, même avec
   déport CPU : c'est un problème de mémoire d'états d'optimiseur, pas de
   débit. Pour l'échelle du papier, il faut **8 × H100 80 Go** (ou une
   location équivalente), soit ~105 h-GPU pour l'échelle 0,6B seule.
3. **Le cache de préfixe partagé est le poste à optimiser**, et son
   comportement est maintenant **mesuré** : il divise le coût par 3,3× à
   1 024 de contexte et par 106× à 4 096, sans coût mémoire significatif. Il
   ne faut l'activer **qu'au-dessus d'un contexte d'environ 512 tokens** :
   en dessous, il est plus lent que le rejeu.
4. **Ce chiffrage n'autorise aucun entraînement.** Il propose à J15 un
   protocole réduit **à pré-enregistrer** (prompts, seed, IC, budget) avant
   toute exécution. Le budget est désormais chiffré sur mesures ; le
   protocole d'entraînement reste à écrire.

---

## 6. Limites de ce chiffrage

* Les mesures sont des **forwards sans gradient** sur le **modèle AR** de
  0,6B. Le rollout d-OPD est un modèle **de diffusion par blocs** : le coût
  par token est approché, pas mesuré.
* Le scoring est mesuré **job par job** : le papier construit le cache une
  fois par trajectoire et le tronque par position (§1.5), donc la mesure est
  **conservatrice**. En revanche, elle suppose un préremplissage du préfixe
  par job, ce qui ne remplace pas l'implémentation réelle du cache tronqué.
* Le coût de backward de l'élève est **estimé** (≈ 3× le forward) ; la
  rétention des états de débruitage, le mask-token et le découpage en blocs
  de `StudentPolicy` ne sont pas encore implémentés.
* La projection suppose que **toutes** les positions masquées sont scorées
  (100 %) : le papier applique une porte de vérificateur et ne corrige que
  les positions ayant un futur visible. C'est une hypothèse **majorante**.
* La notion d'« epoch » du papier est reconstituée (1 rollout/prompt) à
  partir de 128 × 483 ≈ 61 816 ; le document n'explicite pas la
  correspondance.
* Les projections sont des ordres de grandeur destinés à **décider**
  (local 0,6B / location 8B), pas à fixer un protocole.
* Les débits mesurés varient de quelques pour cent d'une exécution à l'autre
  (la campagne versionnée est semée ; les appels GPU ne sont pas
  déterministes au niveau de l'ordre des opérations). L'ordre de grandeur des
  conclusions est inchangé.
