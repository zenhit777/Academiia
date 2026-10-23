# Budget RTX 5090 — chiffrage d'une voie d'entraînement d-OPD locale

**Date** : 2026-10-23 · **Mission** : J14 · **Machine** : 1 × NVIDIA GeForce
RTX 5090, 32 607 MiB (31,82 GiB), WSL2 · **Cible de référence** : Qwen3-0.6B

> **Portée.** Cette note chiffre un **budget**, pas un protocole d'entraînement.
> Aucun entraînement n'a été exécuté : toutes les mesures ci-dessous sont des
> **forwards sans gradient** (`torch.no_grad`), sans optimizer, sans
> modification de poids. Les projections de la §4 sont des **extrapolations**
> explicites, signalées comme telles ; elles ne valent pas protocole
> pré-enregistré.
>
> **Source des mesures** : `_artefact/bench_rollout_forward.json` et
> `_artefact/bench_teacher_branches.json` (artefacts temporaires, non
> versionnés). reproductibles par
> `_artefact/bench_rollout_forward.py` et
> `_artefact/bench_teacher_branches.py`.

---

## 1. Mesures (forward seul, Qwen3-0.6B, BF16, RTX 5090)

### 1.1 Empreinte et préremplissage

| Grandeur | Valeur |
|---|---|
| Paramètres | 596 049 920 |
| Poids BF16 en mémoire | **1 136,9 MiB** |
| Couches / `hidden_size` / têtes KV / `head_dim` | 28 / 1024 / 8 / 128 |
| Vocabulaire | 151 936 |

| Préremplissage (batch 1) | tok/s | Pic alloué |
|---|---:|---:|
| 512 tokens | 21 591 | 1 240 MiB |
| 2 048 tokens | 80 283 | 1 446 MiB |
| 4 096 tokens | 79 511 | 1 723 MiB |

### 1.2 Décodage avec cache KV

| Contexte | Lot 1 | Lot 8 | Lot 16 | Lot 32 | Lot 64 |
|---|---:|---:|---:|---:|---:|
| 1 024 tokens | 33,7 | 347,2 | 687,6 | **1 379,5** | **2 299,4** |
| 4 096 tokens | 44,6 | 338,7 | 620,9 | 763,7 | 51,7 *(effondrement)* |

tok/s. Le lot 64 à 4 096 de contexte s'effondre (51,7 tok/s) : le cache KV
(28,7 GiB pour 64 séquences) plus les activations **dépassent le budget**
(36,5 GiB alloués au pic, OOM au-delà).

### 1.3 Cache KV (théorique = mesuré)

| Contexte | Par séquence | Lot 16 | Lot 32 | Lot 64 |
|---|---:|---:|---:|---:|
| 1 024 | 112 MiB | 1 792 MiB | 3 584 MiB | 7 168 MiB |
| 4 096 | 448 MiB | 7 168 MiB | 14 336 MiB | 28 672 MiB |

### 1.4 Coût du scoring professeur (lots de branches, `k = 16`)

Un forward de `k` branches en parallèle, contexte complet rejoué
(identique à la méthode de `_suffix_loglikelihoods` du diagnostic) :

| Contexte | Branches | tok/s | s / forward | Pic alloué |
|---|---:|---:|---:|---:|
| 512 | 16 | 106 115 | **0,077** | 2 273 MiB |
| 1 024 | 16 | 97 828 | 0,168 | 3 378 MiB |
| 2 048 | 16 | 87 484 | 0,375 | 5 586 MiB |
| 4 096 | 16 | 76 412 | 0,858 | 10 003 MiB |
| 2 048 | 64 | 88 560 | 1,480 | 18 835 MiB |
| 4 096 | 64 | 3 255 | 15,7 | 36 501 MiB |
| 4 096 | 128 | — | **OOM** | — |

Le scoring est **linéaire en la longueur de contexte** et sature à ~90 000
tok/s jusqu'à 2 048 tokens ; au-delà de 4 096 tokens × 64 branches, la mémoire
devient le facteur limitant.

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
(§1.4) : un rollout de 4 000 tokens coûte ≈ 8× le scoring d'un rollout de
512 tokens par position masquée.

| Longueur max | Conséquence |
|---|---|
| 4 000 (papier) | scoring ≈ 0,86 s par position masquée à 4 096 de contexte ; **incompatible** avec un nombre de prompts significatif sur une carte |
| **512** | contexte ≤ 1 024 ⇒ 0,168 s par position ; cache KV 112 MiB/séquence |
| 256 | contexte ≤ 768 ; marge mémoire maximale |

**Longueur retenue : 256 à 512 nouveaux tokens**, avec plafond de contexte
1 024. C'est la principale réduction d'échelle, et elle est **dictée par la
mémoire et le scoring**, pas par un choix de commodité.

### 3.4 Itérations et taille de lot

| Contrainte | Valeur | Source |
|---|---|---|
| Débits de décodage soutenus | 1 379 tok/s (lot 32, 1 024) ; 2 299 tok/s (lot 64, 1 024) | §1.2 mesuré |
| Cache KV par séquence | 112 MiB (1 024) / 448 MiB (4 096) | §1.3 mesuré |
| Marge mémoire (élève + professeur entraînés) | ~24 GiB | §3.1, §3.2 |

| Lot de rollouts | Contexte 1 024 | Cache KV | Verdict |
|---:|---|---:|---|
| 16 | 1 024 | 1,8 GiB | confortable |
| **32** | 1 024 | 3,6 GiB | **lot retenu** |
| 64 | 1 024 | 7,2 GiB | possible, gain marginal (+67 % de débit) |
| 32 | 4 096 | 14,3 GiB | possible, mais scoring 5× plus cher par position |

**Lot retenu : 32 rollouts simultanés**, contexte 1 024. Un lot de 64 ne
rapporte que 67 % de débit en plus pour le doublement du cache ; il sera
préféré si la pression mémoire est faible.

---

## 4. Projection d'un run significatif

**Hypothèses** (à confirmer par un protocole pré-enregistré, J15) :

* élève = professeur = Qwen3-0.6B ; `N = 4`, 4 pas de débruitage, `k = 16` ;
* rollout : 1 par prompt, 256 tokens générés, contexte ≤ 1 024 ;
* **un état de débruitage par bloc** (App. C.2 du papier), soit
  64 blocs scorés par rollout ;
* ~2,2 positions masquées par bloc (`N = 4`, rétention 0,45) ;
* un **seul** rollout par prompt et par itération, comme le papier
  (128 prompts × 483 itérations = 61 824 ≈ 61 816 prompts).

| Poste | Coût calculé | Base |
|---|---|---|
| Génération du rollout (élève) | 8 000 × 256 = 2,05 M tokens ÷ 1 379 tok/s ≈ **0,4 h** | mesuré §1.2 |
| Scoring professeur (cache de préfixe partagé) | 8 000 × 64 blocs × 2,2 positions × 10–40 ms ≈ **3–12 h** | **inférence** — non mesuré |
| Scoring professeur (sans cache, contexte rejoué) | 8 000 × 64 × 2,2 × 0,168 s ≈ **52 h** | mesuré §1.4 |
| Entraînement élève (forward + backward + backward sur états) | 4 M passes de tokens ÷ ~7 000 tok/s effectifs ≈ **0,2 h** | **inférence** (forward mesuré, backward ×3 estimé) |
| **Total, avec cache de préfixe** | **≈ 4–13 h** | |
| **Total, sans cache de préfixe** | **≈ 52 h** | |

Le cache de préfixe (App. B du papier : préfixe causal partagé entre les
branches, lot réduit au-delà de 16 GiB) n'est **pas** une optimisation
optionnelle : sans lui, le scoring professeur est le poste dominant et rend
le run déraisonnable.

### 4.1 Extrapolation à l'échelle du papier

| Protocole | Prompts | Rollout | Temps mural projeté (1 × 5090) |
|---|---:|---:|---|
| **Réduit recommandé** | 8 000 | 256 tok | **4–13 h** |
| Réduit long | 20 000 | 256 tok | 10–32 h |
| 1 epoch papier | 61 816 | 256 tok | 30–100 h |
| 1 epoch papier, rollout 4 000 tok | 61 816 | 4 000 tok | 480–1 600 h *(hors de portée)* |

### 4.2 Comparaison avec 8 × H100

| | Papier (8 × H100 80 Go) | Local (1 × RTX 5090) | Ratio |
|---|---:|---:|---:|
| GPU-heures pour atteindre la performance d'OPDLM, échelle 0,6B | 8 × 13,15 ≈ **105 h-GPU** | run réduit 8 000 prompts : **4–13 h-GPU** | ~10× moins |
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
   256 tokens, lot 32, contexte 1 024, cache de préfixe partagé. Budget :
   **4–13 h** sur une RTX 5090, dont l'essentiel est le scoring professeur.
2. **L'échelle 4B–8B est hors de portée** d'une seule RTX 5090, même avec
   déport CPU : c'est un problème de mémoire d'états d'optimiseur, pas de
   débit. Pour l'échelle du papier, il faut **8 × H100 80 Go** (ou une
   location équivalente), soit ~105 h-GPU pour l'échelle 0,6B seule.
3. **Le poste à optimiser en priorité est le scoring professeur** : le cache
   de préfixe partagé (App. B) et la réduction de la longueur de rollout
   divisent le coût par 10 à 100. C'est aussi le poste que la porte J14 a
   validé numériquement (cible future-aware) sans le valider en coût.
4. **Ce chiffrage n'autorise aucun entraînement.** Il propose à J15 un
   protocole réduit **à pré-enregistrer** (prompts, seed, IC, budget) avant
   toute exécution.

---

## 6. Limites de ce chiffrage

* Les mesures sont des **forwards sans gradient** sur le **modèle AR** de
  0,6B. Le rollout d-OPD est un modèle **de diffusion par blocs** : le coût
  par token est approché, pas mesuré.
* Le coût du scoring avec cache de préfixe est une **inférence** (10–40 ms
  par position masquée) : elle doit être **mesurée** avant tout engagement de
  budget.
* Le coût de backward de l'élève est **estimé** (≈ 3× le forward) ; la
  rétention des états de débruitage, le mask-token et le découpage en blocs
  de `StudentPolicy` ne sont pas encore implémentés.
* La notion d'« epoch » du papier est reconstituée (1 rollout/prompt) à
  partir de 128 × 483 ≈ 61 816 ; le document n'explicite pas la
  correspondance.
* Les projections sont des ordres de grandeur destinés à **décider**
  (local 0,6B / location 8B), pas à fixer un protocole.
