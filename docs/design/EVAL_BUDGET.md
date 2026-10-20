# Budget d'évaluation — coûts mesurés (Dream-7B, RTX 5090 32 Go, WSL2)

**Date** : 2026-10-20 · **Mission** : J6bis D1 · **Statut** : **chiffrage mesuré**
(aucune valeur supposée : chaque coût est extrait d'un `results_*.json` ou d'un
log du harnais).

> **Règle de vocabulaire** : les durées ci-dessous sont des **mesures** sur cette
> machine, dans ce protocole. Elles ne valent pas pour une autre machine, un
> autre protocole ou un autre modèle. Toute projection est signalée comme telle.

## 1. Source des mesures

| Source | Nature |
|---|---|
| `eval/dream_a2d/*/*/results_*.json` | champ `total_evaluation_time_seconds` des 6 runs J6 (champ fourni par le fork lm-eval) |
| `harness/dream/.cache/logs/*.log` | horodatages de début/fin des runs J6bis |
| `_artefact/smoke_c4/` | smoke GSM8K n=8, harnais reconstruit (batch 4) |
| `_artefact/d3_sweep/b{1,2,4,8,16}/` | balayage de batch sur 16 prompts (D3) |

Protocole **inchangé** dans tous ces runs : `temperature=0.1`, `top_p=0.9`,
`alg=entropy`, `diffusion_steps = max_new_tokens` (un pas de débruitage par
token), `dtype=bfloat16`, chat template, 0-shot, `--batch_size 4`,
graines `0,1234,1234,1234` (valeurs par défaut du fork).

## 2. Latence mesurée par tâche et par modèle

| Modèle | Tâche | Tokens (pas de débruitage) | n | Durée mesurée (s) | **s / prompt** | Débit |
|---|---|---:|---:|---:|---:|---:|
| Dream-7B-**Base** | GSM8K | 256 | 300 | 3 036,4 | **10,12** | 9,9 prompts/min |
| Dream-7B-**Instruct** | GSM8K | 256 | 300 | 3 041,3 | **10,14** | 9,9 prompts/min |
| Dream-7B-**Base** | MATH-175 | 512 | 175 | 6 028,6 | **34,45** | 1,7 prompt/min |
| Dream-7B-**Instruct** | MATH-175 | 512 | 175 | 5 988,5 | **34,22** | 1,8 prompt/min |
| Dream-7B-**Base** | IFEval | 1280 | 8 | 2 607,8 | **325,97** | 0,18 prompt/min |
| Dream-7B-**Instruct** | IFEval | 1280 | 8 | 2 620,9 | **327,61** | 0,18 prompt/min |

**Lecture** : le coût est **quasi linéaire en le budget de tokens** (256 → 512
multiplie le coût par 3,4 ; 256 → 1280 par 32). C'est attendu : un pas de
débruitage par token, donc le nombre d'étapes de décodage est linéaire en
`max_new_tokens`.

### 2.1 Surcoûts fixes (à ajouter à toute évaluation)

| Poste | Mesure | Comment |
|---|---:|---|
| **Amorçage CUDA** (première passe : allocation, noyaux, chemins du code distant) | **≈ 96 s** | ajustement affine sur 3 points à batch 4 : 8 prompts (187,6 s), 16 prompts (252,7 s), 300 prompts (3 036,4 s) ⇒ 96,0 s + 39,2 s × nb_lots (le lot de 16 prédit 252,8 s pour 252,7 s mesurés) |
| **Import torch** (venv sur DrvFs) | **15,8 s** | 3 mesures : 15,86 / 15,78 / 15,77 s (`/mnt/d`) contre 1,05 s en local (`/usr`) |
| **Chargement du checkpoint** (15 Go depuis DrvFs) + tokeniseur | **≈ 25 s** | smoke C4 : 226 s de mur − 187,6 s d'évaluation = 38,4 s (dont 15,8 s d'import) |
| **Écriture des échantillons** | négligeable | ≈ 1,3 Mo pour 300 prompts GSM8K |

**Coût fixe total par run : ≈ 140 s** (amorçage + import + chargement), quelle
que soit la taille de l'évaluation. Il est **dilué** sur les gros jeux et
**dominant** sur les petits (IFEval n=8 : 96 s d'amorçage pour 8 prompts).

## 3. Projection vers les jeux complets (projection, pas mesure)

Effectifs des jeux complets : GSM8K **1 319** problèmes, MATH (test) **5 000**,
IFEval **541** prompts. La colonne « mesuré × effectif » est une **projection
linéaire** de la Section 2 (hypothèse : coût par prompt constant ; le coût fixe
est ajouté une fois).

| Modèle | Tâche | n complet | **Coût projeté par modèle** | Coût pour 2 modèles | Commentaire |
|---|---|---:|---:|---:|---|
| Base | GSM8K | 1 319 | 13 350 s = **3 h 42** | **7 h 25** | |
| Base | MATH | 5 000 | 172 250 s = **47 h 51** | **95 h 42** | hors budget (le jeu du papier est MATH-500 : ×10,3 ⇒ **4 h 39** par modèle, **9 h 19** pour les deux) |
| Base | IFEval | 541 | 176 350 s = **48 h 59** | **97 h 59** | **hors budget** : le papier lui-même a abaissé le budget de 8 192 à 1 280 tokens pour IFEval (« *too expensive* », App. A.1) |

**Conclusion budget (inchangée par rapport à J6)** : à 256/512/1 280 tokens et
batch 4, **seul GSM8K complet est jouable** sur cette machine. MATH-500 (5 000
problèmes) ne l'est pas non plus ; MATH-175 (175) est jouable en 1 h 40 par
modèle.

## 4. Balayage des leviers de vitesse (D3) — 16 prompts GSM8K, protocole figé

Seul le **batch** varie ; tout le reste (temperature, top-p, pas de débruitage,
tokens, graines) est identique. Mesures dans `_artefact/d3_sweep/`.

| Batch | Durée d'évaluation (s) | s / prompt (brut) | s / prompt (amorce 96 s retirée) | Gain vs batch 1 | nb de lots |
|---:|---:|---:|---:|---:|---:|
| 1 | 291,7 | 18,23 | 12,2 | ×1,00 | 16 |
| 2 | 273,7 | 17,11 | 11,1 | ×1,10 | 8 |
| **4** (référence J6) | 252,7 | 15,80 | **9,8** | **×1,25** | 4 |
| 8 | 244,2 | 15,26 | 9,3 | ×1,32 | 2 |
| 16 | 352,8 | 22,05 | 16,0 | ×0,76 | 1 |

* **Le batching ne rapporte que ~25 à 32 %** : le décodage à diffusion enchaîne
  `max_new_tokens` passes causales sur le même prompt, et le coût par passe est
  dominé par la **lecture des poids** (15 Go par pas), pas par le GEMM. Un lot
  plus grand amortit mieux le poids, mais le gain reste marginal.
* **Batch 16 est plus lent** (×0,76) : un seul lot de 16 × 256 positions fait
  grimper l'attention et l'empreinte mémoire ; le gain attendu est annulé.
* **Batch 4 est un bon compromis** : proche de l'optimum mesuré, et c'est la
  valeur utilisée par J6 (les résultats déposés sont donc comparables).

### 4.1 Avertissement critique : le batch change les sorties

À **graines identiques** (`0,1234,1234,1234`) et **protocole identique**, les
sorties **diffèrent** selon la taille de lot :

| Comparaison | Réponses identiques au caractère |
|---|---:|
| batch 2 vs batch 1 | **0 / 16** |
| batch 4 vs batch 1 | **5 / 16** |
| batch 8 vs batch 1 | **4 / 16** |
| batch 16 vs batch 1 | **4 / 16** |

Scores sur ces 16 prompts (`flexible-extract`) : 0,4375 (b1) · 0,4375 (b2) ·
0,5000 (b4) · 0,4375 (b8) · 0,5000 (b16).

**Conséquence** : pour un modèle de diffusion, le **taille de lot fait partie
du protocole effectif**. Deux évaluations à même graine mais à lot différent ne
sont **pas** reproductibles. Toute reprise doit donc fixer `--batch_size 4`
(cf. `docs/design/G1_PROTOCOL.md`), et toute affirmation de reproductibilité
doit préciser le lot.

*Cause* [INFERENCE] : le décodage à diffusion est stochastique **par séquence**
(`alg=entropy`, `temperature=0.1`) et l'état d'échantillonnage dépend de la
composition du lot (padding à gauche, longueur commune au lot, ordre de
traitement) ; le même germe n'est donc pas consommé dans le même ordre.

## 5. Leviers **non** disponibles (et pourquoi)

| Levier demandé | Statut | Raison (mesurée / lue dans le code) |
|---|---|---|
| **SDPA** | **déjà actif** | `DreamModel._supports_sdpa = True` (`modeling_dream.py:533`) et `config._attn_implementation` n'est pas forcé dans `config.json` ⇒ transformers 4.46 sélectionne `sdpa`. Passer en `eager` serait un **ralentissement**, et le fork n'expose pas `attn_implementation` (`_create_model_and_tokenizer` ne transmet que `pretrained`, `torch_dtype`, `trust_remote_code`). |
| **`torch.compile`** | **inaccessible par le harnais** | le chargeur du fork (`diffllm.py:_create_model_and_tokenizer`) ne passe aucun kwarg supplémentaire à `from_pretrained` ; `torch.compile` exigerait de modifier le code distant `modeling_dream.py` — c'est un changement de **numérique du modèle**, donc un changement de protocole, interdit ici. |
| **Réduire `diffusion_steps`** | **interdit** | `diffusion_steps = max_new_tokens` fait partie du protocole App. A.1 (« un pas de débruitage par token ») : le modifier change l'expérience, pas seulement son coût. |
| **Réduire `max_new_tokens`** | **interdit** | idem : le budget de tokens est un paramètre de protocole (256 / 512 / 1 280). |
| **Quantification (FP8/INT8)** | **interdit** | le protocole est en **bfloat16** ; la campagne J5 en FP8 a été déclarée non conforme. |

## 6. Règle de réduction de coût (pour le protocole G1)

Si une évaluation dépasse le budget de 2 h, **on réduit `n`**, jamais les
paramètres de décodage :

1. réduire `n` (sous-ensemble d'items **identiques** pour tous les modèles
   comparés — le même `--limit` ⇒ les mêmes items, dans l'ordre du dataset) ;
2. ne jamais toucher `temperature`, `top_p`, `alg`, `diffusion_steps`,
   `max_new_tokens`, `num_fewshot`, `--batch_size` ;
3. rapporter l'intervalle de confiance de l'effectif réduit et **ne pas
   conclure** au-delà de ce qu'il porte.

## 7. Ce qui n'est **pas** mesuré

* Les durées ci-dessus valent pour **ce** modèle (Dream-7B), **ce** protocole et
  **cette** machine. Rien n'est mesuré pour Nemotron-Labs-Diffusion-3B, pour
  DiffusionGemma, ni pour le vLLM FP8 de la campagne J5.
* Le coût **par échantillon** du diffusion Nemotron-3B est **projeté** dans
  `docs/design/G1B_NEMOTRON3B.md` (marqué `[INFERENCE]`) : le débit de tokens
  par forward (TPF) rapporté par l'outil `evaluate.py` de NVlabs vaut 5,89 sur
  **8B**, pas sur 3B.
* Le coût d'un **sweep d'hyperparamètres** (λ × 541 prompts IFEval) est projeté
  à partir des latences ci-dessus : ≈ 2 × 541 × 326 s = **98 h** par point de
  grille en IFEval complet, d'où la décision d'exclure IFEval du critère G1
  (cf. `docs/design/G1_PROTOCOL.md`).
