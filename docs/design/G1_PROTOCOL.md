# Protocole de la porte G1 — définition, critères, budget

**Date** : 2026-10-20 · **Mission** : J6bis D2 · **Décision** : N2-4 (review
architecte) — le critère C0.5 du prompt J6 (« ±3 points, IFEval complet ») est
**remplacé** par le présent protocole.

> **Pourquoi le critère est remplacé.** Le critère ±3 exigeait de rejouer les
> jeux **complets** (GSM8K 1 319, MATH-500 500, IFEval 541) pour quatre modèles
> et deux points de grille. À la latence **mesurée** sur cette machine
> (326 s/prompt en IFEval à 1 280 pas de débruitage, cf.
> `docs/design/EVAL_BUDGET.md`), IFEval seul coûterait **98 h par point de
> grille**, et le jeu complet n'est pas jouable. De plus, « la cible du papier
> est dans l'IC 95 % » est une **non-rejection**, pas une preuve d'équivalence :
> l'argument de `docs/reports/J6.md` §1.1 ne démontre pas ±3 points
> (MAJ-05). Le présent protocole **remplace** la question « le score de D_b
> reproduit-il la Table 16 ? » par la question **que l'article pose
> réellement** : « le transfert et le merge improves-ils le modèle de
> diffusion ? ».

## 1. Question à laquelle G1 doit répondre

> **Est-ce que l'A2D-Transfer et l'A2D-Merge améliorent un modèle de diffusion
> réel (Dream-7B) quand on lui applique un vecteur de tâche autoregressif
> (Qwen2.5-7B) ?**

La porte G1 est **franchie** si, et seulement si :

1. les **quatre** modèles comparés (D_b, D_p, Transfer α=0,9, Merge λ=0,6) ont
   été évalués sur les **mêmes items**, avec le **même protocole de décodage**,
   et l'identité des items est **prouvée** par empreinte ;
2. chaque fusion a **passé** la vérification v2 (`post_write_verification.ok`
   et `verify_merge` ⇒ VERDICT OK) ;
3. les **critères d'écart** de la Section 4 sont atteints ;
4. les **intervalles de confiance appariés** sont rapportés.

Si un seul de ces quatre points échoue, G1 reste **ouverte** et le rapport le
dit sans ambiguïté.

## 2. Les quatre modèles

| # | Modèle | Rôle | Origine | Coefficient |
|---|---|---|---|---|
| M0 | `checkpoints/dream_7b_base` (**D_b**) | référence basse | public | — |
| M1 | `checkpoints/dream_7b_instruct` (**D_p**) | référence haute | public | — |
| M2 | `checkpoints/dream_transfer_a09` | transfert A2D (eq. 1) | **J7** | **α = 0,9** (Table 6) |
| M3 | `checkpoints/dream_merge_l06` | merge A2D (eq. 2) | **J7** | **λ = 0,6** (Table 6) |

* α = 0,9 et λ = 0,6 sont les coefficients de la **Table 6** de l'article pour la
  famille Dream ; ils ne sont **pas** réajustés sur la machine (l'article
  sélectionne sur une ancre, App. A.5 ; cette sélection est hors périmètre de
  G1).
* τ_A = Qwen2.5-7B-Instruct − Qwen2.5-7B (paire de checkpoints complets) ;
  τ_D = Dream-7B-Instruct − Dream-7B-Base.
* Le mapping est l'**identité** : Dream conserve l'architecture, les noms et
  les formes de son parent AR (App. A.2) ⇒ les 339 tenseurs sont fusionnés,
  embeddings et tête de sortie inclus.

## 3. Items, protocole, preuve d'identité

### 3.1 Les items : **ceux déjà mesurés**, pas de nouveaux

M0 et M1 sont **déjà évalués** (J6) : aucune ré-évaluation n'est nécessaire ni
souhaitable. On **réutilise** ces mesures, à condition de **prouver** que :

* le jeu d'items est le même pour M0, M1, M2 et M3 ;
* la configuration de décodage est la même.

La preuve est **mécanique** : `scripts/eval_item_fingerprint.py` calcule, pour
chaque run, le SHA-256 de la liste des `(doc_id, doc_hash)` et le SHA-256 des
`prompt_hash` (le prompt exact envoyé au modèle, chat template inclus). Les
empreintes **de référence** sont versionnées dans
`eval/dream_a2d/ITEM_FINGERPRINTS.json` :

| Tâche | n | `items_sha256` (préfixe) | `prompts_sha256` (préfixe) |
|---|---:|---|---|
| GSM8K | 300 | `2d56d1fdcbd24b57…` | `cc3e4a4bfa53ea71…` |
| MATH-175 | 175 | `e8d1dae81491a7c8…` | `a00782289787b2e9…` |
| IFEval | 8 | `0d66e89ed6abd889…` | `62df58fa1ef767e7…` |

M0 et M1 ont **déjà** les mêmes empreintes par tâche (vérifié le 2026-10-20) :
les baselines sont donc bien sur le même jeu d'items.

**Contrôle à exécuter pour M2 et M3** (bloquant) :

```bash
python scripts/eval_item_fingerprint.py --root eval/dream_a2d \
    --out eval/dream_a2d/ITEM_FINGERPRINTS.json
# puis : pour chaque tache, items_sha256 et prompts_sha256 doivent etre
# IDENTIQUES a ceux de M0/M1 ; et le bloc `config` doit etre identique
# (meme batch_size, memes graines, meme model_args a `pretrained` pres).
```

### 3.2 Protocole de décodage : **figé**

| Paramètre | Valeur | Source |
|---|---|---|
| `temperature` | 0,1 | App. A.3 / script amont du fork |
| `top_p` | 0,9 | idem |
| `alg` | `entropy` | idem |
| `diffusion_steps` | `= max_new_tokens` (1 pas/token) | idem |
| `dtype` | `bfloat16` | idem |
| chat template | appliqué | idem |
| `num_fewshot` | 0 | App. A.1 |
| `max_new_tokens` | 256 (GSM8K) · 512 (MATH) · 1280 (IFEval) | App. A.1 |
| `batch_size` | **4** | mesuré (J6) — **fait partie du protocole effectif** (§3.3) |
| graines | `0,1234,1234,1234` | défaut du fork |
| extracteur | GSM8K : `flexible-extract` ; MATH : `math_verify` | App. A.3 |

### 3.3 Avertissement : le **batch** fait partie du protocole

Mesuré le 2026-10-20 (D3, 16 prompts GSM8K, graine identique) : à protocole
identique, **les sorties changent selon la taille de lot** — 0 à 5 réponses
identiques sur 16 entre le lot 1 et les lots 2/4/8/16, et des scores qui
fluctuent (0,4375 → 0,5000). Raison [INFERENCE] : le décodage à diffusion est
stochastique par séquence et l'état d'échantillonnage dépend de la composition
du lot (padding, longueur commune, ordre de traitement).

**Conséquence** : M2 et M3 doivent être évalués avec `--batch_size 4`, exactement
comme M0 et M1. Toute comparaison par item (bootstrap apparié) n'a de sens
qu'à lot identique. Le harnais le fixe par défaut (`harness/dream/run_evals.sh`).

## 4. Critères (décision N2)

### 4.1 Critères principaux

| # | Critère | Seuil | Tâches |
|---|---|---|---|
| **C1** | **Transfer − Base** | **≥ +10 points** | MATH-175 |
| **C2** | **Transfer − Base** | **≥ +8 points** | GSM8K |
| **C3** | **Merge − Instruct** | **≥ −2 points** (non-régression) | MATH-175 |
| **C4** | **Merge − Instruct** | **≥ −2 points** (non-régression) | GSM8K |

* Les seuils C1/C2 sont **calés sur l'article** : le papier reports **+19,4**
  points (Table 8, transfert Nemotron) et **+15,2** points (Table 7/12, merge).
  Les seuils retenus (10 et 8) sont une **frange basse** de ces gains : ils
  demandent un effet net sans exiger la reproduction du chiffre exact.
* C3/C4 sont des **non-régressions** : le merge ne doit pas dégrader le
  modèle post-entraîné. Le seuil de 2 points correspond à l'ordre de grandeur
  du bruit d'échantillonnage observé (IC 95 % de ±4,3 à ±7,1 points sur les
  effectifs de J6).
* Chaque écart est rapporté **avec son intervalle de confiance bootstrap
  apparié** (§4.2). Un écart dont l'IC contient 0 est rapporté comme tel : il
  n'est ni un gain ni une perte.

### 4.2 Rapport des intervalles

Pour chaque comparaison (M2−M0, M3−M1, sur chaque tâche) :

* **IC bootstrap apparié à 95 %** : 10 000 ré-échantillonnages des items
  (avec remise), **appariés** (le même item est tiré pour les deux modèles),
  percentile 2,5 % / 97,5 %, **graine fixée** (0 par défaut) ;
* décompte des paires discordantes (type McNemar) : `B seul`, `A seul`,
  `both`, `neither` ;
* IC de Wilson de chaque précision individuelle.

Outil : `harness/dream/.venv/bin/python scripts/paired_bootstrap.py --run-a … --run-b … --metric … --filter …`

```bash
# exemple de contrôle (déjà exécuté le 2026-10-20, M0 vs M1)
harness/dream/.venv/bin/python scripts/paired_bootstrap.py \
  --run-a eval/dream_a2d/dream_7b_base/gsm8k_cot_a2d \
  --run-b eval/dream_a2d/dream_7b_instruct/gsm8k_cot_a2d \
  --metric exact_match --filter flexible-extract
# -> 300 items apparies ; 68,67 % vs 84,00 % ; ecart +15,33 points ;
#    IC 95 % [+10,00 ; +20,67] ; discordantes B seul 58 / A seul 12
```

La métrique `math_verify` est **recalculée** par le script avec la fonction
`process_results` **du fork lui-même** (mêmes normalisations, même vérification
symbolique) : contrôle de fidélité effectué le 2026-10-20, le recalcul
reproduit exactement les scores publiés (24,00 % et 36,57 % pour
MATH-175).

### 4.3 Ce qui est **exclu** du critère

* **IFEval complet n'est pas dans le critère.** Coût mesuré : 541 prompts ×
  326 s = **49 h par modèle** ; c'est hors budget, et le papier lui-même a
  abaissé son budget IFEval pour cette raison. Un **sous-ensemble fixe de 150
  prompts** peut être joué en **tâche de nuit** (13,6 h par modèle, donc
  > 24 h pour les deux — à réduire davantage, ou à ne pas faire) ; il est
  **non bloquant** : son absence n'empêche pas de franchir G1.
  L'IFEval déjà mesuré (n=8) reste un **contrôle de fonctionnement**.
* **Aucune revendication « ±3 reproduit »** : le critère C0.5 de J6 est
  **abandonné** (N2-4). Les écarts à la Table 16 sont rapportés comme
  *observation de baseline*, avec leur IC, et **sans** conclusion
  d'équivalence. Si l'on veut un jour une telle affirmation, il faut des jeux
  complets et un test d'équivalence (TOST), pas un IC qui *contient* la cible.
* **MATH-500 n'est pas dans le critère** : le jeu mesuré est MATH-175
  (175/5000). Les cibles de la Table 16 (20,8 / 38,4) portent sur MATH-500 :
  elles sont rappelées comme *référence externe*, et les écarts sont
  explicitement des écarts **à périmètre différent**.

## 5. Budget

| Contrainte | Valeur |
|---|---|
| Durée d'**une** évaluation (1 modèle × 1 tâche) | **≤ 2 h** |
| Durée totale de la porte G1 (2 modèles × 2 tâches + vérifications) | **≤ 12 h** |
| Si 2 h sont dépassées | **réduire `n`** (jamais les paramètres de décodage) |
| Jobs GPU | aucun > 60 min sans découpage explicite ; ici : 4 évaluations ≤ 1 h 40 chacune |

Coûts **mesurés** (M0/M1 déjà faits, mêmes items) :

| Modèle | Tâche | n | Durée mesurée | Marge / 2 h |
|---|---|---:|---:|---|
| M0 | GSM8K | 300 | 50 min 36 s | 3,2× |
| M0 | MATH-175 | 175 | 1 h 40 min 29 s | 1,2× |
| M1 | GSM8K | 300 | 50 min 41 s | 3,2× |
| M1 | MATH-175 | 175 | 1 h 39 min 49 s | 1,2× |

⇒ **MATH-175 tient dans le budget de 2 h**, GSM8K largement. Le coût par
modèle fusionné est identique (même protocole, même nombre de tokens).

Si une évaluation dépasse 2 h (machine saturée, ou `n` augmenté) : on
**réduit `n`** au même sous-ensemble pour **les quatre** modèles (pour rester
appariable) et on recalcule les empreintes d'items de référence. On ne
touche **jamais** `temperature`, `top_p`, `alg`, `diffusion_steps`,
`max_new_tokens`, `num_fewshot` ni `--batch_size`.

## 6. Plan d'exécution (J7 — **hors périmètre de J6bis**)

1. **Fusion M2** (transfert α=0,9) :
   ```bash
   python scripts/merge_a2d.py --mode transfer \
       --diffusion-base checkpoints/dream_7b_base \
       --tau-a-post checkpoints/qwen25_7b_it --tau-a-base checkpoints/qwen25_7b \
       --alpha 0.9 --mapping identity --output checkpoints/dream_transfer_a09
   ```
   La **vérification post-écriture v2** est obligatoire et doit afficher
   `c_a_eff ≈ 0,9`, `max_ulp ≤ 1`, `n_unchanged_bit_identical = n_unchanged`.
2. **Vérification post hoc** :
   ```bash
   python scripts/verify_merge.py --merged checkpoints/dream_transfer_a09 \
       --reference checkpoints/dream_7b_base --mode transfer \
       --tau-a-post checkpoints/qwen25_7b_it --tau-a-base checkpoints/qwen25_7b \
       --alpha 0.9 --mapping identity
   ```
3. **Fusion M3** (merge λ=0,6) : mode `merge`, `τ_D` = paire
   `dream_7b_instruct` / `dream_7b_base`. La couverture des deux sources est
   **identique** (les deux index contiennent les mêmes 339 noms de tenseurs,
   vérifié le 2026-10-20) ⇒ politique `strict` acceptée, `both = 339`.
4. **Évaluation de M2 et M3** :
   ```bash
   MODEL_DIR=checkpoints/dream_transfer_a09 TASK=gsm8k LIMIT=300 BATCH=4 \
       bash harness/dream/run_evals.sh
   MODEL_DIR=checkpoints/dream_transfer_a09 TASK=math LIMIT=25 BATCH=4 \
       bash harness/dream/run_evals.sh
   # idem pour dream_merge_l06
   ```
5. **Empreintes d'items** : rejouer `scripts/eval_item_fingerprint.py` et
   vérifier l'identité des `items_sha256` / `prompts_sha256` avec M0/M1
   (**bloquant**).
6. **Écarts appariés** : `scripts/paired_bootstrap.py` sur les quatre paires
   (M2−M0, M3−M1) × (GSM8K, MATH-175).
7. **Verdict** : consigner dans `CHECKLIST.md` (table des portes) et dans
   `docs/reports/J7.md` : franchie / non franchie, critère par critère.

## 7. Ce que G1 ne démontrera pas

* **La validité scientifique de la bibliothèque de fusion** au sens d'une
  réplication : G1 démontre un **effet mesuré** sur une famille de modèles,
  avec des items identiques et une vérification mécanique des coefficients.
* **La généralisation à d'autres familles** (DiffusionGemma en G2, Nemotron-3B
  en G1b).
* **L'exactitude de la reproduction de la Table 16** (voir §4.3).
* **L'effet du choix de τ** : seule la paire Qwen2.5-7B → Dream-7B est testée.
