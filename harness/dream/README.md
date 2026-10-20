# Harnais d'évaluation Dream-7B

Évaluation de **Dream-v0-Base-7B** et **Dream-v0-Instruct-7B** selon le
protocole A2D App. A.1 (arXiv:2610.08108, PDF p. 17-20), via le **fork
lm-evaluation-harness de l'équipe Dream** (`DreamLM/Dream`, dossier
`eval_instruct/lm_eval`, modèle `diffllm`).

> Ce harnais a été **reconstruit dans le dépôt** par la mission J6bis (2026-10-20).
> La version J6 vivait hors du dépôt (`/home/sjteb/dream-harness`) ; elle est
> décrite plus bas, avec la correspondance exacte des résultats.

## Contenu

| Chemin | Rôle |
|---|---|
| `setup.sh` | construit l'environnement **dans le dépôt** (idempotent) |
| `requirements-dream.txt` | versions épinglées (torch 2.8.0+cu128, transformers 4.46.2, …) |
| `tasks/` | tâches lm_eval : `gsm8k_cot_a2d.yaml`, `ifeval_a2d.yaml` + utilitaires |
| `run_evals.sh` | lanceur paramétré (chemins relatifs, skip, entrelacement) |
| `summarize.py` | consolidation des `results_*.json` → scores + IC de Wilson |
| `.venv/`, `.fork/`, `.cache/` | **non versionnés** (voir `harness/README.md`) |

## Reconstruire l'environnement

```bash
bash harness/dream/setup.sh
```

* clone le fork `DreamLM/Dream` dans `.fork/`, **épinglé sur le commit**
  `31f94a60d187e3fd481fee3bbc2c732eb94a879c` (c'est celui qui a produit les
  résultats J6 : `git_hash` / `upper_git_hash` dans chaque `results_*.json`) ;
* crée `.venv/` (Python 3.12) et installe `torch==2.8.0` (cu128, nécessaire pour
  sm_120 / RTX 5090), `transformers==4.46.2`, `datasets==2.21.0`, puis le fork
  lm-eval en editable ;
* redirige `HF_HOME`, `HF_DATASETS_CACHE`, `HUGGINGFACE_HUB_CACHE`,
  `PIP_CACHE_DIR`, `XDG_CACHE_HOME` et `TMPDIR` vers `.cache/` : **aucune
  écriture hors du dépôt**.

**Coût mesuré sur DrvFs (`/mnt/d`) :** venv de **7,1 Go** (7 537 409 029 o) ;
`import torch` = **15,8 s** contre **1,05 s** depuis le disque local (`/usr`,
ext4), soit un facteur **×15** dû au système de fichiers. Le `du` du venv
prend à lui seul **269 s**. C'est le prix de la règle « aucune écriture hors du
dépôt » : le harnais est dans le dépôt, donc sur DrvFs.

## Lancer une évaluation

```bash
# tout : base + instruct, GSM8K + MATH + IFEval (entrelacé par tâche)
bash harness/dream/run_evals.sh

# un seul modèle, une seule tâche, effectif réduit
MODEL_DIR=checkpoints/dream_7b_base TASK=gsm8k LIMIT=8 \
  bash harness/dream/run_evals.sh
```

| Variable | Défaut | Rôle |
|---|---|---|
| `MODEL_DIR` | `checkpoints/dream_7b_base,checkpoints/dream_7b_instruct` | checkpoints à évaluer (chemins relatifs à la racine du dépôt) |
| `TASK` | `gsm8k,math,ifeval` | tâches ; `math` = les 7 sous-tâches `minerva_math_*` |
| `LIMIT` | `300` / `25` / `8` | effectif par tâche (`tache=N` ou nombre unique) |
| `SEED` | `0,1234,1234,1234` | graines lm_eval (`random,numpy,torch,fewshot`) — **valeur par défaut du fork**, celle de J6 |
| `OUTDIR` | `eval/dream_a2d` | racine des résultats |
| `BATCH` | `4` | taille de batch |
| `SKIP_EXISTING` / `FORCE` | `1` / `0` | ne pas rejouer une tâche déjà faite / la rejouer quand même |

Le protocole de décodage est **figé** dans le script : `temperature=0.1`,
`top_p=0.9`, `alg=entropy`, `diffusion_steps = max_new_tokens` (un pas de
débruitage par token), `dtype=bfloat16`, chat template appliqué, 0-shot,
budgets 256 (GSM8K) / 512 (MATH) / 1280 (IFEval), prompt système GSM8K de
l'App. A.1 cité mot pour mot.

**Sortie** : `OUTDIR/<alias_modele>/<tache>/results_*.json` et
`…/samples_*.jsonl` — noms de dossiers **portables** (plus de chemin absolu
encodé dans le nom, cf. `scripts/restructure_eval_dirs.py`).

## Résultats : provenance exacte

| Jeu de résultats | Commande d'origine | Version | Durée |
|---|---|---|---|
| `eval/dream_a2d/dream_7b_base/gsm8k_cot_a2d` (n=300) | `/home/sjteb/dream-harness/run_dream_evals2.sh` (V2, entrelacé) | fork `31f94a6`, graines `0,1234,1234,1234` | 00:02:25 → 00:53:06 |
| `eval/dream_a2d/dream_7b_instruct/gsm8k_cot_a2d` (n=300) | idem | idem | 00:53:54 → 01:44:41 |
| `eval/dream_a2d/dream_7b_base/minerva_math_algebra` (n=175) | idem | idem | 01:44:41 → 03:25:13 |
| `eval/dream_a2d/dream_7b_instruct/minerva_math_algebra` (n=175) | idem | idem | 03:25:13 → 05:05:05 |
| `eval/dream_a2d/dream_7b_base/ifeval_a2d` (n=8) | idem, **relance avec `LIMIT_IFEVAL=8`** après interruption de la passe à 30 | idem | 05:43:24 → 06:26:56 |
| `eval/dream_a2d/dream_7b_instruct/ifeval_a2d` (n=8) | idem | idem | 06:26:56 → 07:10:41 |

* **V1 vs V2.** La copie V1 (`run_dream_evals.sh`) est celle qui était dans le
  dépôt : boucle par modèle (base entière puis instruct entière) et **pas** de
  garde `SKIP`. C'est la **V2** (entrelacée par tâche, avec garde `SKIP`) qui a
  produit les résultats déposés. La V2 est récupérée et reparamétrée dans
  `run_evals.sh`.
* **IFEval.** La passe `LIMIT_IFEVAL=30` a été **interrompue** (4/30 prompts à
  322,6 s/prompt) puis relancée avec `LIMIT_IFEVAL=8` ; les résultats
  déposés portent `n-samples.effective = 8`.
* **MATH.** `LIMIT_MATH=25` par sous-tâche × 7 sous-tâches = **175** problèmes
  (le jeu du papier est MATH-500, 500 problèmes) ⇒ la métrique est notée
  **MATH-175** dans tout le dépôt.
* **Le harness a été rejoué après reconstruction** (J6bis C4) : 8 prompts
  GSM8K sur `dream_7b_base`, **mêmes items** que les 8 premiers de la passe à
  300 (comparaison des `doc_id` et des questions : identique), **mêmes
  graines**, **même commit de fork**. Score sur ces 8 items :
  `flexible-extract` 0,625 / `strict-match` 0,375. Durée : 3 min 46 s
  (13:42:02 → 13:45:48), dont ~16 s d'import torch et ~2 min de chargement du
  modèle depuis DrvFs.

## Synthèse

```bash
python harness/dream/summarize.py --root eval/dream_a2d \
    --out eval/dream_a2d/SUMMARY.json --markdown eval/dream_a2d/SUMMARY.md
```

Les cibles de la Table 16 de l'article sont **MATH-500** alors que la mesure est
**MATH-175** : la comparaison est à périmètre différent (voir
`docs/reports/AUDIT_J6.md`, MAJ-04). IFEval (n=8) est un contrôle de
fonctionnement, pas une mesure.
