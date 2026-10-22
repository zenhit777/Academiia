# Harnais d'évaluation — convention `harness/<famille>/`

Ce dossier regroupe les **harnais d'évaluation** du projet. Un harnais = tout ce
qu'il faut pour **rejouer une évaluation conforme au protocole** d'un article :
le code de lancement, les tâches, l'environnement, et l-outillage de synthèse.

## Convention

```
harness/
├── README.md          ← ce fichier (la convention)
└── <famille>/
    ├── README.md      ← ce que fait ce harnais, comment le rejouer, ce qui vient d'où
    ├── setup.sh       ← construction de l'environnement (idempotent)
    ├── requirements-*.txt
    ├── tasks/         ← tâches lm_eval (YAML + code associé)
    ├── run_evals.sh   ← lanceur paramétré (chemins RELATIFS)
    ├── summarize.py   ← consolidation des résultats (scores + IC)
    ├── .venv/         ← environnement Python (NON versionné)
    ├── .fork/         ← clone du harnais amont, épinglé par commit (NON versionné)
    └── .cache/        ← HF_HOME, PIP_CACHE_DIR, TMPDIR, XDG_CACHE_HOME (NON versionné)
```

* **Une famille = un dossier.** `modules/a2d/harness/dream/` pour Dream-7B ;
  `harness/nemotron/` pour Nemotron-Labs-Diffusion-3B (à venir, porte G1b) ;
  `harness/diffusiongemma/` pour le setup médical (porte G2).
* **Tout ce qui n'est pas partageable vit dans le dossier de la famille**, sous
  `.venv/`, `.fork/` et `.cache/`. Ces trois répertoires sont ignorés par git
  (`.gitignore`) : un environnement n'est pas versionnable.
* **Aucune écriture hors du dépôt.** `setup.sh` redirige `HF_HOME`,
  `PIP_CACHE_DIR`, `TMPDIR` et `XDG_CACHE_HOME` vers `harness/<famille>/.cache/`
  **avant** toute opération réseau. Un harnais ne doit jamais laisser de trace
  dans `~/.cache/`, `/tmp` ou ailleurs.
* **Chemins relatifs.** Les lanceurs dérivent la racine du dépôt de leur propre
  emplacement (`$(dirname "${BASH_SOURCE[0]}")/../..`) et résolvent les
  checkpoints relativement à cette racine. Aucun chemin absolu en dur.
* **Les paramètres d'évaluation sont des variables d'environnement**, pas des
  constantes dans le script : `MODEL_DIR`, `TASK`, `LIMIT`, `SEED`, `OUTDIR`,
  `BATCH`. Le protocole (température, top-p, budget de tokens, nombre de
  few-shot) est en revanche **figé dans le script** : changer un paramètre de
  décodage change l'expérience et doit être un choix explicite, consigné dans le
  rapport.
* **Les résultats ne sont pas dans le harnais** : ils vont dans `eval/<famille>/`,
  versionnés, avec une structure `eval/<famille>/<alias_modele>/<tache>/` dont
  les noms de dossiers ne contiennent **aucun chemin absolu**.

## Règles de résultat

1. Un résultat versionné doit être **rejouable** : le harnais, son commit et
   ses versions de paquets sont identifiables depuis le dépôt.
2. Tout résultat porte son **effectif**, ses **graines** et son **protocole**
   dans les métadonnées (`results_*.json` du fork lm-eval).
3. Une évaluation à effectif réduit est un **contrôle**, pas une mesure : le
   rapport doit le dire (cf. `modules/a2d/docs/design/EVAL_BUDGET.md`).
4. Les dossiers dont le nom encode un chemin absolu sont **interdits** : voir
   `modules/a2d/eval/restructure_eval_dirs.py` (mission J6bis C6).

## Harnais présents

| Famille | Dossier | Modèles | Protocole | État |
|---|---|---|---|---|
| Dream | [`dream/`](dream/) | Dream-v0-Base/Instruct-7B | A2D App. A.1 (T=0.1, top-p=0.9, entropie) | opérationnel (reconstruit et vérifié le 2026-10-20) |
