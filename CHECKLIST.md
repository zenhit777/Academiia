# CHECKLIST.md — Vérifications avant de considérer le projet "prêt"

## État des portes (au 2026-10-21, fin de la mission J8)

> Vue d'ensemble (ce qui est démontré, ce qui ne l'est pas, contraintes machine, recommandation
> DiffusionGemma) : [`modules/a2d/STATUS.md`](modules/a2d/STATUS.md).

| Porte | Contenu | État | Preuve |
|---|---|---|---|
| **J4Bis** | infrastructure GPU/service (compose, service FP8, smoke) | **OK (infra)** — validée le 2026-10-19 | `infra/vllm/RAPPORT_J4BIS-BIS.md` |
| **J5** | campagne d'évaluation par génération (103 prompts) | **NON CONFORME** au protocole du papier (FP8, `temperature=0`, 128 tokens, texte libre, sans juge) : contrôle de non-régression, pas une mesure de qualité | `modules/a2d/legacy/eval/RESULTS.md` |
| **G1** | mécanique A2D sur Qwen2.5-7B → Dream-7B (Tableaux 6, 16, 21) | **OUVERTE** — **G1 NON FRANCHIE** (J7, 2026-10-21). *Fait et mesuré* : géométrie Table 21 (écarts ≤ 0,11 %) ; baseline D_b/D_p **exécutée** sur GSM8K (n=300) et MATH-175 (n=175), IFEval n=8 ; **J7** : fusions **M2** (transfert α=0,9) et **M3** (merge λ=0,6) produites et **vérifiées** (`post_write_verification` v2 ok, `verify_merge` VERDICT OK, 0 ULP) puis **évaluées** (6 runs, 6 h 41 de GPU, protocole figé, lot 4) ; empreintes d'items **identiques sur les 4 modèles** ; **plancher de bruit de décodage mesuré** (0,00 pt à lot 4 : le décodage est déterministe). *Verdict amendé* (M1/M2/M3, revue N2 HEAD `16a0232`) : **C2 ATTEINT** (transfert − base GSM8K : **+14,33** pts, IC [+9,33 ; +19,67] ; le papier rapporte +15,2 sur la même paire) ; **C1 NON ATTEINT** (transfert − base MATH-175 : **+9,71** pts, IC [+2,86 ; +16,57] — 0,29 pt sous le seuil de +10, IC excluant 0) ; **C3/C4 NON ÉTABLIS** (merge − instruct : −2,29 [−9,14 ; +4,57] et +1,67 [−2,33 ; +5,67] : **la dégradation de 2 points ne peut pas être exclue**). Un seul point de la règle de franchissement (§1) échoue ⇒ **G1 reste ouverte** (aucun seuil rejoué après résultats). **J8 phase A** — bilan de clôture et analyse de la troncature MATH (`modules/a2d/docs/reports/G1_BILAN.md`) : M2 épuise le budget de 512 tokens sur **82/175** générations, dont **67 sans réponse finale** (M0 : 50 ; M1 : 3 ; M3 : 55) ; sur ces 67 items, **M2 − M0 = −2,99** [−11,94 ; +5,97] alors que sur les 108 autres **M2 − M0 = +17,59** [+9,26 ; +26,85] — l'effet C1 est **dilué par le plafond de génération**, le verdict est **inchangé**. **J8 phase A2** — le diagnostic à 1 024 tokens est **annulé sur coût mesuré** (313,93 s/item ⇒ 5,8 h par modèle, au-delà du plafond de 5 h ; aucun résultat écrit) : `modules/a2d/docs/reports/G1_BILAN_1024.md`, `modules/a2d/eval/dream_a2d_trunc/ATTEMPT_1024.json`. Définition de franchissement : `modules/a2d/docs/design/G1_PROTOCOL.md` | `modules/a2d/docs/reports/J6.md`, `modules/a2d/docs/reports/J6bis.md`, `modules/a2d/docs/reports/J7.md`, `modules/a2d/docs/reports/AUDIT_J6.md`, `modules/a2d/docs/reports/AUDIT_J6BIS.md`, `modules/a2d/docs/reports/AUDIT_J6TER.md`, `modules/a2d/docs/reports/REVIEW_N2_J6.md`, `modules/a2d/docs/reports/REVIEW_N2_J6QUATER.md`, `modules/a2d/docs/reports/AUDIT_J7.md`, `modules/a2d/docs/reports/G1_BILAN.md`, `modules/a2d/docs/reports/G1_BILAN_1024.md`, `modules/a2d/docs/A2D_SPEC.md`, `modules/a2d/eval/dream_a2d/SUMMARY.md` |
| **G1b** | faisabilité Nemotron-Labs-Diffusion-3B (Ministral-3-3B) | **NON DÉMARRÉE.** Étude de faisabilité en lecture seule (aucun poids téléchargé) : identifiants/licences/taille, noms de paramètres, brouillon de mapping App. A.2, pile d'évaluation, cibles chiffrées des Tableaux 1c/17/21, conclusion GO/NO-GO | `modules/a2d/docs/design/G1B_NEMOTRON3B.md` |
| **G2** | reproduction du setup médical DiffusionGemma du papier (adapters publiés) | **DÉBLOQUÉE CÔTÉ BIBLIOTHÈQUE (J8, phase B) ; ÉVALUATION NON FAITE.** Le mode `merge-dg-scaled` (alias `adapter-scale`) écrit `base' = D_b + λ·τ_A` et laisse τ_D **non fusionné**, avec `lora_B × (1−λ)` et `lora_alpha` inchangé (App. A.2) ; vérification V5/V6 et manifeste `diffusion_adapter`. La porte elle-même reste non franchie : le décodeur DG 26B + juge 27B ne tiennent pas sur 32 Go sans quantification (protocole à déclarer). | `modules/a2d/docs/VP_ADAPTERS.md`, `modules/a2d/docs/design/G2_adapter_scaling.md`, `modules/a2d/a2d/adapters.py` |
| **G3** | raisonnement (hors papier) | **NON DÉMARRÉE** (conception d'abord) | — |

> Rappel : la conclusion « Jainius ne marche pas » (J4Bis) **n'est pas** une conclusion sur la
> méthode A2D. Le build `Jainius` est une ablation hors papier (τ_A général appliqué à un
> modèle déjà post-entraîné, coefficient non sélectionné sur ancre, évaluation non conforme).
> Les portes ci-dessus remplacent cette conclusion.

## Avant de push sur GitHub ou uploader sur HuggingFace :

### 1. Intégrité du checkpoint Jainius (build legacy)
- [ ] Les scripts legacy sont bien dans `modules/a2d/legacy/scripts/` (`merge_jainius.py`, `compute_tau_a.py`, `measure_effect.py`) et les archives dans `modules/a2d/legacy/docs/` — aucun de ces fichiers ne doit réapparaître dans `modules/a2d/cli/` ou `modules/a2d/a2d/`
- [ ] `python modules/a2d/cli/verify_checkpoint.py` passe sans erreur
- [ ] Tous les headers safetensors disent "BF16" (pas "U16")
- [ ] Aucun fichier `.safetensors` ne renvoie de JSONDecodeError au parsing

### 2. Cohérence des scripts
- [ ] `python -c "import ast,glob; [ast.parse(open(f,encoding='utf-8').read()) for f in sorted(glob.glob('academiia/**/*.py', recursive=True))+sorted(glob.glob('modules/a2d/**/*.py', recursive=True))+sorted(glob.glob('infra/vllm/**/*.py', recursive=True))]"` → OK
- [ ] Les chemins par défaut pointent vers `checkpoints/` et `_artefact/tau_a` (pas `/tmp/`)
- [ ] `modules/a2d/cli/merge_a2d.py --help` : modes `transfer|merge|merge-dg-scaled (adapter-scale)|ablation-dp-plus-tau-a|ablation-dp-minus-tau-d`,
      `--coverage {strict,zero-fill}`, `--allow-empty`, `--allow-empty-coefficient`, `--adapter-dtype`,
      α sans plafond (avertissement > 1.5), λ borné à [0,1] (avertissement > 1)
- [ ] Un merge **sans τ_D** est une **erreur dure** (sortie 2, aucune dégradation en transfert)
- [ ] Un transfert à `α=0` ou un merge à `λ=0` est **refusé** (sortie 2) tant que le plan contient
      des vecteurs non nuls ; `--allow-empty-coefficient` l'assume (avertissement au manifeste)
- [ ] `modules/a2d/legacy/scripts/merge_jainius.py --help` : défaut `--alpha 0.3`, `--on-shape-mismatch error`, manifeste `merge_config.json` écrit
- [ ] `pytest modules/a2d/tests tests -q` → tout vert, **99 tests** au total : 39 (`test_a2d_library.py`)
      + 24 (`test_a2d_mutation.py`) + 9 (`test_pipeline.py`) + 6 (`test_eval_compare.py`)
      + 7 (`test_g1_verdict.py`) + 14 (`tests/test_core.py`, code generique)
- [ ] `test_a2d_library.py` : λ=0 ⇒ D_p, λ=1 ⇒ transfert α=1,
      τ_D absent refusé, linéarité en α, LoRA ≡ `peft.merge_and_unload`, `α_eff` round-trip, refus d'écraser,
      embeddings de formes différentes, nom cible fantôme ⇒ erreur, `verify_merge` détecte un α erroné ;
      et `test_a2d_mutation.py` : **23 tests de mutation** — inversion de signe, permutation α↔λ,
      permutation τ_A↔τ_D, τ_D partiel en `strict`, τ_A absent, plan vide, tenseur non fusionné altéré,
      mode erroné, lignes d'embedding intersectées, seuils ULP BF16)

### 3. Vérification d'une fusion (nouveau, obligatoire)
- [ ] La fusion passe par `modules/a2d/cli/merge_a2d.py` (jamais une copie de la logique dans un script)
- [ ] `merge_config.json` contient : α, λ, hashes d'index **et** empreintes de contenu des shards de **toutes** les
      sources, commit git, versions, la politique de couverture et les effectifs `both` / `A-only` / `D-only`
      par famille, l'état du garde-fou « coefficient nul » (`coefficient_guard`), et par famille `‖τ_A‖`, `‖τ_D‖`,
      `cos(τ_A,τ_D)`, `ratio_max_norm_tau_over_W_global` et
      `ratio_applied_delta_norm_over_W_median` (par famille)
- [ ] `post_write_verification.ok == true` **et** `post_write_verification.version == 2` :
      coefficients **signés** (`c_a_eff`, `c_d_eff`) comparés à ceux attendus du mode, tenseurs non fusionnés
      **bit-identiques** à la référence, écart maximal à `round_bf16(expected)` **= 0 ULP**
      (`max_ulp_limit = 0` ; un écart d'1 ULP est un échec)
      (ou `post_write_verification.skipped == true` avec `reason`, si `--no-verify` a été utilisé)
- [ ] `python modules/a2d/cli/verify_merge.py --merged <out> --reference <ref> ...` → VERDICT OK
      (et `verify_merge --mode merge` **sans τ_D** → sortie 2)
- [ ] Aucun `.npy` de vecteur de tâche n'a été écrit par la bibliothèque (τ n'est jamais persisté)

### 4. État du repo git
- [ ] `git status --short` est vide (working tree clean — `.gitignore` couvre `__pycache__/`, `*.pyc`, `.venv/`, `*.npy`)
- [ ] `checkpoints/` et `_artefact/` sont bien dans `.gitignore`

### 5. Structure HuggingFace
- [ ] `checkpoints/Jainius/README.md` existe (model card)
- [ ] `config.json` contient `"architectures": ["DiffusionGemmaForBlockDiffusion"]`
- [ ] Tous les fichiers de l'original DiffusionGemma sont présents

### 6. Conformité à la source (optionnel mais recommandé)
- [ ] `python modules/a2d/cli/verify_checkpoint.py --checkpoint checkpoints/Jainius --reference checkpoints/diffusiongemma_it` passe sans erreur (noms, shapes, dtypes identiques ; sidecars présents)

### 7. Évaluation (si une évaluation est produite)
- [ ] Protocole conforme à l'App. A.1 du papier (harnais, budgets par tâche, prompt système GSM8K cité, décodage)
- [ ] Consigné : modèle, dtype/quantification, décodeur, paramètres, graine, nombre de prompts, intervalle de confiance
- [ ] Écart au papier interpreted avec l'intervalle de confiance (une sous-échantillonnage n'autorise pas un verdict « conforme »)

## Si un point échoue :
→ Ne pas push/upload. Corriger d'abord, puis relancer la checklist.
