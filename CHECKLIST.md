# CHECKLIST.md — Vérifications avant de considérer le projet "prêt"

## État des portes (au 2026-10-20, après revue N2 et mission J6bis)

| Porte | Contenu | État | Preuve |
|---|---|---|---|
| **J4Bis** | infrastructure GPU/service (compose, service FP8, smoke) | **OK (infra)** — validée le 2026-10-19 | `vLLM/RAPPORT_J4BIS-BIS.md` |
| **J5** | campagne d'évaluation par génération (103 prompts) | **NON CONFORME** au protocole du papier (FP8, `temperature=0`, 128 tokens, texte libre, sans juge) : contrôle de non-régression, pas une mesure de qualité | `eval/RESULTS.md` |
| **G1** | mécanique A2D sur Qwen2.5-7B → Dream-7B (Tableaux 6, 16, 21) | **OUVERTE.** *Fait et mesuré* : géométrie Table 21 (écarts ≤ 0,11 %) ; baseline D_b/D_p **exécutée** sur GSM8K (n=300) et MATH-175 (n=175), IFEval n=8 (contrôle de fonctionnement) ; fusion transfert α=0,9 **vérifiée mécaniquement** (`α_eff` = 0,913, borne BF16). *Non atteint* : critère de reproduction ±3 pts (Base MATH-175 **+3,2** ; IFEval n=8 non concluant) ; **aucun modèle fusionné évalué** ⇒ la bibliothèque n'est validée que mécaniquement. **Bibliothèque prête pour J7** (fusions M2/M3 + évaluations G1 selon `docs/design/G1_PROTOCOL.md`) : audits N1 versionnés (`docs/reports/AUDIT_J6BIS.md`, `docs/reports/AUDIT_J6TER.md`), **résidu F5 clos (J6quater)**, garde-fou « coefficient nul » actif, `--no-verify` tracé — **aucune fusion ni évaluation GPU n'a été lancée**. Définition de franchissement : `docs/design/G1_PROTOCOL.md` | `docs/reports/J6.md`, `docs/reports/J6bis.md`, `docs/reports/AUDIT_J6.md`, `docs/reports/AUDIT_J6BIS.md`, `docs/reports/AUDIT_J6TER.md`, `docs/reports/REVIEW_N2_J6.md`, `docs/A2D_SPEC.md`, `eval/dream_a2d/SUMMARY.md` |
| **G1b** | faisabilité Nemotron-Labs-Diffusion-3B (Ministral-3-3B) | **NON DÉMARRÉE.** Étude de faisabilité en lecture seule (aucun poids téléchargé) : identifiants/licences/taille, noms de paramètres, brouillon de mapping App. A.2, pile d'évaluation, cibles chiffrées des Tableaux 1c/17/21, conclusion GO/NO-GO | `docs/design/G1B_NEMOTRON3B.md` |
| **G2** | reproduction du setup médical DiffusionGemma du papier (adapters publiés) | **BLOQUÉE (B7).** Les adaptateurs de diffusion DG doivent rester **non fusionnés** et être **mis à l'échelle** (App. A.2) ; la bibliothèque fusionne aujourd'hui τ_D dans les poids, ce qui ne serait pas conforme. Conception (aucun code) : `docs/design/G2_adapter_scaling.md` | `docs/VP_ADAPTERS.md`, `docs/design/G2_adapter_scaling.md` |
| **G3** | raisonnement (hors papier) | **NON DÉMARRÉE** (conception d'abord) | — |

> Rappel : la conclusion « Jainius ne marche pas » (J4Bis) **n'est pas** une conclusion sur la
> méthode A2D. Le build `Jainius` est une ablation hors papier (τ_A général appliqué à un
> modèle déjà post-entraîné, coefficient non sélectionné sur ancre, évaluation non conforme).
> Les portes ci-dessus remplacent cette conclusion.

## Avant de push sur GitHub ou uploader sur HuggingFace :

### 1. Intégrité du checkpoint Jainius (build legacy)
- [ ] `python scripts/verify_checkpoint.py` passe sans erreur
- [ ] Tous les headers safetensors disent "BF16" (pas "U16")
- [ ] Aucun fichier `.safetensors` ne renvoie de JSONDecodeError au parsing

### 2. Cohérence des scripts
- [ ] `python -c "import ast,glob; [ast.parse(open(f,encoding='utf-8').read()) for f in sorted(glob.glob('scripts/**/*.py', recursive=True))]"` → OK
- [ ] Les chemins par défaut pointent vers `checkpoints/` et `_artefact/tau_a` (pas `/tmp/`)
- [ ] `scripts/merge_a2d.py --help` : modes `transfer|merge|ablation-dp-plus-tau-a|ablation-dp-minus-tau-d`,
      `--coverage {strict,zero-fill}`, `--allow-empty`, `--allow-empty-coefficient`, α sans plafond
      (avertissement > 1.5), λ borné à [0,1] (avertissement > 1)
- [ ] Un transfert à `α=0` ou un merge à `λ=0` est **refusé** (sortie 2) tant que le plan contient
      des vecteurs non nuls ; `--allow-empty-coefficient` l'assume (avertissement au manifeste)
- [ ] `scripts/merge_jainius.py --help` : défaut `--alpha 0.3`, `--on-shape-mismatch error`, manifeste `merge_config.json` écrit
- [ ] `pytest scripts/tests -q` → tout vert, **65 tests** au total : 29 (`test_a2d_library.py`)
      + 21 (`test_a2d_mutation.py`) + 9 (`test_pipeline.py`) + 6 (`test_eval_compare.py`)
- [ ] `test_a2d_library.py` : λ=0 ⇒ D_p, λ=1 ⇒ transfert α=1,
      τ_D absent refusé, linéarité en α, LoRA ≡ `peft.merge_and_unload`, `α_eff` round-trip, refus d'écraser,
      embeddings de formes différentes, nom cible fantôme ⇒ erreur, `verify_merge` détecte un α erroné ;
      et `test_a2d_mutation.py` : **21 tests de mutation** — inversion de signe, permutation α↔λ,
      permutation τ_A↔τ_D, τ_D partiel en `strict`, τ_A absent, plan vide, tenseur non fusionné altéré,
      mode erroné, lignes d'embedding intersectées, seuils ULP BF16)

### 3. Vérification d'une fusion (nouveau, obligatoire)
- [ ] La fusion passe par `scripts/merge_a2d.py` (jamais une copie de la logique dans un script)
- [ ] `merge_config.json` contient : α, λ, hashes d'index **et** empreintes de contenu des shards de **toutes** les
      sources, commit git, versions, la politique de couverture et les effectifs `both` / `A-only` / `D-only`
      par famille, l'état du garde-fou « coefficient nul » (`coefficient_guard`), et par famille `‖τ_A‖`, `‖τ_D‖`,
      `cos(τ_A,τ_D)`, `‖τ‖/‖W‖` (global **et** médiane par tenseur)
- [ ] `post_write_verification.ok == true` **et** `post_write_verification.version == 2` :
      coefficients **signés** (`c_a_eff`, `c_d_eff`) comparés à ceux attendus du mode, tenseurs non fusionnés
      **bit-identiques** à la référence, écart maximal à `round_bf16(expected)` ≤ 1 ULP
      (ou `post_write_verification.skipped == true` avec `reason`, si `--no-verify` a été utilisé)
- [ ] `python scripts/verify_merge.py --merged <out> --reference <ref> ...` → VERDICT OK
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
- [ ] `python scripts/verify_checkpoint.py --checkpoint checkpoints/Jainius --reference checkpoints/diffusiongemma_it` passe sans erreur (noms, shapes, dtypes identiques ; sidecars présents)

### 7. Évaluation (si une évaluation est produite)
- [ ] Protocole conforme à l'App. A.1 du papier (harnais, budgets par tâche, prompt système GSM8K cité, décodage)
- [ ] Consigné : modèle, dtype/quantification, décodeur, paramètres, graine, nombre de prompts, intervalle de confiance
- [ ] Écart au papier interpreted avec l'intervalle de confiance (une sous-échantillonnage n'autorise pas un verdict « conforme »)

## Si un point échoue :
→ Ne pas push/upload. Corriger d'abord, puis relancer la checklist.
