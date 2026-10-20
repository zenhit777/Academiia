# CHECKLIST.md — Vérifications avant de considérer le projet "prêt"

## État des portes (au 2026-10-20)

| Porte | Contenu | État | Preuve |
|---|---|---|---|
| **J4Bis** | infrastructure GPU/service (compose, service FP8, smoke) | **OK (infra)** — validée le 2026-10-19 | `vLLM/RAPPORT_J4BIS-BIS.md` |
| **J5** | campagne d'évaluation par génération (103 prompts) | **NON CONFORME** au protocole du papier (FP8, `temperature=0`, 128 tokens, texte libre, sans juge) : contrôle de non-régression, pas une mesure de qualité | `eval/RESULTS.md` |
| **G1** | mécanique A2D validée sur Qwen2.5-7B → Dream-7B (Tableaux 6, 16, 21) | **ouverte, critères de reproduction atteints** — géométrie (Table 21) reproduite à ≤ 0,11 % ; baseline GSM8K et MATH-500 du Base reproduites (cible du papier dans l'IC 95 %) ; fusion réelle α=0.9 vérifiée (`α_eff` = 0,913, couvert par la borne BF16) | `docs/reports/J6.md`, `docs/A2D_SPEC.md`, `eval/dream_a2d/SUMMARY.md` |
| **G2** | reproduction du setup médical DiffusionGemma du papier (adapters publiés) | **non entamée** (hors périmètre J6) | `docs/VP_ADAPTERS.md` |
| **G3** | raisonnement (hors papier) | **non entamée** (conception d'abord) | — |

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
- [ ] `scripts/merge_a2d.py --help` : modes `transfer|merge|ablation-dp-plus-tau-a|ablation-dp-minus-tau-d`, α/λ sans plafond, avertissement > 1.5
- [ ] `scripts/merge_jainius.py --help` : défaut `--alpha 0.3`, `--on-shape-mismatch error`, manifeste `merge_config.json` écrit
- [ ] `pytest scripts/tests -q` → tout vert (dont `test_a2d_library.py` : λ=0 ⇒ D_p, λ=1 ⇒ transfert α=1,
      τ_D absent refusé, linéarité en α, LoRA ≡ `peft.merge_and_unload`, `α_eff` round-trip, refus d'écraser,
      embeddings de formes différentes, nom cible fantôme ⇒ erreur, `verify_merge` détecte un α erroné)

### 3. Vérification d'une fusion (nouveau, obligatoire)
- [ ] La fusion passes par `scripts/merge_a2d.py` (jamais une copie de la logique dans un script)
- [ ] `merge_config.json` contient : α, λ, hashes d'index de **toutes** les sources, commit git, versions,
      et par famille de tenseurs `‖τ_A‖`, `‖τ_D‖`, `cos(τ_A,τ_D)`, `‖τ‖/‖W‖` (global **et** médiane par tenseur)
- [ ] `post_write_verification.ok == true` (`α_eff` / `λ_eff` dans la tolérance)
- [ ] `python scripts/verify_merge.py --merged <out> --reference <ref> ...` → VERDICT OK
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
