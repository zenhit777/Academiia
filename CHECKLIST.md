# CHECKLIST.md — Vérifications avant de considérer le projet "prêt"

## Avant de push sur GitHub ou uploader sur HuggingFace :

### 1. Intégrité du checkpoint Jainius
- [ ] `python scripts/verify_checkpoint.py` passe sans erreur
- [ ] Tous les headers safetensors disent "BF16" (pas "U16")
- [ ] Aucun fichier `.safetensors` ne renvoie de JSONDecodeError au parsing

### 2. Cohérence des scripts
- [ ] `python -c "import ast,glob; [ast.parse(open(f,encoding='utf-8').read()) for f in sorted(glob.glob('scripts/*.py'))]"` → OK
- [ ] Les chemins par défaut pointent vers `checkpoints/` et `_artefact/tau_a` (pas `/tmp/`)
- [ ] `scripts/merge_jainius.py --help` : défaut `--alpha 0.3`, `--on-shape-mismatch error`, manifest `merge_config.json` écrit

### 3. État du repo git
- [ ] `git status --short` est vide (working tree clean — `.gitignore` couvre `__pycache__/`, `*.pyc`, `.venv/`, `*.npy`)
- [ ] `checkpoints/` et `_artefact/` sont bien dans `.gitignore`

### 4. Structure HuggingFace
- [ ] `checkpoints/Jainius/README.md` existe (model card)
- [ ] `config.json` contient `"architectures": ["DiffusionGemmaForBlockDiffusion"]`
- [ ] Tous les fichiers de l'original DiffusionGemma sont présents
 ### 5. Conformité à la source (optionnel mais recommandé)
 - [ ] `python scripts/verify_checkpoint.py --checkpoint checkpoints/Jainius --reference checkpoints/diffusiongemma_it` passe sans erreur (noms, shapes, dtypes identiques ; sidecars présents).

## Si un point échoue :
→ Ne pas push/upload. Corriger d'abord, puis relancer la checklist.
