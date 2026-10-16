# CHECKLIST.md — Vérifications avant de considérer le projet "prêt"

## Avant de push sur GitHub ou uploader sur HuggingFace :

### 1. Intégrité du checkpoint Jainius
- [ ] `python scripts/verify_checkpoint.py` passe sans erreur
- [ ] Tous les headers safetensors disent "BF16" (pas "U16")
- [ ] Aucun fichier `.safetensors` ne renvoie de JSONDecodeError au parsing

### 2. Cohérence des scripts
- [ ] `python -c "import ast; [ast.parse(open(f'scripts/{f}').read()) for f in ['download_models.py','compute_tau_a.py','merge_jainius.py','fix_dtype.py','verify_checkpoint.py']]"` → OK
- [ ] Les chemins par défaut pointent vers `checkpoints/` et `_artefact/tau_a` (pas `/tmp/`)

### 3. État du repo git
- [ ] `git status --short` est vide (working tree clean)
- [ ] `checkpoints/` et `_artefact/` sont bien dans `.gitignore`

### 4. Structure HuggingFace
- [ ] `checkpoints/Jainius/README.md` existe (model card)
- [ ] `config.json` contient `"architectures": ["DiffusionGemmaForBlockDiffusion"]`
- [ ] Tous les fichiers de l'original DiffusionGemma sont présents

## Si un point échoue :
→ Ne pas push/upload. Corriger d'abord, puis relancer la checklist.
