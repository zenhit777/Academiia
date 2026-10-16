# CLAUDE.md — Instructions critiques pour Claude Code

## ⚠️ ANOMALIES CONNUES (à vérifier AVANT toute action)

### Checkpoint Jainius (`_artefact/checkpoints/Jainius/`)
- **Header corrompu** : `model-00001-of-00011.safetensors` peut avoir un header JSON parse error.
  → Solution : relancer `python scripts/fix_dtype.py --checkpoint _artefact/checkpoints/Jainius`
- **Dtype U16 vs BF16** : certains fichiers peuvent afficher "U16" au lieu de "BF16".
  → C'est normal après le fix (le binaire est correct, seul le header string change).

### Avant de considérer le projet "prêt" :
1. `python scripts/verify_checkpoint.py` — vérifie l'intégrité du checkpoint
2. Vérifier que tous les headers safetensors disent "BF16"
3. Ne jamais modifier directement les fichiers `.safetensors` dans `_artefact/`

## Contexte projet

**Academiia** : application A2D (AR-to-Diffusion Transfer) à DiffusionGemma.
- τ_A = Gemma4_IT − Gemma4_Base → Jainius = D_p + 0.8 × τ_A_mapped
- Checkpoint final : `_artefact/checkpoints/Jainius/` (~45 Go, BF16)

## Structure
```
Academiia/
├── CLAUDE.md          ← Tu es ici (critique)
├── AGENTS.md          ← Instructions générales agents IA
├── .cursorrules       ← Règles Cursor IDE
├── scripts/           ← 4 scripts reproductibles
├── _artefact/checkpoints/ ← Checkpoints (~245 Go, non versionnés)
└── docs/              ← Prépublication A2D (PDF)
```

## Workflow
```bash
python scripts/download_models.py --output-dir ./_artefact/checkpoints
python scripts/compute_tau_a.py --base _artefact/checkpoints/gemma4_base \
    --posttrained _artefact/checkpoints/gemma4_it --output-dir /tmp/a2d_jainius
python scripts/merge_jainius.py --diffusiongemma _artefact/checkpoints/diffusiongemma_it \
    --tau-dir /tmp/a2d_jainius --output _artefact/checkpoints/Jainius
python scripts/fix_dtype.py --checkpoint _artefact/checkpoints/Jainius
```

## Règles de travail
- Ne jamais commiter `_artefact/` (dans .gitignore)
- Pour HuggingFace : `huggingface-cli upload <user>/Jainius ./_artefact/checkpoints/Jainius --repo-type model`
- Les scripts sont la source de vérité — ne pas réécrire la logique A2D
