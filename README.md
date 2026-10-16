# Jainius — DiffusionGemma optimisé par A2D

Checkpoint `Jainius` obtenu en appliquant **A2D** (AR-to-Diffusion Transfer) de [Qin et al., EPFL, 2026](docs/2610.08108v1%20-%20A2D%20(Enhancing%20diffusion%20language%20models%20with%20autoregressive%20post-training%20weights).pdf) sur **DiffusionGemma**.

## Principe

```
tau_A = Gemma4_IT − Gemma4_Base          # vecteur tâche AR post-training
Jainius = DiffusionGemma + 0.8 * tau_A_mapped
```

- **656 tenseurs** : couches decoder de Gemma 4 → DiffusionGemma, shapes alignées, τ_A appliqué (α=0.8)
- **391 tenseurs** : encoder vision, router, etc. — inchangés (pas d'équivalent dans Gemma 4)

## Workflow reproductible

```bash
# 1. Télécharger les checkpoints (≈165 Go au total)
python scripts/download_models.py --output-dir ./checkpoints

# 2. Calculer tau_A = A_p - A_b (~10 min, ~3 GB RAM peak)
python scripts/compute_tau_a.py \
    --base ./checkpoints/gemma4_base \
    --posttrained ./checkpoints/gemma4_it \
    --output-dir /tmp/a2d_jainius

# 3. Appliquer le merge A2D (~15 min, ~5 GB RAM peak)
python scripts/merge_jainius.py \
    --diffusiongemma ./checkpoints/diffusiongemma_it \
    --tau-dir /tmp/a2d_jainius \
    --output ./Jainius

# 4. Convertir en BF16 (format HuggingFace standard)
python scripts/fix_dtype.py --checkpoint ./checkpoints/Jainius
```

## Résultats

| | DiffusionGemma original | Jainius |
|--|---|---|
| Tenseurs | 1047 (11 fichiers) | 1047 (11 fichiers) |
| Taille | ~52 Go | ~48 Go |
| Format | BF16 | **BF16** ✓ |
| A2D appliqué | — | **656 tenseurs** |

## Limites

- A2D n'est appliqué que sur les 656 tenseurs decoder (les 391 tenseurs encoder vision restent inchangés)
- Le modèle n'a pas été testé en inference (nécessite un GPU dédié)
- α=0.8 est une valeur intermédiaire du papier A2D

## Références

- **A2D** : Qin et al., "Enhancing Diffusion Language Models with Autoregressive Post-Training Weights", arXiv:2610.08108, 2026
- **DiffusionGemma** : Google DeepMind, "DiffusionGemma Technical Report", arXiv:2608.00146, 2026
