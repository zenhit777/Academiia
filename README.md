# Jainius — DiffusionGemma + vecteur instruction IT (ablation `Dp + ατA`, inspirée A2D)

Checkpoint `Jainius` obtenu en appliquant à **DiffusionGemma** le vecteur
tâche **τ_A = Gemma4_IT − Gemma4_Base**, suivant l'ablation « Dp + ατA »
du papier [Qin et al., EPFL, 2026](docs/2610.08108v1%20-%20A2D%20(Enhancing%20diffusion%20language%20models%20with%20autoregressive%20post-training%20weights).pdf).
Ce N'EST PAS le A2D-Merge complet du papier (`D_p + λ(τ_A − τ_D)`) :
celui-ci exigerait τ_D (base diffusion publique, inexistante). Le nom
« optimisé par A2D » des versions précédentes était donc abusif.

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
    --output-dir _artefact/tau_a

# 3. Mesurer l'effet reel apres arrondi BF16 (CPU seul, recommande avant merge)
python scripts/measure_effect.py \
    --diffusiongemma ./checkpoints/diffusiongemma_it \
    --tau-dir _artefact/tau_a \
    --alphas 0.05,0.1,0.2,0.3,0.5,0.8

# 4. Appliquer le merge Dp + alpha*tau_A (~15 min, ~5 GB RAM peak)
python scripts/merge_jainius.py \
    --diffusiongemma ./checkpoints/diffusiongemma_it \
    --tau-dir _artefact/tau_a \
    --output ./checkpoints/Jainius \
    --alpha 0.3

# 5. Verifier le checkpoint (format HuggingFace strict)
python scripts/verify_checkpoint.py --checkpoint ./checkpoints/Jainius
```
(`scripts/fix_dtype.py` est deprecie : le merge sauvegarde deja en BF16
canonique. Conserve uniquement pour reparer d'anciens checkpoints corrompus.)

## Résultats

| | DiffusionGemma original | Jainius (rebuild v2 2026-10-16) |
|--|---|---|
| Tenseurs | 1047 (11 fichiers) | 1047 (11 fichiers) |
| Taille | ~52 Go | ~51.6 Go |
| Format | BF16 | **BF16** ✓ (strict, vérifié) |
| A2D appliqué | — | **656 tenseurs** (391 inchangés, bit-identiques) |

## Limites

- A2D n'est appliqué que sur les 656 tenseurs decoder (les 391 tenseurs encoder vision restent inchangés). Contrairement au papier (embeddings/norms/output-head inchangés), le mapping par défaut inclut `embed_tokens` et les normalisations — voir variante `--exclude-norms-embed` (papier) et `scripts/measure_effect.py`.
- Le modèle n'a pas été testé en inference GPU dédiée (voir `vLLM/` : BF16 + `--cpu-offload-gb 30` sur 5090 32 Go, ou test INT8 rapide `scripts/smoke_int8.py`).
- α=0.8 était le défaut historique (rebuild v2) mais dégrade SLAKE de 3.8 pts dans le papier (83.1 → 79.3) pour `Dp + ατA`. Le papier retient 0.1–0.3 en merge et jusqu'à 1.0 en transfert de domaine — d'où le balayage 0.05–0.5 recommandé et le défaut actuel 0.3. Aucun gain démontré à ce jour : `D_p` reste le témoin.
- Poids dérivés de Gemma : soumis aux conditions d'utilisation Gemma (à rappeler dans la model card avant publication).

## Références

- **A2D** : Qin et al., "Enhancing Diffusion Language Models with Autoregressive Post-Training Weights", arXiv:2610.08108, 2026
- **DiffusionGemma** : Google DeepMind, "DiffusionGemma Technical Report", arXiv:2608.00146, 2026
