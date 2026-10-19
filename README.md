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
 # 1. Télécharger les checkpoints (~155 Go au total, mesuré : 3 × ~51.6 Go)
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

## Résultats

| | DiffusionGemma original | Jainius (rebuild v2 2026-10-16) |
|--|---|---|
| Tenseurs | 1047 (11 fichiers) | 1047 (11 fichiers) |
| Taille | ~52 Go | ~51.6 Go |
| Format | BF16 | **BF16** ✓ (strict, vérifié) |
| A2D appliqué | — | **656 tenseurs** (391 inchangés, bit-identiques) |

## Limites

 - A2D n'est appliqué que sur les 656 tenseurs decoder (les 391 tenseurs encoder vision restent inchangés). Contrairement au papier (embeddings/norms/output-head inchangés), le mapping par défaut inclut `embed_tokens` et les normalisations — voir variante `--include-norms-embed` (papier) et `scripts/measure_effect.py`.
 - vLLM (WSL2, 2026-10-18) : avec le runner V1 forcé (`VLLM_USE_V2_MODEL_RUNNER=0`) + `--enforce-eager` + offload 30 Go, le serveur démarre et charge Jainius, mais la première génération échoue (assert GPU scatter/gather). Cause probable non prouvée : incompatibilité du runner V1 avec la diffusion. Le compose force désormais le runner V2 (défaut vLLM) — à valider sur GPU, avec témoin `D_p`. Voir `vLLM/KNOWN_ISSUES.md`.
- α=0.8 était le défaut historique (rebuild v2) mais dégrade SLAKE de 3.8 pts dans le papier (83.1 → 79.3) pour `Dp + ατA`. Le papier retient 0.1–0.3 en merge et jusqu'à 1.0 en transfert de domaine — d'où le balayage 0.05–0.5 recommandé et le défaut actuel 0.3. Aucun gain démontré à ce jour : `D_p` reste le témoin.
- Poids dérivés de Gemma : soumis aux conditions d'utilisation Gemma (à rappeler dans la model card avant publication).
 ## Mesure Dp + α·τ_A après arrondi BF16 (v2, 2026-10-17)

 Le script `scripts/measure_effect.py` rapporte, en plus du % d'éléments modifiés,
 l'ampleur relative du delta (`ratio_rel = ||α·τ|| / ||Dp||`) et la fidélité de
 l'arrondi BF16 (`cosinus(applied, α·τ)`). Résultats α=0.3 sur les 656 tenseurs :

 | Famille | ratio_rel médiane | cosinus (pondéré taille) | % modifié |
 |---------|-------------------|--------------------------|-----------|
 | attn    | 2,98 %            | 0,9984                   | 99,1 %    |
 | mlp/experts | 2,82 %        | 0,9984                   | 99,1 %    |
 | embed   | 2,66 %            | 0,9980                   | —         |
 | norms   | 0,044 %           | 0,606                    | —         |
 | router  | ≈ 0               | non mesuré (τ≈0)         | —         |

 Le `ratio_rel` médian est de ~3 % : le delta théorique représente une fraction
 minime du module Dp, même si la plupart des éléments binaires changent (l'ULP BF16
 est minuscule). Le cosinus des normes = 0,61 montre que l'arrondi BF16 détruit
 la direction du delta normatif ; les embeddings et attn/mlp sont préservés à >99 %.
 Ces chiffres ne permettent pas de conclure sur la qualité d'inférence : seule une
 évaluation (perplexité, bench) peut le faire.
## Scripts vLLM (WSL2)

### `vLLM/smoke_chat.py` — Test de génération via l'API vLLM

Client HTTP **stdlib uniquement** (aucune dépendance, ne charge pas le modèle) :
`POST /v1/chat/completions` sur un serveur vLLM déjà lancé, contrôles objectifs
(HTTP 200, texte non vide, pas de répétition dégénérée évidente), sauvegarde
`eval/smoke_<modèle>_<horodatage>.json` :

```bash
python vLLM/smoke_chat.py --url http://localhost:8000 --model jainius
```

### `vLLM/diagnose_image.sh` — Diagnostic de l'image Docker vLLM

Sans GPU ni chargement de modèle : version de vLLM dans l'image, présence de
`VLLM_WSL2_ENABLE_PIN_MEMORY` / `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA` /
`VLLM_USE_V2_MODEL_RUNNER` dans `envs.py`, de `WSL2_ENABLE_PIN_MEMORY` dans
`platforms/cuda.py`, usage de `req_states` dans `diffusion_gemma.py` :

```bash
cd vLLM && ./diagnose_image.sh [image]
```

### Comparaison des scripts vLLM

| Script | Usage | Sortie |
|--------|-------|--------|
| `smoke_chat.py` | Génération contrôlée (référence) | texte + JSON sauvegardé |
| `diagnose_image.sh` | Diagnostic image Docker sans GPU | rapport stdout |
| `smoke.sh` | Health-check curl rapide | texte brut |

---


## Références

- **A2D** : Qin et al., "Enhancing Diffusion Language Models with Autoregressive Post-Training Weights", arXiv:2610.08108, 2026
- **DiffusionGemma** : Google DeepMind, "DiffusionGemma Technical Report", arXiv:2608.00146, 2026
