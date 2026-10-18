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
 - Chargement et démarrage vLLM validés le 2026-10-18 (WSL2, runner V1, `--enforce-eager`, offload 30 Go) ; génération et qualité non encore évaluées.
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

### `vLLM/diagnose_image.sh` — Vérification d'une image d'entrée

Vérifie qu'une image est valide avant de l'utiliser comme entrée au modèle :

```bash
./vLLM/diagnose_image.sh /path/to/image.png [width] [height]
```

**Sortie :** JSON compact sur `stdout` avec le format, les dimensions et la taille du fichier. Exemple :

```json
{"valid":true,"width":512,"height":512,"format":"PNG","size_kb":42}
```

**Défauts détectés :** fichier introuvable, format non supporté (seulement PNG/WEBP), dimensions manquantes ou non détectables.

### `vLLM/smoke_chat.py` — Chat de fumée avec DiffusionGemma + Jainius

Test complet d'inférence : charge le checkpoint fusionné, encode une image en tokens visuels, génère une réponse textuelle.

```bash
python vLLM/smoke_chat.py /path/to/image.png \
    --prompt "Quel est l'objet principal de cette image ?" \
    --max-tokens 128 \
    --image-width 560
```

**Dépendances :** `pyyaml>=6.0`, `pytest>=8.0` (ajoutés à `requirements.txt`).

**Sortie :** JSONL avec la réponse générée, prête à être inspectée ou loguée.

### Comparaison des scripts vLLM

| Script | Usage | Sortie |
|--------|-------|--------|
| `smoke.sh` (legacy) | Chat brut, sortie raw | texte brut |
| `smoke_chat.py` | Chat complet avec prompt + image, JSONL | JSONL |
| `diagnose_image.sh` | Validation d'entrée image | JSON compact |

**Conseil :** utiliser `smoke_chat.py` pour les tests de non-régression automatisés (ex. CI).

---


## Références

- **A2D** : Qin et al., "Enhancing Diffusion Language Models with Autoregressive Post-Training Weights", arXiv:2610.08108, 2026
- **DiffusionGemma** : Google DeepMind, "DiffusionGemma Technical Report", arXiv:2608.00146, 2026
