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
Jainius = DiffusionGemma + 0.3 * tau_A_mapped
```

- **α = 0.3** (défaut ; 0.8 est déprécié — il produit un texte dégénéré, cf. « Évaluation par génération »)
- **355 tenseurs** : projections decoder de Gemma 4 → DiffusionGemma, shapes alignées, τ_A appliqué
- **692 tenseurs** : inchangés (embeddings, normalisations, routeur MoE, encoder vision)
- Variante `--include-norms-embed` : 656 tenseurs fusionnés / 391 inchangés (embeddings et norms inclus)

## Workflow reproductible

```bash
 # 1. Télécharger les checkpoints (~155 Go au total, mesuré : 3 × ~51.6 Go)
python scripts/download_models.py --output-dir ./checkpoints

# 2. Calculer tau_A = A_p - A_b (~10 min, ~3 GB RAM peak)
#    ATTENTION : safetensors pré-fault la projection du shard AR de 46,5 GiB ->
#    il faut >= 50 Go de RAM disponibles (sur WSL2 : memory= dans .wslconfig)
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

| | DiffusionGemma original (D_p) | Jainius (α=0.3, 2026-10-19) |
|--|---|---|
| Tenseurs | 1047 (11 fichiers) | 1047 (11 fichiers) |
| Format | BF16 | **BF16** ✓ (strict : 11/11 shards, en-têtes « BF16 ») |
| Conformité à la source | — | ✓ (`verify_checkpoint.py --reference` : 1047 tenseurs, 0 écart de shape/dtype) |
| A2D appliqué | — | **355 tenseurs** (692 inchangés, bit-identiques) |
| Provenance | — | `merge_config.json` (α, variante, date, commit) |

## Limites

 - A2D n'est appliqué qu'aux projections decoder (355 tenseurs sur 1047) : embeddings, normalisations, routeur MoE et encoder vision restent ceux de D_p. C'est la variante papier (embeddings/norms inchangés) ; la variante `--include-norms-embed` fusionne 656 tenseurs (391 inchangés) et a été mesurée elle aussi (voir « Évaluation par génération »).
 - **Aucun gain démontré** : sur 103 prompts, Jainius (α=0.3) est moins bon que D_p sur la justesse des réponses (88,1 % contre 100 %) et 5,5× plus lent. `D_p` reste le témoin ; le merge A2D tel qu'implémenté ici dégrade le modèle de diffusion.
 - α=0.8 (défaut historique, rebuild v2) ne produit pas un texte cohérent : il perturbe les projections d'attention de ~50 % de leur norme. α=0.3 est le coefficient retenu.
 - Les mesures de génération ont été faites en **FP8** (quantification à la volée) : c'est la seule configuration qui tient dans 32 Go de VRAM, le modèle en BF16 faisant 48,1 GiB. Le comportement en BF16 n'est pas mesuré. Le runner v1 est inutilisable (crash scatter CUDA, cf. `vLLM/KNOWN_ISSUES.md`).
- `tau_A` ne peut pas être recalculé sur cette machine : `safetensors.safe_open` pré-fault la projection du shard de 46,5 GiB des checkpoints AR, au-delà des 30 GiB de RAM de la VM WSL2 (ENOMEM). Augmenter `memory=` dans `%UserProfile%\.wslconfig` (≥ 64 Go) débloque l'étape 2 du workflow.
- Poids dérivés de Gemma : soumis aux conditions d'utilisation Gemma (https://ai.google.dev/gemma/docs/gemma_4_license). Les repos sources (`google/gemma-4-26B-A4B`, `google/gemma-4-26B-A4B-it`, `google/diffusiongemma-26B-A4B-it`) ne publient **aucun fichier LICENSE** (vérifié le 2026-10-19) : rien à recopier, la model card déclare `apache-2.0` + le lien des conditions Gemma.

## Évaluation par génération (2026-10-19)

Serveur vLLM 0.24.0 (image épinglée par digest), RTX 5090 32 Go, WSL2/Ubuntu 24.04,
`--quantization fp8`, `--max-model-len 2048`, `--max-num-seqs 1`, `--enforce-eager`,
`max_tokens=128`, `temperature=0`, 103 prompts (`eval/prompts.jsonl` : faits FR/EN,
maths, formats de sortie, questions libres).

| Métrique | D_p (témoin) | Jainius α=0.3 | Jainius α=0.3 + norms/embed | Jainius α=0.8 (ancien build) |
|---|---|---|---|---|
| Requêtes OK | 103/103 | 103/103 | 3/3 (smoke) | HTTP 200 mais texte dégénéré |
| Réponses non vides | 100 % | 100 % | 100 % | 100 % |
| Répétitions dégénérées | 0 % | 0 % | 0 % | 100 % |
| Valeur attendue présente | **100 %** | 88,1 % (faits 83,3 %, maths 100 %) | — (smoke) | — |
| Format respecté | 93,8 % | 75,0 % | — | — |
| Latence moyenne | 405 ms | 2 120 ms | ≈ 2 100 ms | — |
| Débit | 116,4 tok/s | 21,1 tok/s | ≈ 20 tok/s | — |

Lecture : α=0.3 rend un texte cohérent (α=0.8 non), mais **n'améliore pas** le
témoin — les erreurs restantes sont des corruptions de tokens (« Can로 » pour
« Canberra », « New area » pour « New Delhi »), symptomatiques d'un modèle de
diffusion moins « peaked ». Le ralentissement d'un facteur ~5,5 est cohérent avec
le sampler's data-dependent (arrêt anticipé sur `confidence_threshold` /
`stability_threshold`, `max_denoising_steps=48` par défaut) : moins confiant ⇒
plus d'itérations de débruitage par token. Le sampler étant stochastique même à
`temperature=0`, la justesse de Jainius varie d'une exécution à l'autre
(91,0 % puis 88,1 % sur deux passes) ; `D_p` est resté à 100 %.

Détail des métriques et réponses : `eval/results_Dp.json`,
`eval/results_jainius_a03.json`, rapport `eval/RESULTS.md`.
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
