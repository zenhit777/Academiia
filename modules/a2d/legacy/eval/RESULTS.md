# Résultats d'évaluation par génération — 2026-10-19

Comparaison du checkpoint livré `Jainius` (α=0.3) avec son témoin `D_p`
(`checkpoints/diffusiongemma_it`), en génération réelle via vLLM.

## Protocole

| Paramètre | Valeur |
|---|---|
| Serveur | vLLM 0.24.0, image `vllm/vllm-openai@sha256:251eba5cc7c1…` (épinglée par digest) |
| Matériel | RTX 5090 32 607 MiB, WSL2 / Ubuntu 24.04, 30 GiB de RAM hôte |
| Flags | `--quantization fp8` (quantification à la volée des poids BF16), `--cpu-offload-gb 0`, `--max-model-len 2048`, `--max-num-seqs 1`, `--gpu-memory-utilization 0.92`, `--enforce-eager` |
| Client | `scripts/eval_compare.py` (stdlib), `eval/prompts.jsonl` — 103 prompts (faits FR/EN, maths, formats, questions libres) |
| Génération | `max_tokens=128`, `temperature=0` |
| Un modèle à la fois | oui (48,1 GiB de poids : impossible de servir les deux en même temps) |

Les métriques sont calculées à partir des réponses brutes du serveur
(`eval/results_Dp.json`, `eval/results_jainius_a03.json`) : latence, débit,
réponses non vides, répétitions dégénérées (heuristique de `vLLM/smoke_chat.py`),
présence de la valeur attendue, respect du format demandé.

## Résultats

| Métrique | D_p (témoin) | Jainius α=0.3 |
|---|---|---|
| Requêtes HTTP 200 | 103/103 | 103/103 |
| Réponses non vides | 100 % | 100 % |
| Répétitions dégénérées | 0 % | 0 % |
| Valeur attendue présente | **100 %** (67/67) | 88,1 % (59/67) — faits 83,3 %, maths 100 % |
| Format respecté | 93,8 % (15/16) | 75,0 % (12/16) |
| Latence moyenne / médiane / p95 | 405 / 317 / 925 ms | 2 120 / 2 207 / 2 232 ms |
| ms par token | 8,6 | 47,3 |
| Débit | 116,4 tok/s | 21,1 tok/s |
| Tokens générés (total) | 4 850 | 4 613 |

### Par catégorie

| Catégorie | n | Attendu (D_p) | Attendu (Jainius) | Format (D_p) | Format (Jainius) |
|---|---|---|---|---|---|
| faits | 48 | 100 % | 83,3 % | — | — |
| maths | 19 | 100 % | 100 % | — | — |
| format | 16 | — | — | 93,8 % | 75,0 % |
| questions libres | 20 | — | — | — | — |

> **Variabilité entre exécutions** : le sampler de diffusion est stochastique même à
> `temperature=0`. Deux passes successives du même checkpoint Jainius ont donné
> 91,0 % puis 88,1 % de justesse (textes différents pour les mêmes prompts) ; le
> témoin `D_p` a donné 100 % lors des deux passes. Les chiffres ci-dessus sont
> ceux de la seconde passe (`eval/results_*.json`), exécutée avec la version
> corrigée de `scripts/eval_compare.py` (le contrôle « one_line » ne comptait
> pas le préfixe « thought » du modèle de diffusion, ce qui faisait échouer à tort
> 3 prompts `one_line` de Jainius et 4 de `D_p` lors de la première passe).

## Ce que l'on peut conclure

1. **α=0.3 ne dégrade pas le modèle en dégénérescence** : 0 % de répétitions
   dégénérées, 100 % de réponses non vides, justesse mathématique intacte
   (100 %). Le checkpoint α=0.8 qui était auparavant livré produisait, lui, un
   texte vide de sens (100 % de répétitions).
2. **Aucun gain** : sur les mêmes 103 prompts, `D_p` fait mieux partout
   (100 % contre 88,1 % de justesse, 93,8 % contre 75,0 % de format). Le merge
   `D_p + ατ_A` tel qu'implémenté ici **dégrade** le modèle de diffusion ; le
   témoin `D_p` reste la référence.
3. **Nature des erreurs restantes** : ce sont des corruptions de tokens, pas des
   erreurs de raisonnement — « La capitale de l'Australie est **Can로** » pour
   Canberra, « The capital of India is **New area** » pour New Delhi, « The play
   * way* was » pour *Hamlet*. Un modèle de diffusion moins « peaked » résout
   les tokens ambigus vers des quasi-synonymes ou des caractères d'une autre
   langue. Les maths, elles, sont justes.
4. **Ralentissement d'un facteur ~5** : cohérent avec le sampler's
   data-dependent de vLLM (`max_denoising_steps=48`, arrêt anticipé sur
   `confidence_threshold` / `stability_threshold`) — un modèle moins confiant
   fait plus d'itérations de débruitage par token. Le mécanisme n'est pas
   instrumenté dans le log serveur : c'est une explication plausible, pas une
   mesure.

## Variante testée

La variante `--include-norms-embed` (656 tenseurs fusionnés, 391 inchangés) a
été fusionnée et servie elle aussi : elle reste cohérente (3/3 prompts) mais
introduit davantage d'artefacts lexicaux que la variante par défaut
(« The capital of **way** is Paris »). La variante par défaut (embeddings et
normes inchangés, 355 tenseurs fusionnés) est donc celle qui est livrée.

## Limites de ce protocole

- FP8 (quantification à la volée) : c'est la seule configuration qui tient dans
  32 Go de VRAM sans offload CPU, le modèle en BF16 faisant 48,1 GiB. Les
  chiffres ne disent rien du comportement en BF16.
- `--max-model-len 2048` et `max_tokens=128` : les réponses longues sont
  tronquées, ce qui peut expliquer une partie des échecs de format
  « one_line » (le modèle continue après la première phrase).
- 103 prompts ne font pas un benchmark : c'est un contrôle de non-régression
  entre deux checkpoints, pas une mesure de qualité absolue.
- La perplexité ne s'applique pas (modèle de diffusion) : pas de métrique de
  perplexité n'est fournie.
