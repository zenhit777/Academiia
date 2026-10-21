# Academiia — A2D (AR-to-Diffusion Transfer) sur DiffusionGemma et Dream-7B

Ce dépôt implémente la méthode **A2D** (Qin et al., EPFL, 2026,
[arXiv:2610.08108](https://arxiv.org/abs/2610.08108)) :
transférer un vecteur de tâche autoregressif vers un modèle de diffusion, puis composer
ce transfert avec le vecteur de post-training du modèle de diffusion.

**La spécification de référence est [`docs/A2D_SPEC.md`](docs/A2D_SPEC.md)** : équations,
tableau des coefficients (Table 6 du papier), cibles chiffrées (Tables 1, 16, 21) et
règles de mapping, chaque valeur référencée par son tableau/section.

**État consolidé du projet** : [`docs/STATUS.md`](docs/STATUS.md) — état des portes G1/G2/G1b/G3,
ce qui est démontré mécaniquement, ce qui ne l'est pas, contraintes de la machine et
recommandation DiffusionGemma.

> **Note de statut (mission J6, 2026-10-20).** Le checkpoint `checkpoints/Jainius` est une
> **ablation hors papier** : il applique à DiffusionGemma-IT le vecteur *général*
> `τ_A = Gemma4_IT − Gemma4_Base` avec α = 0.3, alors que le papier (Tableaux 1b, 6, 9, 19)
> étudie pour DiffusionGemma un vecteur **de domaine** (IT → médical, LoRA) et sélectionne
> α ou λ **sur une tâche ancre**. Ce build n'est donc plus une référence scientifique ;
> il reste un témoin de l'outillage legacy. Le pipeline conforme est
> `scripts/merge_a2d.py` + `scripts/a2d/` (voir « Bibliothèque A2D »).

## Trajectoire par portes

| Porte | Contenu | Statut |
|---|---|---|
| **G1** | mécanique de fusion sur la paire publique Qwen2.5-7B → Dream-7B (Tableaux 6, 16, 21) | **OUVERTE — G1 NON FRANCHIE (J7, 2026-10-21)**. géométrie Table 21 : **écarts mesurés ≤ 0,11 %** ; baseline D_b/D_p **exécutée** sur GSM8K (n=300) et MATH-175 (n=175) ; critère de reproduction ±3 pts **non atteint** (C0.5 : Base MATH-175 +3,2 ; IFEval n=8 non concluant). **J7** : les modèles fusionnés **M2** (transfert α=0,9) et **M3** (merge λ=0,6) ont été **produits, vérifiés** (`post_write_verification` v2, `verify_merge` VERDICT OK, 0 ULP) et **évalués** sur les **mêmes items** (empreintes identiques sur les 4 modèles ; plancher de bruit de décodage **nul** au lot 4). Verdict amendé (M1/M2/M3) : **C2 atteint** (transfert − base, GSM8K : **+14,33** pts, IC [+9,33 ; +19,67] ; papier **+15,2** sur la même paire) ; **C1 non atteint** (MATH-175 : **+9,71** pts, IC [+2,86 ; +16,57] — **0,29 pt sous** le seuil de +10) ; **C3/C4 non établis** ; **un seul point** de la règle de franchissement échoue ⇒ **la porte reste ouverte**. **J8 phase A** — bilan de clôture (`docs/reports/G1_BILAN.md`) : M2 épuise 512 tokens sur 82/175 générations MATH, dont 67 sans réponse finale ; sur ces 67 items M2 − M0 = −2,99 [−11,94 ; +5,97], contre **+17,59** [+9,26 ; +26,85] sur les 108 autres ⇒ l'effet C1 est **dilué par le plafond**, le verdict est inchangé. Rapport : [`docs/reports/J7.md`](docs/reports/J7.md) ; protocole : [`docs/design/G1_PROTOCOL.md`](docs/design/G1_PROTOCOL.md) |
| **G1b** | faisabilité Nemotron-Labs-Diffusion-3B (Ministral-3-3B) | **non démarrée** — étude de faisabilité en lecture seule : [`docs/design/G1B_NEMOTRON3B.md`](docs/design/G1B_NEMOTRON3B.md) |
| **G2** | reproduction du setup médical DiffusionGemma du papier (adaptateurs publiés, voir [`docs/VP_ADAPTERS.md`](docs/VP_ADAPTERS.md)) | **DÉBLOQUÉE CÔTÉ BIBLIOTHÈQUE (J8)** — mode `merge-dg-scaled` implémenté et testé (adaptateur τ_D **non fusionné**, `lora_B × (1−λ)`, vérification V5/V6) ; **évaluation non faite** : le décodeur DG 26B + juge 27B ne tiennent pas sur 32 Go sans quantification (changement de protocole à déclarer). Conception/implémentation : [`docs/design/G2_adapter_scaling.md`](docs/design/G2_adapter_scaling.md) |
| **G3** | raisonnement (hors papier : conception d'abord) | **non démarrée** |

## Bibliothèque A2D (pipeline conforme)

```
scripts/a2d/            # bibliothèque : mapping, sources de vecteurs, moteur de fusion
scripts/merge_a2d.py    # CLI : transfert / merge / ablations
scripts/verify_merge.py # vérification post hoc (alpha_eff, lambda_eff par famille)
```

* **Modes** : `transfer` (`D_b + α·τ_A`, eq. 1), `merge` (`D_p + λ·(τ_A − τ_D)`, eq. 2),
  `merge-dg-scaled` / `adapter-scale` (DiffusionGemma : `D_b + λ·τ_A`, avec l'adaptateur
  τ_D **non fusionné** remis à l'échelle par `(1−λ)`, App. A.2),
  `ablation-dp-plus-tau-a` (Table 9), `ablation-dp-minus-tau-d` (Table 10).
* **Sources de vecteur** : paire de checkpoints complets (τ = post − base) **ou** adaptateur
  LoRA (`ΔW = (α/r)·B·A`, par module ; l'égalité avec `peft.merge_and_unload` est testée).
* **τ n'est jamais persisté** : il est recalculé à la volée, tenseur par tenseur, en FP32
  (3-4 handles `safe_open`) ; la provenance est portée par les hashes d'index des sources
  dans le manifeste `merge_config.json`.
* **Mapping** : identité (Qwen→Dream : *tous* les paramètres, embeddings et tête inclus) et
  Gemma4→DiffusionGemma (vision, embeddings, normes, tête inchangés ; routeur = famille à
  part, fusionné par défaut et déclaré).
* **α** ≥ 0 **sans plafond** (le papier va jusqu'à α = 1.4, Table 6 ; avertissement au-delà de 1.5) ;
  **λ** borné par l'eq. (2) à [0,1] (avertissement au-delà de 1). Un merge sans τ_D est une
  **erreur dure** (sortie 2) : aucune dégradation implicite en transfert.
* Arithmétique FP32, **un seul** cast BF16, écriture atomique, **vérification post-écriture
  obligatoire** (v2 : coefficients **signés** `c_a_eff` / `c_d_eff` par régression des moindres
  carrés sur [τ_A, τ_D], bornes dérivées du résidu BF16 **mesuré**, tenseurs non fusionnés
  **bit-identiques** à la référence, écart à `round_bf16(expected)` **= 0 ULP** (`max_ulp=0`,
  critère par défaut) ; échec dur hors tolérance). `--no-verify` la désactive mais **laisse une
  trace** au manifeste (`post_write_verification.skipped`).
* **Garde-fou « coefficient nul »** : un transfert à `α = 0` (eq. 1) ou un merge à `λ = 0`
  (eq. 2) n'écrirait qu'une **copie de la référence** (`D_b` / `D_p`) — indiscernable d'un
  modèle non fusionné, la vérification post-écriture ne pouvant pas le voir (elle compare les
  coefficients effectifs aux coefficients *demandés*, 0). Ces fusions sont donc **refusées**
  (sortie 2) tant que le plan contient des vecteurs non nuls ; `--allow-empty-coefficient`
  assume explicitement le no-op (avertissement inscrit au manifeste, état dans
  `coefficient_guard`).
* **Plan de fusion sur l'UNION des modules de τ_A et de τ_D** (classes `both` / `A-only` /
  `D-only`) ; politique `--coverage {strict,zero-fill}` : `strict` (défaut si les deux sources
  sont des paires de checkpoints complets) **refuse** une couverture inégale et liste les
  modules fautifs ; `zero-fill` traite un vecteur absent comme nul et le **déclare** au
  manifeste. Choix explicite obligatoire dès que le mode combine deux sources dont l'une
  n'est pas une paire de checkpoints (adaptateur LoRA, vecteur legacy). Un plan
  vide ⇒ refus (`--allow-empty` pour l'assumer).

```bash
# Transfert A2D (eq. 1) : D_b + alpha*tau_A
python scripts/merge_a2d.py --mode transfer \
    --diffusion-base checkpoints/dream_7b_base \
    --tau-a-post checkpoints/qwen25_7b_it --tau-a-base checkpoints/qwen25_7b \
    --alpha 0.9 --mapping identity --output checkpoints/dream_transfer_a09

# Merge A2D (eq. 2) : D_p + lambda*(tau_A - tau_D)
python scripts/merge_a2d.py --mode merge \
    --diffusion checkpoints/dream_7b_instruct \
    --tau-a-post checkpoints/qwen25_7b_it --tau-a-base checkpoints/qwen25_7b \
    --tau-d-post checkpoints/dream_7b_instruct --tau-d-base checkpoints/dream_7b_base \
    --lambda 0.6 --mapping identity --output checkpoints/dream_merge_l06

# DiffusionGemma (App. A.2) : base' = D_b + lambda*tau_A + adaptateur (1-lambda) NON fusionne
python scripts/merge_a2d.py --mode merge-dg-scaled \
    --diffusion checkpoints/diffusiongemma_it \
    --tau-a-post checkpoints/gemma4_it --tau-a-base checkpoints/gemma4_base \
    --tau-d-adapter checkpoints/dg_adapters/diffusion-vqarad \
    --lambda 0.3 --mapping gemma4-diffusiongemma \
    --output checkpoints/diffusiongemma_merge_l03
# -> checkpoints/diffusiongemma_merge_l03/adapter_scaled/ (lora_B x 0.7, lora_alpha inchange)

# Verification post hoc (detecte un coefficient errone)
python scripts/verify_merge.py --merged checkpoints/dream_transfer_a09 \
    --reference checkpoints/dream_7b_base --mode transfer \
    --tau-a-post checkpoints/qwen25_7b_it --tau-a-base checkpoints/qwen25_7b --alpha 0.9
```

## Baseline Dream-7B (porte G1) — protocole A2D App. A.1

Harnais : **fork lm-evaluation-harness de Dream** (`DreamLM/Dream`, `eval_instruct/lm_eval`,
modèle `diffllm`) dans un venv **séparé** (`harness/dream/requirements-dream.txt`) ;
`temperature=0.1`, `top_p=0.9`, `alg=entropy`, `diffusion_steps = max_new_tokens` (un pas de
débruitage par token), bfloat16, chat template appliqué, 0-shot pour GSM8K et MATH-175,
prompt système GSM8K de l'App. A.1 cité mot pour mot.

| Modèle | Tâche | Métrique paper-compatible | Score | n | IC 95 % (Wilson) | Cible Table 16 | Écart |
|---|---|---|---|---|---|---|---|
| Dream-7B-Base | GSM8K | flexible-extract | 68,7 % | 300 | [63,2 ; 73,7] | 66,5 | +2,2 |
| Dream-7B-Base | MATH-175 | math_verify | 24,0 % | 175 | [18,3 ; 30,8] | 20,8 *(MATH-500)* | +3,2 |
| Dream-7B-Instruct | GSM8K | flexible-extract | 84,0 % | 300 | [79,4 ; 87,7] | 83,0 | +1,0 |
| Dream-7B-Instruct | MATH-175 | math_verify | 36,6 % | 175 | [29,8 ; 43,9] | 38,4 *(MATH-500)* | −1,8 |
| **Dream-7B-Transfert α=0,9** (M2) | GSM8K | flexible-extract | **83,0 %** | 300 | [78,3 ; 86,8] | 81,7 *(transfert, Fig. 2a)* | +1,3 |
| **Dream-7B-Transfert α=0,9** (M2) | MATH-175 | math_verify | **33,7 %** | 175 | [27,1 ; 41,0] | 40,2 *(transfert, MATH-500)* | −6,5 |
| **Dream-7B-Merge λ=0,6** (M3) | GSM8K | flexible-extract | **85,7 %** | 300 | [81,2 ; 89,2] | 83,7 *(merge, Fig. 2b)* | +2,0 |
| **Dream-7B-Merge λ=0,6** (M3) | MATH-175 | math_verify | **34,3 %** | 175 | [27,7 ; 41,6] | 45,0 *(merge, MATH-500)* | −10,7 |

> **MATH-175** = sous-ensemble `minerva_math` du fork Dream (25 problèmes × 7 sujets = 175,
> sur les 5 000 du test MATH). Le jeu du papier est **MATH-500** (500 problèmes, split PRM800K) :
> la comparaison ci-dessus est **à périmètre différent** et ne vaut pas reproduction
> (MAJ-04 de `docs/reports/AUDIT_J6.md`).

**Ce que cette campagne établit, et ce qu'elle n'établit pas.** Elle établit que la
**mécanique d'évaluation** (harnais, tâches, prompts système, extraction, scoring) fonctionne
de bout en bout et que les **checkpoints de référence** D_b / D_p ont été **exécutés** dans le
protocole App. A.1. Elle **n'établit pas** la validité scientifique de la **bibliothèque de
fusion** : au 2026-10-20, **aucun modèle fusionné n'avait été évalué** — la seule vérification
sur un modèle 7B était **mécanique** (fusion transfert α = 0,9 sur Dream-7B-Base, `α_eff`
mesuré, sans évaluation).

**Depuis J7 (2026-10-21), les deux modèles fusionnés sont évalués** sur les mêmes items que
D_b / D_p (empreintes identiques, lot 4) : le **transfert** (M2, α = 0,9) donne un gain **net
et statistiquement appuyé** sur GSM8K (**+14,33** pts, IC [+9,33 ; +19,67] ; papier : **+15,2**
sur la même paire) et un gain **réel mais sous le seuil pré-enregistré** sur MATH-175
(**+9,71** pts, IC [+2,86 ; +16,57]) ; le **merge** (M3, λ = 0,6) est **statistiquement
indiscernable** de son modèle de référence. Les quatre critères amendés n'étant pas tous
atteints, **G1 reste ouverte** (règle de franchissement §1). Par ailleurs, « la cible du papier
tombe dans l'IC 95 % » est une **non-rejection**, pas une preuve d'équivalence à ±3 points : le
critère C0.5 est **NON atteint** (Base MATH-175 : +3,2, hors ±3 ; IFEval : n = 8, non concluant)
et la porte **G1 reste ouverte** (revue N2 ; protocole de remplacement :
`docs/design/G1_PROTOCOL.md`).

Détail, échantillons et commande de rejeu : `docs/reports/J6.md`, `docs/reports/J6bis.md`,
`docs/reports/J7.md` (verdict J7 et preuves), `eval/dream_a2d/SUMMARY.md`,
`eval/dream_a2d/G1_VERDICT.json`.

## Géométrie des vecteurs (Table 21 du papier) — écarts mesurés ≤ 0,11 %

`scripts/measure_geometry.py` recalcule, sur le backbone (attention + MLP + normes ;
embeddings et tête exclus) :

| Grandeur | Mesuré (J6) | Table 21 | Écart |
|---|---|---|---|
| ‖τ_A‖ | 23.01 | 23.0 | +0.05 % |
| ‖τ_D‖ | 31.47 | 31.5 | −0.11 % |
| ‖Δ_b‖ | 729.04 | 729 | +0.01 % |
| ‖Δ_p‖ | 732.14 | 732 | +0.02 % |
| cos(Δ_b, Δ_p) | +0.9986 | +0.999 | — |
| cos(τ_A, τ_D) | +0.0130 | +0.013 | — |

## Workflow legacy (Jainius, déprécié — déplacé dans `legacy/`)

`legacy/scripts/merge_jainius.py` est un **wrapper déprécié** de la bibliothèque ; il conserve son
interface historique (τ en `.npy`, α borné à [0,1], manifeste `merge_config.json`).
`legacy/scripts/compute_tau_a.py` est un **chemin legacy** (il persiste τ) : il n'est pas supprimé,
mais le pipeline conforme ne s'en sert pas. Les archives de la période Jainius
(`AUDIT.md`, `GUIDE_RESTANT.md`) sont dans `legacy/docs/`.

```bash
# 1. Télécharger les checkpoints (~155 Go au total, mesuré : 3 × ~51.6 Go)
python scripts/download_models.py --output-dir ./checkpoints
#    (famille Dream : --set dream)

# 2. Calculer tau_A = A_p - A_b (~10 min, ~3 GB RAM peak)   [LEGACY]
#    ATTENTION : safetensors pré-fault la projection du shard AR de 46,5 GiB ->
#    il faut >= 50 Go de RAM disponibles (sur WSL2 : memory= dans .wslconfig)
python legacy/scripts/compute_tau_a.py \
    --base ./checkpoints/gemma4_base \
    --posttrained ./checkpoints/gemma4_it \
    --output-dir _artefact/tau_a

# 3. Mesurer l'effet reel apres arrondi BF16 (CPU seul)   [LEGACY]
python legacy/scripts/measure_effect.py \
    --diffusiongemma ./checkpoints/diffusiongemma_it \
    --tau-dir _artefact/tau_a \
    --alphas 0.05,0.1,0.2,0.3,0.5,0.8

# 4. Appliquer le merge Dp + alpha*tau_A (~15 min, ~5 GB RAM peak)   [LEGACY]
python legacy/scripts/merge_jainius.py \
    --diffusiongemma ./checkpoints/diffusiongemma_it \
    --tau-dir _artefact/tau_a \
    --output ./checkpoints/Jainius \
    --alpha 0.3

# 5. Verifier le checkpoint (format HuggingFace strict)
python scripts/verify_checkpoint.py --checkpoint ./checkpoints/Jainius
```

## Principe du build legacy

```
tau_A = Gemma4_IT − Gemma4_Base          # vecteur tâche AR post-training
Jainius = DiffusionGemma + 0.3 * tau_A_mapped
```

- **α = 0.3** (défaut ; 0.8 est déprécié — il produit un texte dégénéré, cf. « Évaluation par génération »)
- **355 tenseurs** : projections decoder de Gemma 4 → DiffusionGemma, shapes alignées, τ_A appliqué
  (attention 115, MLP 90, experts 60, **routeur 90**)
- **692 tenseurs** : inchangés (embeddings, normalisations, `layer_scalar`, encoder vision)
- Variante `--include-norms-embed` : 656 tenseurs fusionnés / 391 inchangés (embeddings et norms inclus)

## Résultats (build legacy Jainius)

| | DiffusionGemma original (D_p) | Jainius (α=0.3, 2026-10-19) |
|--|---|---|
| Tenseurs | 1047 (11 fichiers) | 1047 (11 fichiers) |
| Format | BF16 | **BF16** ✓ (strict : 11/11 shards, en-têtes « BF16 ») |
| Conformité à la source | — | ✓ (`verify_checkpoint.py --reference` : 1047 tenseurs, 0 écart de shape/dtype) |
| Vecteur appliqué | — | **355 tenseurs** (attn 115, mlp 90, experts 60, **routeur 90**) ; 692 inchangés, bit-identiques |
| Provenance | — | `merge_config.json` (α, variante, date, commit) |

## Limites (build legacy)

 - **Hors protocole du papier.** Ce build applique un vecteur **général** (base → IT général) à un
   modèle **déjà post-entraîné** : c'est un *double comptage* par rapport au setup du papier, où
   `τ_A` est un vecteur **de domaine** (IT → médical) et où le coefficient est choisi sur une
   tâche ancre (Table 6 : α = 1.0 SLAKE / 0.7 VQA-RAD ; λ = 0.1 / 0.3). Voir
   [`docs/A2D_SPEC.md`](docs/A2D_SPEC.md) et [`docs/VP_ADAPTERS.md`](docs/VP_ADAPTERS.md).
 - **« τ_D inexistant » n'est plus exact.** `τ_D` est bien défini (`D_p − D_b`, §2.1) ; pour
   DiffusionGemma il correspond au delta LoRA de post-training de domaine, **publié**
   (`gevaertlab/diffusiongemma-radiology-vqa`, cf. `docs/VP_ADAPTERS.md`). Ce qui manque à ce
   dépôt n'est pas τ_D mais la **chaîne complète** (adapters + merge + évaluation conforme).
 - **Le routeur est fusionné** (90 tenseurs sur 355), contrairement à ce qu'indiquaient
   `AGENTS.md`/`KNOWN_ISSUES.md` (« routeur inchangé »). C'est sans effet mesurable ici
   (τ_routeur ≈ 0), mais la documentation est désormais alignée sur le code.
 - A2D n'est appliqué qu'aux projections decoder (355 tenseurs sur 1047) : embeddings,
   normalisations, `layer_scalar` et encoder vision restent ceux de D_p — conforme à
   l'App. A.2 du papier (embeddings/normes/tête inchangés pour DiffusionGemma) ; la variante
   `--include-norms-embed` fusionne 656 tenseurs (391 inchangés).
 - **Aucun gain démontré** : sur 103 prompts, Jainius (α=0.3) est moins bon que D_p sur la justesse des réponses (88,1 % contre 100 %) et 5,5× plus lent. `D_p` reste le témoin ; le merge A2D tel qu'implémenté ici dégrade le modèle de diffusion.
 - α=0.8 (défaut historique, rebuild v2) ne produit pas un texte cohérent : il perturbe les projections d'attention de ~50 % de leur norme. α=0.3 est le coefficient retenu.
 - Les mesures de génération ont été faites en **FP8** (quantification à la volée) : c'est la seule configuration qui tient dans 32 Go de VRAM, le modèle en BF16 faisant 48,1 GiB. Le comportement en BF16 n'est pas mesuré. Le runner v1 est inutilisable (crash scatter CUDA, cf. `vLLM/KNOWN_ISSUES.md`).
- `tau_A` (chemin legacy, `.npy`) ne peut pas être recalculé sur cette machine : `safetensors.safe_open` pré-fault la projection du shard de 46,5 GiB des checkpoints AR, au-delà des 30 GiB de RAM de la VM WSL2 (ENOMEM). La **bibliothèque conforme** (`scripts/merge_a2d.py`) n'a pas ce problème : elle recalcule τ à la volée, tenseur par tenseur, sans jamais matérialiser le vecteur.
- Poids dérivés de Gemma : soumis aux conditions d'utilisation Gemma (https://ai.google.dev/gemma/docs/gemma_4_license). Les repos sources (`google/gemma-4-26B-A4B`, `google/gemma-4-26B-A4B-it`, `google/diffusiongemma-26B-A4B-it`) ne publient **aucun fichier LICENSE** (vérifié le 2026-10-19) : rien à recopier, la model card déclare `apache-2.0` + le lien des conditions Gemma.

## Évaluation par génération (2026-10-19) — **non conforme au protocole du papier**

Cette campagne (J5) **ne constitue pas une évaluation de qualité au sens de l'article** :
FP8 à la volée au lieu de BF16, `temperature=0` au lieu du réglage natif (entropie, T = 0.1,
top-p = 0.9), 128 tokens au lieu des budgets par tâche, 103 prompts de texte libre au lieu de
IFEval/GSM8K/MATH-500, et **aucun juge**. Les chiffres ci-dessous sont des **contrôles de
non-régression** entre deux checkpoints, pas des scores de benchmark.

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

 Le script `legacy/scripts/measure_effect.py` rapporte, en plus du % d'éléments modifiés,
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
