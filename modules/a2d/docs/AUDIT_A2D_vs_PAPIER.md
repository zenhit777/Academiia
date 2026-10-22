# Audit — Le pipeline `scripts/` confronté à l'article A2D (arXiv:2610.08108v1)

Date : 2026-10-19. Périmètre : vérifier si le dépôt Academiia met en œuvre la
méthode décrite dans l'article A2D (arXiv:2610.08108, <https://arxiv.org/abs/2610.08108>), identifier ce qui a
été manqué/omis, et expliquer l'écart entre nos résultats et la publication.

Sources primaires : PDF extrait (`_artefact/a2d_paper.txt`), scripts relus
intégralement, index des checkpoints inspectés, mesures internes
(`_artefact/measure_effect*.json`), release externe Van Puyvelde et al.
(arXiv:2607.01436, HF `gevaertlab/diffusiongemma-radiology-vqa`).

> **Note de lecture (J8, 2026-10-21)** : cet audit décrit l'état du dépôt **au
> 2026-10-19**. Depuis, le pipeline legacy a été déplacé dans `modules/a2d/legacy/scripts/`
> (`compute_tau_a.py`, `merge_jainius.py`, `measure_effect.py`) et les archives
> dans `modules/a2d/legacy/docs/`. Les chemins `scripts/...` cités ci-dessous sont ceux de
> l'époque : ils restent exacts comme provenance historique, pas comme
> emplacement actuel.

---

## 1. Ce que prescrit exactement l'article

### 1.1 Équations (§2.3, Tableaux 4, 7, 12)

| Construction | Formule | Remarque |
|---|---|---|
| τ_A (vecteur AR) | `A_p − A_b` | AR post-trained − AR base/pré-tâche |
| τ_D (vecteur diffusion) | `D_p − D_b` | diffusion post-trained − diffusion base |
| **A2D-Transfer** | `D_b + α·τ_A` | sur le modèle de diffusion **base** (ou IT si pas de base publique) |
| **A2D-Merge** | `D_b + (1−λ)·τ_D + λ·τ_A` = `D_p + λ·(τ_A − τ_D)` | la construction principale du papier |

- Table 12 (Dp + λτA / Dp − λτD / merge) : la fusion bat les deux effets isolés
  (Avg. Acc. 76.8 vs 75.9/74.2, parent 72.3) ; **le A2D-Merge est le cœur de la
  méthode**.
- Tableau 9 du papier pour DiffusionGemma : `Dp + ατA` où **Dp = diffusion
  médical post-traité** et **τA = vecteur médical** (pas notre configuration).

### 1.2 Coefficients (Tableau 6 — sélection sur « anchor benchmark »)

| Famille | Vecteur | Transfert α | Merge λ |
|---|---|---|---|
| Qwen2.5-7B → Dream-7B | instruct (SFT) | 0.9 (IFEval) | 0.6 (IFEval) |
| Qwen2.5-Coder → Dream-Coder | code | 0.6 | 0.2 |
| Qwen2.5-Coder → DiffuCoder | code | 0.1 | 0.2 |
| AceCoder-Rule → DiffuCoder | code RL | 0.2 | 0.5 |
| Qwen3-8B → DreamReasoner | reasoning | 0.5 | 0.2 |
| Ministral-3-3B → Nemotron-LD-3B | instruct | 0.9 | 0.3 |
| Ministral-3-8B → Nemotron-LD-8B | instruct | **1.4** | 0.4 |
| Ministral-3-8B-Reasoning → Nemotron-LD-8B | reasoning | 0.8 | 0.2 |
| **Gemma 4 → DiffusionGemma** | **SLAKE (SFT)** | **1.0** (SLAKE) | **0.1** (SLAKE) |
| **Gemma 4 → DiffusionGemma** | **VQA-RAD (SFT)** | **0.7** (VQA-RAD) | **0.3** (VQA-RAD) |

- Les balayages montent jusqu'à α=1.4/1.5 : un plafond α ∈ [0,1] est un choix
  qui n'est pas dans le papier.
- Le papier sélectionne le coefficient sur une tâche ancre **puis le garde
  fixe** sur les autres ; il n'utilise jamais un α choisi « pour éviter la
  dégénérescence ».

### 1.3 Spécifique DiffusionGemma (A.1 + A.2)

- **Pas de checkpoint base diffusion public** : pour cette famille, `D_b` est
  le checkpoint **IT** (« instruction-tuned reference checkpoint », notes
  Tableaux 7/8), et le papier étudie un **fine-tuning de domaine**
  (VQA-RAD / SLAKE / VQA-Med) des deux branches IT.
- Les fine-tunes sont des **LoRA** (« identical LoRA recipe », Van Puyvelde et
  al. arXiv:2607.01436) : rank-64, α=128, sur **attention + MLP partagé** ;
  experts, routeur et vision tower **gelés**. Checkpoints publiés :
  `gevaertlab/diffusiongemma-radiology-vqa` (sous-dossiers `diffusion-*` et
  `ar-*` par dataset), code `github.com/mxvp/discrete_diffusion_RRG`.
- Mapping (A.2) : « We map the AR updates to the corresponding
  DiffusionGemma language-model projections. The diffusion encoder and decoder
  share their base weights […] **The vision encoder, embeddings, normalization
  parameters, and output head are unchanged**. »
- « The diffusion fine-tuning adapters, which hold separate encoder and
  decoder updates, **stay unmerged and are scaled according to the experimental
  formula** » → `(1−λ)·τ_D` est réalisé en **mettant à l'échelle des adapters**,
  pas seulement par une différence de poids.
- C'est une divergence importante : `τ_A` du setup DG est **IT → médical**
  (local, petits modules), pas base → IT.

### 1.4 Protocole d'évaluation DG (A.1)

- Décodeur natif block-diffusion (Transformers) : block 256, ≤48 pas de
  débruitage, **entropy bound 0.1**, température **0.8 → 0.4** annealée, arrêt
  anticipé quand les prédictions cessent de changer, **thinking désactivé**.
- Datasets : VQA-RAD (451 test), SLAKE (1061), VQA-Med-2019 (500), 0-shot,
  image + question, budget 256 tokens.
- Score : exactitude sémantique jugée par **Qwen3.8-27B** (greedy, thinking off).
- Résultats cibles (Tableau 1b, colonnes VQA-RAD/SLAKE/VQA-Med) :
  - AR donor : 75.8 / 66.5 / 44.0
  - Base (IT) : 57.4 / 63.2 / 43.2
  - A2D-Transfer : **74.1 / 67.3 / 45.2**
  - Post-trained (Dp médical) : 71.6 / 62.2 / 38.6
  - A2D-Merge : **74.5 / 63.8 / 38.8**
  - (B.5/Tableau 19, variante SLAKE : transfer 56.5/**77.1**/37.2 ; merge
    61.6/**83.3**/40.0.)

### 1.5 Portée et géométrie (B.7/Tableau 21, §Limitations)

- Les vecteurs de **post-training locaux** transfèrent ; les vecteurs de
  **continued pretraining** (70–82× plus gros) ne transfèrent pas (Tableau 15).
- `cos(Δ_b, Δ_p) ≥ 0.98` (la direction de conversion AR→diffusion reste
  stable) ; `cos(τ_A, τ_D)` faible (mises à jour largement distinctes).
- Pour Qwen2.5→Dream-7B : `‖τ_A‖=23.0`, `‖τ_D‖=31.5`, `‖Δ_b‖=729`.

---

## 2. Ce que fait réellement le dépôt

### 2.1 Script par script

| Script | Rôle réel | Écarts vs papier |
|---|---|---|
| `download_models.py` | 3 modèles : `gemma4_base`, `gemma4_it`, `diffusiongemma_it` | Ne connaît ni les LoRA médicaux (`gevaertlab/...`), ni les modèles Dream/Qwen des expériences principales. |
| `compute_tau_a.py` | `τ_A = A_p − A_b` par différence de tenseurs (safe_open), filtre `decoder`, manifest | Un seul type de vecteur (poids complets). **Aucun chemin adapters/LoRA**. Bloqué ENOMEM sur les shards AR de 46,5 GiB (VM 30 GiB). |
| `merge_jainius.py` | `Jainius = D_p + α·τ_A` uniquement ; α ∈ [0,1] ; mapping DG ; exclusion norms/embed/`layer_scalar` par défaut | **Pas de τ_D, pas de λ, donc pas de A2D-Merge** (docstring l'admet). Pas d'adapters. |
| `measure_effect.py` | Métrologie de l'arrondi BF16 (ratio, cos, fidélité) | N'est pas un balayage de performance ; libellé trompeur « variante papier A2D : INCLURE norms/embeddings » — c'est l'inverse pour DG (A.2 : norms/embeddings **inchangés**). |
| `eval_compare.py` | 103 prompts texte, matching `expected` par sous-chaîne, `temperature=0`, `max_tokens=128`, vLLM FP8 | **Pas d'images, pas de VQA-RAD/SLAKE/VQA-Med, pas de juge LLM, pas de BF16, pas des réglages du décodeur natif** (température 0.8→0.4, entropy 0.1, block 256). |
| `verify_checkpoint.py` | Intégrité stricte (headers, dtypes BF16, index, sidecars, `--reference`) | Correct ; n'a pas vocation à vérifier la conformité méthode. |
| `smoke_int8.py` | Test de non-casse (forward INT8/offload) | Hors méthode ; documenté comme non-preuve. |
| `_common.py` | Mapping Gemma4↔DG, helpers, manifest | Mapping **DG uniquement** ; pas de mapping générique inter-familles. |
| `modules/a2d/tests/` | Sûreté du merge (overwrite, manifest, shapes, README, exclusion) | Aucun test de τ_D/λ/adapters (fonctions inexistantes). |

### 2.2 Constat sur `Jainius`

`Jainius = diffusiongemma_it + 0.3·τ_A` avec `τ_A = gemma4_it − gemma4_base`.

En langage du papier :
- Proche de la formule **A2D-Transfer** (`D_b + ατ_A`), où `D_b` = IT — mais
  avec un **τ_A hors-papier** (base→IT général au lieu d'un vecteur de tâche
  IT→médical) et **α=0.3** (aucune sélection sur ancre).
- Ce n'est **ni** l'ablation « Tab. 9 » (qui suppose `D_p` médical et τ_A
  médical) **ni** le A2D-Merge. L'étiquette « Dp + ατA (Tab. 9) » du dépôt est
  donc inexacte.

### 2.3 Audit du mapping (faits mesurés)

- DG : 1047 tenseurs = `decoder` 661 (dont 655 couches + embed + norm final +
  4 `self_conditioning`) + `encoder` 386 (vision 356 + `layer_scalar` LM 30).
- Mappables par nom : **656** ; exclus par défaut : **301** (norms, q/k_norm,
  `layer_scalar`, embed_tokens) ; fusionnés : **355** =
  attn 115 (q30+k30+v25+o30) + mlp 90 + experts 60 + **routeur 90**.
- Non mappables (restent en D_p, conformément A.2) : vision 356,
  `self_conditioning` 4, `decoder.norm` 1, `encoder.language_model.layer_scalar`
  30.
- Tous les noms transformés existent des deux côtés (aucune cible fantôme,
  aucune collision) ; les 30 `layer_scalar` de l'encodeur et les 4
  `self_conditioning` restent inchangés — conforme au principe « unmatched
  parameters kept from the diffusion recipient ».
- Remarque : le routeur et les experts **sont** fusionnés par défaut (contra
  `AGENTS.md`/`KNOWN_ISSUES.md` qui les disent « inchangés »). Pour notre
  vecteur général c'est sans effet mesurable (`τ_routeur ≈ 0`, `frac_changed`
  à α=0.3 : 0,1 %) ; pour un vecteur LoRA médical, experts/routeur/vision sont
  gelés (τ=0) donc le débat est moot, mais la doc est fausse.

### 2.4 Mesures internes (à confronter au Tableau 21)

- À α=0.3, `‖ατ‖/‖D_p‖` médian ≈ 2,98 % (attn), 2,82 % (mlp/experts),
  cos ≈ 0,998 ; BF16 ne détruit quasiment rien (fidélité ≈ 1,000) ;
  norms : 0,044 % et cos 0,61 (arrondi destructif) ; **routeur : τ ≈ 0**.
- Autrement dit notre τ_A a une échelle « plausible » (≈10 % de la norme du
  modèle), comparable à Ministral (Tableau 21 : ≈11–15 % de ‖Δ_b‖) ; la cause
  de l'échec n'est donc **pas** d'abord l'échelle, mais la **nature du
  vecteur** et **les conditions d'évaluation** (voir §4).

### 2.5 Évaluation (rappel)

- FP8 à la volée (seule config tenant dans 31,84 GiB), `temperature=0`,
  `max_tokens=128`, 103 prompts texte, scoring par présence de sous-chaîne.
- Variabilité déjà observée d'une passe à l'autre (91,0 % → 88,1 %) :
  le sampler de diffusion est stochastique même à `temperature=0`.
- Conclusion consignée « aucun gain » — correcte **pour cette configuration**,
  mais non comparable à la publication.

---

## 3. Preuves externes que la reproduction est possible

1. **Van Puyvelde et al.** publient code + checkpoints :
   - `https://github.com/mxvp/discrete_diffusion_RRG`
   - `https://huggingface.co/gevaertlab/diffusiongemma-radiology-vqa`
     (sous-dossiers `diffusion-vqarad`, `ar-vqarad`, `diffusion-slake`,
     `ar-slake`, `diffusion-vqamed`, `ar-vqamed`).
   - LoRA rank-64/α=128 sur attention + MLP partagé, experts/routeur/vision
     gelés → `τ_A`/`τ_D` médicaux sont **analytiques** : `ΔW = scale·B·A` sur
     les modules adaptés, sans différence de poids complète.
2. Résultats cibles déjà publiés pour ces paires (Tableaux 1b/19 du papier
   A2D) → on peut comparer chiffre à chiffre.
3. Pour une validation de **la mécanique de merge** (sans images ni médical),
   la paire Qwen2.5-7B / Dream-7B est **entièrement publique** (Tableaux 6/16) :
   transfert α=0.9 → IFEval 59,2 ; merge λ=0.6 → IFEval 72,8.

---

## 4. Écarts identifiés (classés)

| # | Écart | Sévérité | Preuve |
|---|---|---|---|
| E1 | **A2D-Merge jamais implémenté** (pas de τ_D, pas de λ) | Bloquant scientifique | `merge_jainius.py` docstring « PAS le A2D-Merge complet : pas de τ_D » ; aucun `tau_d`/`lambda` dans `scripts/`. |
| E2 | **« τ_D inexistant publiquement » est faux** | Bloquant (prémisse) | Le papier définit pour DG `D_b` = IT ; τ_D = delta LoRA `diffusion-*` ; adapters publiés (Van Puyvelde). |
| E3 | **τ_A non conforme** (base→IT général ≠ IT→médical du setup DG) | Bloquant scientifique | A.1/Tableaux 6–9 : vecteurs médicaux ; notre dépôt : vecteur général. |
| E4 | **Pas de support adapters/LoRA** | Bloquant reproduction | A.2 : « adapters stay unmerged and are scaled » ; pipeline = seulement différences de tenseurs. |
| E5 | **Aucune sélection de coefficient** (α=0.3 ad hoc ; plafond [0,1]) | Majeur | Tableau 6 : α=0.7/1.0, λ=0.3/0.1 pour DG ; balayages jusqu'à 1.4/1.5. |
| E6 | **Évaluation non conforme** (FP8, temp=0, 128 tokens, texte, sans juge) | Majeur | A.1 : BF16/natif (0.8→0.4, entropy 0.1, block 256), VQA médicales, juge Qwen3.8-27B. |
| E7 | **Aucune reproduction des tableaux principaux** (familles Dream/DiffuCoder/Nemotron) | Majeur | §3.2 du papier ; entièrement absent au dépôt. |
| E8 | Contrainte machine : `safe_open` pré-fault 46,5 GiB → ENOMEM (VM 30 GiB) | Mineur (contournable) | `KNOWN_ISSUES.md` §Blocage ; contournable via LoRA (B·A) ou modèle 7B (shards ≤ 5 GiB). |
| E9 | Docs : étiquettes « Tab. 9 », « variante papier », « routeur inchangé », « τ_D inexistant » | Mineur (correction doc) | `AGENTS.md`, `KNOWN_ISSUES.md`, docstrings. |

**Ce qui est correct** (à créditer) : arithmétique FP32 → cast BF16 unique,
écriture atomique, manifest de provenance, exclusion vision/embeddings/norms/
head conforme A.2 pour DG, conservation des paramètres sans équivalent AR,
outil de vérification strict, tests de sûreté, métrologie de l'arrondi BF16.

---

## 5. Conséquence sur la conclusion scientifique

- Le résultat « Jainius α=0.3 dégrade D_p » est **réel pour la configuration
  testée** mais **ne réfute pas l'article** : vecteur différent (général au lieu
  de médical), coefficient non sélectionné, évaluation différente (texte/FP8/128
  tokens/sans juge) et petite (103 prompts, sampler stochastique).
- Autrement dit : le développement **n'est pas terminé** vis-à-vis de la
  publication. Le code actuel ne peut pas, en l'état, produire les expériences
  DiffusionGemma du papier (il manque le merge complet, l'arithmétique des
  adapters, les vecteurs médicaux et un protocole d'évaluation conforme).

---

## 6. Plan de remédiation proposé

### R1 — Implémenter le merge complet (cœur du papier) — faible coût
- `merge_a2d.py` (ou extension de `merge_jainius.py`) : entrées `--tau-a`,
  `--tau-d` (optionnel), `--alpha`, `--lambda` ; formule par tenseur
  `out = D + (1−λ)·τ_D + λ·τ_A` (dégénère proprement en transfert si τ_D
  absent) ; lever le plafond α ; manifest complet ; tests unitaires miroirs
  des nouvelles branches (τ_D présent/absent, λ=0/1, garde manifestations).
- Corriger la doc (E9) : remplacer « Tab. 9 » par la vraie attribution
  (Transfer = Tableau 8 pour DG ; merge = Tableaux 7/12), « τ_D disponible via
  les LoRA médicaux », « routeur/experts fusionnés par défaut ».

### R2 — Validation de la mécanique sur la paire publique Dream-7B — faisable ici
- Télécharger `Qwen2.5-7B`, `Qwen2.5-7B-Instruct`, `Dream-7B-Instruct`
  (≈ 45 Go, disque libre 1,43 To) ; shards ≤ 5 GiB → pas de blocage RAM.
- Calculer τ = IT − base, produire transfert α=0.9 et merge λ=0.6, comparer
  aux nombres du Tableau 16 (IFEval 49,7 → 59,2 / 72,8). BF16 7B ≈ 15 Go VRAM
  → tient sur la RTX 5090 ; harness lm-eval du fork Dream.

### R3 — Reproduction du setup DiffusionGemma du papier — cible scientifique
- Télécharger les LoRA `gevaertlab/diffusiongemma-radiology-vqa`
  (VQA-RAD + SLAKE au minimum).
- τ_A = `ar-*` (Gemma-4-IT → Gemma-4-IT+médical), τ_D = `diffusion-*`
  (DG-IT → DG-IT+médical), via `scale·B·A` sur les modules attention/MLP.
- Produire : transfert (α=1.0 SLAKE / 0.7 VQA-RAD) et merge (λ=0.1 / 0.3) ;
  mapper les tenseurs Gemma4 → DG decoder (mapping existant, 205 modules
  adaptés) ; `encoder/vision/norms/embed/router/experts` inchangés (conforme
  LoRA/A.2).
- Évaluer : datasets médicaux requis ; sur cette machine, le BF16 DG (48 GiB)
  ne tient pas en VRAM et le juge Qwen3.8-27B est hors de portée → faire au
  minimum un contrôle closed-form (yes/no, exact-match) sur un sous-ensemble
  d'images, en étiquetant explicitement « non conforme au protocole du papier »,
  ou reporter l'évaluation à une machine plus large (H100/H200).

### R4 — Prérequis machine (rappel)
- VM WSL2 : 30 GiB de RAM (28 libres ce jour) ; liberer le blocage τ_A complet
  nécessiterait `[wsl2] memory=64GB` (ou re-shard des AR). Non requis pour R2/R3
  (τ via B·A).
- VRAM : mesurer `nvidia-smi` avant chaque série GPU (règle AGENTS) ; aucune
  opération GPU dans cet audit.

---

## 7. Annexes — fichiers consultés

- `_artefact/a2d_paper.txt` (extraction du PDF, 31 pages).
- `scripts/{download_models,compute_tau_a,merge_jainius,measure_effect,eval_compare,verify_checkpoint,smoke_int8,_common}.py`,
  `modules/a2d/tests/*`.
- `_artefact/measure_effect*.json` (4 fichiers), `modules/a2d/legacy/eval/RESULTS.md`,
  `KNOWN_ISSUES.md`, `infra/vllm/.env.template`.
- Index de `checkpoints/{gemma4_base,gemma4_it,diffusiongemma_it}` ;
  `checkpoints/diffusiongemma_it/generation_config.json`.
- Van Puyvelde et al. : arXiv:2607.01436 ; HF `gevaertlab/diffusiongemma-radiology-vqa` ;
  code `github.com/mxvp/discrete_diffusion_RRG`.
