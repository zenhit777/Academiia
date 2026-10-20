# A2D — spécification de la méthode et cibles chiffrées

Référentiel **primaire** : `docs/2610.08108v1 - A2D (Enhancing diffusion language models
with autoregressive post-training weights).pdf` (Qin et al., EPFL, arXiv:2610.08108v1, 31 p.).
Second référentiel : `docs/2608.00146v1 - DiffusionGemma (Technical Report).pdf`
(Google DeepMind, arXiv:2608.00146v1, 55 p.).

Chaque valeur chiffrée porte son numéro de tableau / section. Les valeurs marquées
**[mesuré J6]** sont celles recalculées localement le 2026-10-19/20
(`docs/reports/J6.md`), pas des valeurs recopiées.

> L'audit `docs/AUDIT_A2D_vs_PAPIER.md` (19/10) est une source **secondaire** : il a servi
> de piste de lecture, chaque point utilisé ici a été revérifié dans le PDF (Table 1/6/16/21,
> App. A.1/A.2/B.7) — voir « Désaccords avec l'audit » dans `docs/reports/J6.md`.

---

## 1. Notation (§2.1, PDF p. 3)

| Symbole | Définition | Source |
|---|---|---|
| `A_b` | poids AR **base** (avant post-training) | §2.1 |
| `A_p` | poids AR **post-trained** (le « AR donor ») | §2.1 |
| `D_b` | poids diffusion **base** | §2.1 |
| `D_p` | poids diffusion **post-trained** (le « diffusion parent ») | §2.1 |
| `τ_A` | `A_p − A_b` — vecteur de tâche AR | §2.1 (PDF p. 3) |
| `τ_D` | `D_p − D_b` — vecteur de tâche diffusion | §2.1 (PDF p. 3) |

Citation (§2.1) : « *we define the AR and diffusion post-training vectors as
**τ_A = A_p − A_b** and **τ_D = D_p − D_b**, respectively. Notably, task arithmetic typically
composes task vectors defined relative to a shared base model or initialization, whereas τ_A
and τ_D originate from different base checkpoints.* »

---

## 2. Équations

| # | Équation | Section / numéro | PDF |
|---|---|---|---|
| E1 | `D_Transfer(α) = D_b + α·τ_A,  α ≥ 0` | §2.4, **eq. (1)** | p. 6 |
| E2 | `D_Merge(λ) = D_b + (1−λ)·τ_D + λ·τ_A = D_p + λ·(τ_A − τ_D),  λ ∈ [0,1]` | §2.4, **eq. (2)** | p. 6 |
| E3 | `D(α, β) = D_b + α·τ_A + β·τ_D` (plan 2-D ; `D_T = D(1,0)`, `D_p = D(0,1)`) | §2.3 (non numérotée) | p. 5 |

**Identité de E2.** `D_p = D_b + τ_D` est *définitionnel* (§2.1), donc

```
D_b + (1−λ)·τ_D + λ·τ_A
  = (D_p − τ_D) + (1−λ)·τ_D + λ·τ_A
  = D_p + λ·(τ_A − τ_D)            ✓
```

L'identité est exacte **sur le support effectivement fusionné** : dans les familles
cross-architecture, certains paramètres sont exclus (§5, App. A.2), et l'identité ne
vaut alors que sur les tenseurs fusionnés.

**Plafond des coefficients (rectifié J6bis, MIN-01).** Les deux coefficients ne sont **pas**
traités de la même façon par le papier :

* **α n'est pas plafonné.** L'eq. (1) pose `α ≥ 0` sans borne supérieure ; la Table 6 va
  jusqu'à α = 1.4 (Ministral-3-8B → Nemotron-LD-8B) et les balayages de la Table 8 montrent des
  effets non linéaires au-delà. Un plafond α ∈ [0,1] serait donc un choix **hors papier**.
* **λ est borné par l'eq. (2)** : `D_Merge(λ) = D_b + (1−λ)·τ_D + λ·τ_A` n'est une
  combinaison convexe que pour `λ ∈ [0,1]` ; hors de cet intervalle, le rôle de D_b et D_p
  s'inverse. La bibliothèque **avertit** pour λ > 1 (et refuse λ < 0).

L'avertissement de la bibliothèque est émis à partir de 1,5 pour α (**choix de projet** :
seuil d'avertissement arbitraire ; la Fig. 8 du papier est le diagramme des succès communs,
elle ne justifie pas ce seuil) et à partir de 1 pour λ (borne de l'eq. 2).

---

## 3. Table 6 — sélection des coefficients (PDF p. 21)

> Caption : « *Table 6: Coefficient selection for each transfer family. Each coefficient is
> selected on its anchor benchmark and kept fixed on all other benchmarks. For the Nemotron
> Reasoning-donor row, the anchors are listed in transfer / merge order: IFEval without
> thinking and MATH-500 with thinking, respectively. Full sweeps are reported in Tables 8 and 7.* »

| AR → Diffusion | type de vecteur | ancre | transfert α | merge λ |
|---|---|---|---|---|
| Qwen2.5-7B → Dream-7B | instruct (SFT) | IFEval | **0.9** | **0.6** |
| Qwen2.5-Coder → Dream-Coder | code (SFT) | HumanEval+ | 0.6 | 0.2 |
| Qwen2.5-Coder → DiffuCoder | code (SFT) | HumanEval+ | 0.1 | 0.2 |
| AceCoder-Rule → DiffuCoder | code (RL) | MBPP+ | 0.2 | 0.5 |
| Qwen3-8B → DreamReasoner | reasoning | MATH-500 | 0.5 | 0.2 |
| Ministral-3-3B → Nemotron-LD-3B | instruct (SFT) | IFEval | 0.9 | 0.3 |
| Ministral-3-8B → Nemotron-LD-8B | instruct (SFT) | IFEval | **1.4** | 0.4 |
| Ministral-3-8B-Reasoning → Nemotron-LD-8B | reasoning | IFEval / MATH-500 | 0.8 | 0.2 |
| Gemma 4 → DiffusionGemma | SLAKE (SFT) | SLAKE | **1.0** | **0.1** |
| Gemma 4 → DiffusionGemma | VQA-RAD (SFT) | VQA-RAD | **0.7** | **0.3** |

Méthode de sélection (App. A.5, PDF p. 20) : « *For each family and construction, we fix an
anchor benchmark […] The coefficient is searched on the anchor only and then kept fixed on
every other benchmark.* » — le coefficient n'est **jamais** choisi « pour éviter une
dégénérescence ».

---

## 4. Cibles chiffrées

### 4.1 Table 1a — Instruction following (PDF p. 7)

> Caption : « *Table 1: Evaluation of A2D across tasks and model families. […] A2D (transfer)
> applies the AR donor's task vector to the diffusion recipient, while A2D (merge) composes it
> with diffusion post-training. Anchor tasks for scaling coefficient search are highlighted.
> All results are in percentages.* »

Sous-titre : « *(a) Instruction following vectors.* »

| Ligne | Qwen2.5-7B → Dream-7B : IFEval | : MATH-500 | : MBPP+ |
|---|---|---|---|
| AR donor | 75.3 | 36.6 | 69.0 |
| Base | **49.7** | **20.8** | 59.3 |
| A2D (transfer) | 59.2 | 40.2 | 62.7 |
| Instruct | **63.0** | **38.4** | 57.1 |
| A2D (merge) | 72.8 | 45.0 | 58.7 |

### 4.2 Table 1b — Vecteurs raisonnement / SFT / RL (PDF p. 7)

Sous-titre : « *(b) Reasoning, SFT and RL post-training vectors. DR abbreviates DreamReasoner.* »
Colonnes Gemma-4 → DiffusionGemma :

| Ligne | VQA-RAD | SLAKE | VQA-Med |
|---|---|---|---|
| AR donor | 75.8 | 66.5 | 44.0 |
| Base (= checkpoint diffusion **instruction-tuned**) | 57.4 | 63.2 | 43.2 |
| A2D (transfer) | 74.1 | 67.3 | 45.2 |
| Post-trained | 71.6 | 62.2 | 38.6 |
| A2D (merge) | 74.5 | 63.8 | 38.8 |

**Rôle des lignes pour DiffusionGemma** (App. A.1, PDF p. 18) : « *DiffusionGemma is
initialized from Gemma 4-26B-A4B-Instruct […] **It has no public base checkpoint**, so for this
pair we study domain-specific fine-tuning instead of the Base/Instruct setup, using the Gemma 4
and DiffusionGemma fine-tunes released by Van Puyvelde et al. (2026) for SLAKE, VQA-RAD, and
VQA-Med-2019, **with the two instruction-tuned checkpoints as the pre-task references**.* »
⇒ pour cette famille, `D_b` = le checkpoint IT, `D_p` = son fine-tune médical,
`τ_D` = le delta LoRA diffusion, `τ_A` = le delta LoRA AR (IT → médical).

### 4.3 Table 16 — Qwen2.5-7B → Dream-7B, six tâches (PDF p. 25)

> Caption : « *Table 16: Qwen2.5-7B →Dream-7B on six tasks. One transfer model (α=0.9) and one
> merged model (λ=0.6), both selected on IFEval, improve every task over the corresponding
> baseline. All results are in percentages.* »

| Ligne | IFEval | IFBench | GSM8K | MATH-500 | HumanEval+ | MBPP+ |
|---|---|---|---|---|---|---|
| AR donor (Qwen2.5-7B-Instruct) | 75.3 | 31.7 | 83.0 | 36.6 | 76.8 | 69.0 |
| **Base (Dream-7B-Base)** | **49.7** | 14.0 | **66.5** | **20.8** | 48.8 | 59.3 |
| A2D (transfer), α=0.9 | 59.2 | 25.0 | 81.7 | 40.2 | 54.9 | 62.7 |
| **Instruct (Dream-7B-Instruct)** | **63.0** | 19.0 | **83.0** | **38.4** | 51.8 | 57.1 |
| A2D (merge), λ=0.6 | 72.8 | 22.3 | 83.7 | 45.0 | 59.8 | 58.7 |

### 4.4 Table 21 — géométrie des vecteurs (PDF p. 28)

> Caption : « *Table 21: Geometry of post-training and paradigm-conversion directions across
> model pairs. Measurements use backbone parameters and, for Nemotron, compatible input
> embeddings after key mapping, excluding protected rows. Output heads are excluded
> throughout.* »

App. B.7 (PDF p. 25-26) : « *we concatenate parameter differences over **backbone tensors
(attention, MLP, and normalization layers)** […] All measurements **exclude output heads**,
the other model families also **exclude all input embeddings**.* » et
« *We define **Δ_b = D_b − A_b**, **Δ_p = D_p − A_p**.* »

| AR → Diffusion | type | ‖τ_A‖ | ‖τ_D‖ | ‖Δ_b‖ | ‖Δ_p‖ | cos(Δ_b,Δ_p) | cos(τ_A,τ_D) |
|---|---|---|---|---|---|---|---|
| **Qwen2.5-7B → Dream-7B** | instruct (SFT) | **23.0** | **31.5** | **729** | **732** | **+0.999** | **+0.013** |
| Qwen2.5-Coder → Dream-Coder | code (SFT) | 87.0 | 84.6 | 589 | 602 | +0.980 | +0.023 |
| Qwen3-8B → DreamReasoner | reasoning (SFT / SFT+RL) | 189.2 | 295.8 | 459 | 556 | +0.803 | +0.118 |
| Ministral-3 → Nemotron-LD-3B | instruct (SFT), cross-arch. | 43.1 | 17.1 | 280.9 | 271.4 | +0.987 | +0.075 |
| Ministral-3 → Nemotron-LD-8B | instruct (SFT), cross-arch. | 42.2 | 16.3 | 387.0 | 380.4 | +0.994 | +0.070 |
| Qwen2.5-Coder → DiffuCoder | code (SFT) | 87.0 | 33.3 | 627 | 630 | +0.989 | +0.005 |
| AceCoder-Rule → DiffuCoder | code (RL) | 0.68 | 4.10 | 630 | 630 | +1.000 | +0.004 |

**Contraste, même tableau** (vecteurs de continued-pretraining, 70–82× plus gros) :
`‖v_A‖`/`‖v_D‖` = 1603.6/1460.9 (code) et 1876.9/— (math), `cos(v_A,v_D)` = +0.869 (code).
Ces vecteurs **ne transfèrent pas** (Table 15, PDF p. 25).

---

## 5. Règles de mapping des poids (App. A.2, PDF p. 19)

| Famille | Noms/formes | Embeddings | Normes | Tête de sortie | Routeur / experts | Vision |
|---|---|---|---|---|---|---|
| **Qwen-derived** (Dream, Dream-Coder, DiffuCoder, DreamReasoner) | identiques à ceux du parent AR → *« direct element-wise combination of all parameters, with no remapping, freezing, or masking* » | fusionnés | fusionnées | fusionnée | — | — |
| **Nemotron / Ministral** | mapping explicite ; 14 ids de vocabulaire divergents (+ ids 34–35 Base vs Instruct) | 16 lignes **protégées** (valeur du receveur), exclues de `+τ_A` et de `−τ_D` ; +1 ligne (id 100, token de masque) gelée en Base-transfer | fusionnées (backbone) | **valeur du receveur** | paramètres sans équivalent AR : gardés du receveur | — |
| **DiffusionGemma / Gemma 4** | mapping AR → projections du decoder ; encodeur et décodeur **partagent** les poids de base | **inchangés** | **inchangées** | **inchangée** | routeur / experts : **équivalents AR** (Gemma 4 en a) → **fusionnés par défaut**, famille déclarée (`--router-policy skip` pour ne pas les fusionner) ; **self-conditioning** (projections propres à la diffusion) : **pas d'équivalent AR** → conservées ; adapters « *stay unmerged and are scaled according to the experimental formula* » | **inchangée** |

Pour DiffusionGemma, « *The diffusion encoder and decoder share their base weights, so each AR
update is applied once to the shared parameter and affects both branches.* »

**Fusion des tenseurs Dream/Qwen** (App. A.2) : « *Tokens that exist only in the diffusion
tokenizer need no special treatment. The mask token and the other ids beyond the AR vocabulary
never occur in the AR model's training data, so τ_A is exactly zero on their input-embedding
rows for the Qwen2.5-based vectors and small for Qwen3. We merge them with the same formula as
every other row.* »

**Construction des checkpoints fusionnés** (App. A.3, PDF p. 20) : « *We construct merged
checkpoints using **sequential CPU FP32 arithmetic, followed by a single cast to BF16**,
retaining the Dream base model's configuration and tokenizer.* »

---

## 6. Protocole d'évaluation (App. A.1, PDF p. 17-18)

**Dream-7B / Qwen2.5-7B** : « *IFEval, GSM8K, MATH-500 and HumanEval run in the Dream fork of
lm-evaluation-harness at its released decoding settings and per-task budgets, with the chat
template and **one denoising step per generated token**. IFEval is the mean of the four lm-eval
metrics, and GSM8K and MATH-500 are 0-shot. MATH-500 replaces the release's full MATH task at
its 512-token budget.* »

* **IFEval** : fork Dream, `max_gen_toks` 1280 ; métrique = moyenne des 4 métriques lm-eval
  (prompt/inst × strict/loose).
* **GSM8K** : 0-shot, budget 256 tokens ; prompt système (cité mot pour mot) :
  > « *Solve the math problem with brief, direct reasoning. Do NOT restate the problem, do NOT
  > use bullet points, and use as few steps as possible. On the final line, write the answer as
  > '#### \<number\>' and nothing else.* »
* **MATH-500** : 0-shot, budget 512 tokens ; extraction Minerva + équivalence symbolique,
  repli Math-Verify.
* **A.3 (PDF p. 20)** pour le réglage contrôlé §2 : « *Dream uses diffusion-step budgets equal to
  the generation-token budgets, entropy-based unmasking, **temperature 0.1, and top-p 0.9**.* »
  Budgets : 512 tokens pour GSM8K et le code, 1 024 pour MATH-500 dans ce cadre.
* IFEval ≈ 541 prompts (Table 11, Figure 8), MATH-500 = 500 problèmes, HumanEval = 164,
  MBPP = 378 (App. A.3). **Aucune graine** n'est donnée dans le papier.

**DiffusionGemma** (App. A.1, PDF p. 18) : décodeur block-diffusion natif (Transformers),
bloc 256, ≤ 48 pas de débruitage par bloc, **entropy bound 0.1**, température **0.8 → 0.4**
(annealée), arrêt anticipé quand les prédictions cessent de changer, thinking désactivé ;
256 tokens ; 0-shot ; SLAKE (1 061 questions), VQA-RAD (451), VQA-Med-2019 (500) ;
**jugge = Qwen3.8-27B**, greedy, thinking désactivé.

**DiffusionGemma (rapport technique, Table 2 et Algorithm 1, PDF p. 3 et 10)** :
canvas 256, `N = 48` pas de débruitage max, seuil d'arrêt entropique `e_stop = 0.005`,
borne d'entropie `b = 0.1`, température `τ_max = 0.8 → τ_min = 0.4`, diffusion multinomiale
(uniforme) et non masquée, ~12 pas effectifs en moyenne.

---

## 7. Correspondance implémentation ↔ papier

| Élément du papier | Implémentation dans ce dépôt |
|---|---|
| eq. (1) transfert `D_b + ατ_A` | `scripts/merge_a2d.py --mode transfer` |
| eq. (2) merge `D_p + λ(τ_A − τ_D)` | `scripts/merge_a2d.py --mode merge` |
| ablation `D_p + ατ_A` (Table 9) | `scripts/merge_a2d.py --mode ablation-dp-plus-tau-a` |
| ablation `D_p − λτ_D` (Table 10) | `scripts/merge_a2d.py --mode ablation-dp-minus-tau-d` |
| `τ_A = A_p − A_b` (paire de checkpoints) | `a2d.sources.CheckpointPairSource` |
| `τ` d'un adaptateur LoRA `(α/r)·B·A` (App. A.2 : adapters publiés) | `a2d.sources.LoraAdapterSource` |
| mapping Qwen→Dream (identité, tout fusionné) | `a2d.naming.IDENTITY` |
| mapping Gemma4→DiffusionGemma (embeddings/normes/tête inchangés) | `a2d.naming.GEMMA4_TO_DIFFUSIONGEMMA` (réutilise `scripts/_common.py`) |
| FP32 séquentiel + un seul cast BF16 (App. A.3) | `a2d.merge.merge_checkpoint` |
| contrôle post-écriture `α_eff` / `λ_eff` | `a2d.merge.verify_written` + `scripts/verify_merge.py` |
| Table 21 (géométrie) | `scripts/measure_geometry.py` |
| protocole Dream (App. A.1) | `harness/dream/` (harnais versionné) + `eval/dream_a2d/` (résultats) |

---

## 8. Mesures locales de la Table 21 (Dream-7B) — **[mesuré J6]**

Recalculées le 2026-10-20 avec `scripts/measure_geometry.py` sur les tenseurs du backbone
(attention + MLP + norms ; embeddings et tête exclus), entre
`Qwen/Qwen2.5-7B` ↔ `Qwen/Qwen2.5-7B-Instruct` et `Dream-org/Dream-v0-Base-7B` ↔
`Dream-org/Dream-v0-Instruct-7B` (337 tenseurs) :

| Grandeur | Mesuré J6 | Table 21 | Écart |
|---|---|---|---|
| ‖τ_A‖ | 23.01 | 23.0 | +0.05 % |
| ‖τ_D‖ | 31.47 | 31.5 | −0.11 % |
| ‖Δ_b‖ | 729.04 | 729 | +0.01 % |
| ‖Δ_p‖ | 732.14 | 732 | +0.02 % |
| cos(Δ_b, Δ_p) | +0.9986 | +0.999 | — |
| cos(τ_A, τ_D) | +0.0130 | +0.013 | — |

Détail par famille : `attn` (196 tenseurs), `mlp` (84), `norms` (57) — rapport complet :
`_artefact/j6/geometry_dream.json` (non versionné) et `docs/reports/J6.md`.
