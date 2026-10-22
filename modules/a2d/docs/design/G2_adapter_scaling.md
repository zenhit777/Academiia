# G2 — Mise à l'échelle des adaptateurs de diffusion (DiffusionGemma) — **IMPLÉMENTÉ**

**Date** : 2026-10-20 (conception) · **Implémentation** : 2026-10-21 (J8, phase B) ·
**Statut** : mode `merge-dg-scaled` disponible et testé ; l'**évaluation** G2 reste
conditionnée à une machine capable de charger le décodeur DG 26B (voir §5.3).

> Ce document est le cahier des charges de la porte G2. La **phase B de J8** a
> implémenté le mode « base + adaptateur remis à l'échelle » dans la
> bibliothèque : `modules/a2d/a2d/adapters.py`, `MergeConfig(mode="merge-dg-scaled")`
> et les options CLI `--mode merge-dg-scaled` (alias `--mode adapter-scale`),
> `--adapter-dtype`, `--adapter-output-name`. Les tests couvrent la formule,
> l'absence de fusion de τ_D, la mise à l'échelle, la détection de mutation et
> la rechargeabilité du checkpoint. La porte G2 elle-même (évaluation médicale)
> n'est **pas** franchie : elle exige le protocole App. A.1 complet.

## 1. Ce que dit le papier

**App. A.2, « DiffusionGemma / Gemma 4 »** (PDF p. 19, extrait
`_artefact/a2d_paper.txt:1531-1539`) :

> « We map the AR updates to the corresponding DiffusionGemma language-model
> projections. The diffusion encoder and decoder **share their base weights**, so
> each AR update is applied **once** to the shared parameter and affects both
> branches. The diffusion fine-tuning adapters, which hold **separate encoder and
> decoder updates, stay unmerged and are scaled according to the experimental
> formula**. Diffusion-specific adapter updates such as the self-conditioning
> projections have **no AR counterpart**. The vision encoder, embeddings,
> normalization parameters, and output head are unchanged, and all merged models
> keep the diffusion recipient's tokenizer and chat template. »

**Éq. (2)** (PDF p. 6, `_artefact/a2d_paper.txt:464`) :

```
D_Merge(λ) = D_b + (1−λ)·τ_D + λ·τ_A = D_p + λ·(τ_A − τ_D),   λ ∈ [0,1]
```

**App. A.1** (PDF p. 18, `_artefact/a2d_paper.txt:1396-1403`) : DiffusionGemma
« *has no public base checkpoint, so for this pair we study domain-specific
fine-tuning instead of the Base/Instruct setup, using the Gemma 4 and
DiffusionGemma fine-tunes released by Van Puyvelde et al. (2026) for SLAKE,
VQA-RAD, and VQA-Med-2019, **with the two instruction-tuned checkpoints as the
pre-task references*** ».

**Table 7 / Table 8** (PDF p. 24-25) : « *For DiffusionGemma, **D_b denotes the
instruction-tuned reference checkpoint*** ».

**Ancres** (Table 6/7/8) : SLAKE → **α = 1,0** (transfert) / **λ = 0,1**
(merge) ; VQA-RAD (SFT) → **α = 0,7** / **λ = 0,3**.

## 2. La reformulation pour DiffusionGemma

Pour DG, il n'y a pas de paire « base / post-entraîné » de checkpoints
publics : le « base » est le **checkpoint IT**, et le « post-training » est un
**adaptateur LoRA publié** (`gevaertlab/diffusiongemma-radiology-vqa`, §
`modules/a2d/docs/VP_ADAPTERS.md`). Donc :

```
D_b  = D_p  = checkpoint IT de DiffusionGemma           (D_p = D_b + adaptateurs)
τ_A  = Gemma4-IT − Gemma4-Base      (vecteur AR, domaine médical : `ar-*`)
τ_D  = adaptateur de diffusion publié (`diffusion-*`), appliqué SANS fusion
```

L'équation (2) devient :

```
D_Merge(λ) = D_b + λ·τ_A + (1−λ)·τ_D
```

soit, en décomposant le receveur :

```
D_Merge(λ) = [ D_b + λ·τ_A ]  +  (1−λ)·τ_D
             └──── base' ────┘     └─ adaptateur mis à l'échelle ─┘
```

**Interprétation opérationnelle (le point central de ce design)** :

> On **n'écrit pas** l'adaptateur dans les poids. On écrit un **checkpoint de
> base modifié** `base' = D_b + λ·τ_A`, et on **conserve l'adaptateur séparé**,
> dont l'échelle est réduite d'un facteur `(1−λ)`. À l'inférence, le modèle
> est `base' + adaptateur_à_l'échelle`.

C'est exactement ce que l'App. A.2 impose (« stay unmerged and are scaled »).
Les modes `merge` / `ablation-*` continuent de fusionner un `τ_D` dans les
poids (c'est leur définition) ; le mode **`merge-dg-scaled`** ajouté en J8 est
le seul qui traite `τ_D` comme une dépendance d'inférence externe.

## 3. Le mode implémenté : « base + adaptateur remis à l'échelle »

### 3.1 Interface (implémentée)

```
python modules/a2d/cli/merge_a2d.py --mode merge-dg-scaled \
    --diffusion checkpoints/diffusiongemma_it \
    --tau-a-post <gemma4_it> --tau-a-base <gemma4_base> \
    --tau-d-adapter checkpoints/dg_adapters/diffusion-vqarad \
    --lambda 0.3 --mapping gemma4-diffusiongemma \
    --coverage strict \
    --output checkpoints/diffusiongemma_merge_l03
```

Sortie attendue (deux artefacts, **un seul modèle** au moment de l'inférence) :

| Artefact | Contenu |
|---|---|
| `checkpoints/diffusiongemma_merge_l03/` | `base' = D_b + λ·τ_A` (BF16, écrit par le writer A2D) + `merge_config.json` |
| `…/adapter_scaled/` (ou sidecar `a2d_adapter.json` + dossier d'adaptateur) | l'adaptateur LoRA de diffusion, **non fusionné**, avec l'échelle `(1−λ)` |

### 3.2 Trois façons de mettre l'adaptateur à l'échelle — analyse

| Option | Principe | Exactitude | Risque | Verdict |
|---|---|---|---|---|
| **(a)** réécrire `lora_alpha` dans `adapter_config.json` : `α' = α·(1−λ)` | l'échelle PEFT est `α/r` ; changer `α` change l'échelle | **approximative** si `α·(1−λ)` n'est pas un multiple simple de `r` (ex. α=128, λ=0,3 ⇒ α'=89,6 ; l'échelle PEFT devient 89,6/64 = 1,4 au lieu de 2,0·0,7 = 1,4 — **exact** ici, mais fragile en général) | PEFT peut transtyper `lora_alpha` en `int` | acceptable **si** l'échelle effective est **vérifiée numériquement** après écriture |
| **(b)** multiplier les tenseurs `lora_B` par `(1−λ)` (écriture en BF16) | `ΔW = (α/r)·B·A` ⇒ `(α/r)·(s·B)·A = s·ΔW` | **exacte** en arithmétique FP32, mais l'écriture en BF16 du `B` mis à l'échelle ajoute un arrondi sur l'adaptateur lui-même | arrondi BF16 sur `B` (perte sur l'adaptateur, pas sur les poids) | **recommandée**, à condition de conserver `lora_alpha` intact et de vérifier l'écart |
| **(c)** charger l'adaptateur avec un facteur externe | facteur appliqué à l'inférence | exacte si le facteur est appliqué en FP32 | dépend du code d'inférence (HF / vLLM / transformers) | **recommandée comme moyen de contrôle** (permet de vérifier (b) contre (c)) |

**Décision de conception (implémentée)** : (b) comme mode d'écriture
(`modules/a2d/a2d/adapters.py::scale_adapter`), **plus** (c) comme procédure de
vérification (`verify_scaled_adapter`). `lora_alpha` reste **intact** dans
`adapter_config.json` ; seul `lora_B` est multiplié. Le dtype d'écriture est
BF16 par défaut (`--adapter-dtype bfloat16`) ou FP32 (`--adapter-dtype float32`).

### 3.3 Ce que la bibliothèque sait faire (implémenté en J8)

| Besoin | État |
|---|---|
| Produire `base' = D_b + λ·τ_A` **sans** toucher à l'adaptateur | **fait** : `mode="merge-dg-scaled"`, plan limité aux modules de τ_A ; `required_sources()` = `{A}` |
| Écrire un adaptateur **mis à l'échelle** | **fait** : `scale_adapter()` copie le dossier, multiplie `lora_B` par `(1−λ)`, conserve `lora_A`, `adapter_config.json` et les métadonnées safetensors |
| Déclarer dans le manifeste l'adaptateur utilisé et son échelle | **fait** : `merge_config.json["diffusion_adapter"]` (chemins, `r`, `lora_alpha`, `factor`, `scale_before/after`, SHA-256 des shards avant/après, branches encodeur/décodeur/self-conditioning) |
| Vérifier que l'adaptateur n'est **pas** dans les poids | **fait** : `tau_d_merged_into_weights=false`, plan sans τ_D, tenseurs hors plan bit-identiques, `max_ulp=0` |
| Vérifier l'échelle de l'adaptateur | **fait** : `verify_scaled_adapter()` (V5/V6), exécuté après écriture et disponible via `verify_merge.py --merged-adapter` |
| Refuser un τ_D qui n'est pas un adaptateur LoRA | **fait** : `MergeError` explicite (un `D_p − D_b` doit passer par `--mode merge`) |

## 4. Poids partagés encodeur / décodeur — le point technique central

### 4.1 Ce que dit le checkpoint (vérifié le 2026-10-20)

Lecture de `checkpoints/diffusiongemma_it/model.safetensors.index.json`
(1 047 tenseurs) :

| Emplacement | Contenu mesuré |
|---|---|
| `model.decoder.layers.N.*` | 22 tenseurs par couche : `self_attn.{q,k,v,o}_proj`, `self_attn.{q,k}_norm`, `mlp.{gate,up,down}_proj`, `experts.{gate_up,down}_proj`, `router.{proj,scale,per_expert_scale}`, `input_layernorm`, `post_attention_layernorm`, `pre/post_feedforward_layernorm[_1,_2]`, `layer_scalar` |
| `model.encoder.language_model.layers.N.*` | **1 seul tenseur par couche : `layer_scalar`** |
| `model.decoder.self_conditioning.*` | 4 tenseurs (`gate_proj`, `up_proj`, `down_proj`, `pre_norm`) |
| `model.encoder.vision_tower.*` + `model.encoder.embed_vision.*` | 356 tenseurs (vision) |
| `model.decoder.embed_tokens.weight` | 1 (embeddings) |

⇒ **Les poids de base de l'encodeur et du décodeur sont RÉELLEMENT partagés** :
le checkpoint ne stocke **qu'une seule copie** des poids d'attention et de MLP.
L'App. A.2 le dit (« *share their base weights* ») et l'index le confirme.

**Conséquence n° 1** : le vecteur AR `τ_A` s'applique **une seule fois**, sur
`model.decoder.layers.N.<proj>`. Le mapping existant
(`academiia/core/common.py::gemma4_to_diffusiongemma`, `model.language_model.layers.X`
→ `model.decoder.layers.X`) est donc **correct** et il n'y a **rien à
dupliquer**. C'est aussi ce que le build legacy a fait (355 tenseurs : attn
115 + mlp 90 + experts 60 + routeur 90).

**Conséquence n° 2 (le piège)** : l'adaptateur de diffusion publié contient
**deux jeux** de modules LoRA sur les **mêmes poids partagés** :

* `model.decoder.layers.N.{q,k,v,o,gate,up,down}_proj` — **205 modules** ;
* `model.encoder.language_model.layers.N.{...}` — **205 modules** ;
* `model.decoder.self_conditioning.{gate,up,down}_proj` — **3 modules** ;
* total **413 modules** (`modules/a2d/docs/VP_ADAPTERS.md` §1.3).

À l'inférence, les deux jeux s'appliquent au **même** tenseur de base : le
delta effectif est **la somme** des deux :

```
ΔW_partagé(module) = ΔW_decoder(module) + ΔW_encoder(module)
```

C'est le « double comptage » signalé dans `modules/a2d/docs/VP_ADAPTERS.md` §3. Il n'est
pas un défaut des adaptateurs : c'est la structure du modèle DG (poids
partagés, updates séparés encodeur/décodeur).

**Conséquence n° 3** : `τ_D` dans l'équation (2) est le delta **effectif**, donc
la **somme** des deux jeux. Toute vérification numérique de la formule (2) doit
donc comparer à `(1−λ)·(ΔW_dec + ΔW_enc)`, pas à `(1−λ)·ΔW_dec`.

**Conséquence n° 4** : les `self_conditioning` n'ont **aucun équivalent AR**
(App. A.2) : ils reçoivent `(1−λ)·ΔW` et rien d'autre. Ils sont de la famille
« non appariée » : ni τ_A ni — dans le sens de l'équation (2) — le receveur ne
les touche.

### 4.2 Ce que le plan de fusion doit contenir pour DG

| Module du receveur | τ_A (AR) | Adaptateur (τ_D) | Traitement |
|---|---|---|---|
| `model.decoder.layers.N.self_attn.{q,k,v,o}_proj` | ✔ (si présent dans l'adaptateur AR) | ✔ (decodeur + encodeur) | `base' = D_b + λ·τ_A` **puis** adaptateur à `(1−λ)` |
| `model.decoder.layers.N.mlp.{gate,up,down}_proj` | ✔ | ✔ | idem |
| `model.decoder.layers.N.experts.*`, `router.*` | ✘ (gelés côté AR) | ✘ | **inchangés** |
| `model.decoder.layers.N.*_norm`, `layer_scalar` | ✘ | ✘ | **inchangés** (App. A.2 : norms unchanged) |
| `model.decoder.self_conditioning.*` | ✘ (pas d'équivalent AR) | ✔ | `base' = D_b` + adaptateur à `(1−λ)` |
| `model.decoder.embed_tokens.weight` | ✘ (embeddings unchanged) | ✘ | **inchangé** |
| `model.encoder.vision_tower.*`, `embed_vision.*` | ✘ | ✘ | **inchangé** |

⇒ Le plan de fusion de `base'` ne contient que les **205 modules** du vecteur
AR (comme le build legacy : 205 des 355 du legacy incluaient experts et
routeur, qui ne sont ici **pas** dans τ_A). Les 842 autres tenseurs du
checkpoint DG doivent être **bit-identiques** à `D_b`.

### 4.3 Point de vigilance sur les familles

Le mapping `GEMMA4_TO_DIFFUSIONGEMMA` déclare `fusible_families = {attn, mlp,
experts, router}`. Pour le **vecteur AR** DG, seules `attn` et `mlp` portent
réellement des modules de τ_A ; `experts` et `router` ne sont pas dans τ_A et
seront donc absents du plan (le plan est piloté par les modules de τ_A). La
famille déclarée reste un **filtre côté receveur** : elle empêche de fusionner
par erreur les norms, embeddings, tête ou vision. **Aucune modification du
mapping n'est nécessaire** — mais le manifeste doit le dire (famille, effectif,
couverture) pour que la traçabilité soit complète.

## 5. Plan de vérification

### 5.1 Vérifications mécaniques (CPU, avant toute évaluation)

| # | Contrôle | Moyen | Critère |
|---|---|---|---|
| V1 | **Identité numérique de la formule** | recalcul de `base' + (1−λ)·τ_D_effectif` et comparaison à `D_b + λ·τ_A + (1−λ)·τ_D` | écart max **0 ULP** BF16 par élément (`max_ulp=0` : égalité avec `round_bf16(expected)`) |
| V2 | **Projection signée** (réutiliser la vérification v2) | `modules/a2d/cli/verify_merge.py` sur le couple (base', D_b) avec τ_A | `c_a_eff = λ` dans la tolérance ; `c_d_eff` non applicable (τ_D n'est pas dans les poids) |
| V3 | **τ_D n'est PAS dans les poids** | sur le `base'` produit : `base' − D_b − λ·τ_A` doit être **exactement nul** sur les modules de τ_A | égalité avec l'arrondi BF16 attendu (`max_ulp=0`) ; `tau_d_merged_into_weights=false` |
| V4 | **Tenseurs non appariés inchangés** | comparaison binaire `base'` vs `D_b` sur les 842 tenseurs hors plan | **bit-identiques** (contrôle déjà implémenté dans la vérification v2) |
| V5 | **Échelle effective de l'adaptateur** | recharger l'adaptateur original avec le facteur `(1−λ)` (option (c)) et comparer ΔW | `cos(ΔW_scaled, (1−λ)·ΔW) ≥ 1 − 10⁻⁴` et `‖ΔW_scaled − (1−λ)·ΔW‖ / ‖(1−λ)·ΔW‖ ≤ 10⁻²` en BF16 (arrondi de `lora_B`, ε=2⁻⁸) ; `≤ 10⁻³` en FP32 |
| V6 | **Symétrie encodeur/décodeur** | vérifier que les deux jeux de l'adaptateur sont **tous deux** présents et **tous deux** mis à l'échelle | les 413 modules du `diffusion-*` sont présents dans l'adaptateur produit, avec la même facteur |
| V7 | **Provenance** | manifeste : SHA-256 de l'adaptateur **avant** et **après**, `r`, `α`, `α'`, facteur effectif, commit git | toutes les valeurs présentes et vérifiables |
| V8 | **Le plan est non vide** | refus si `n_fused == 0` (déjà implémenté : `MergeError` sauf `--allow-empty`) | aucun `--allow-empty` utilisé |

### 5.2 Vérification expérimentale (GPU, protocole App. A.1)

Protocole **exact** (App. A.1, `_artefact/a2d_paper.txt:1404-1411`) :

| Paramètre | Valeur |
|---|---|
| Décodeur | implémentation **Transformers** du décodeur block-diffusion natif |
| Longueur de bloc | 256 |
| Pas de débruitage | ≤ 48 par bloc |
| Borne d'entropie | 0,1 |
| Température | annealée de 0,8 à 0,4 |
| Arrêt précoce | dès que les prédictions ne changent plus entre pas |
| Thinking | **désactivé** |
| Format | image suivie de la question, **0-shot**, budget **256 tokens** |
| Jeux | SLAKE **1 061** questions de test (anglais) · VQA-RAD **451** questions publiques · VQA-Med-2019 **500** questions |
| Jugement | **Qwen3.8-27B**, décodage greedy, thinking désactivé ; **précision sémantique** |

Modèles à comparer (avec les ancres de la Table 6/7) :

| Modèle | α / λ | Ancre |
|---|---|---|
| `D_b` = `diffusiongemma_it` | — | référence |
| Transfert | **α = 1,0** (SLAKE) / **α = 0,7** (VQA-RAD) | SLAKE / VQA-RAD (SFT) |
| Merge (adaptateur à l'échelle) | **λ = 0,1** (SLAKE) / **λ = 0,3** (VQA-RAD) | idem |

Cible de l'article : « *A2D-Transfer yields … **+16.7 on VQA-RAD** » et
« *A2D-Merge is consistently effective, improving all 9/9 post-trained
baselines* » (Table 1b, `_artefact/a2d_paper.txt:765-770`).

### 5.3 Faisabilité locale (à mesurer avant de s'engager)

* **Coût GPU** : le décodeur DG (bloc 256, ≤48 pas, 3 jeux de 2 012 questions
  au total, avec un **juge LLM** de 27 B) est **hors du budget** de cette
  machine (RTX 5090 32 Go) tel quel : le juge Qwen3.8-27B en FP8 tient
 à peine, et le décodeur DG 26B en BF16 (**48,1 GiB**, mesuré sur le build
  legacy) **ne tient pas** dans 31,84 GiB. → G2 exige soit un **quantification**
  (ce qui change le protocole et doit être déclaré), soit un **découpage**
  (évaluer par blocs de modules ? non — le modèle est monolithique), soit une
  machine plus grande. **Ce point n'est pas résolu par ce document.**
* **Les adaptateurs publiés** (2,68 Go pour les 6 sous-dossiers) doivent être
  téléchargés ; l'inspection headers-only de J6 a déjà donné toute l'information
  structurelle nécessaire.

## 6. Hypothèses **non vérifiées** contre le PDF

| # | Hypothèse | Statut | Comment la trancher |
|---|---|---|---|
| H1 | La « experimental formula » de l'App. A.2 est l'équation (2) appliquée à `τ_D` = delta effectif des adaptateurs, donc facteur `(1−λ)` sur l'adaptateur | **[INFERENCE]** — l'App. A.2 **renvoie** à « the experimental formula » **sans la redonner** ; l'équation (2) est en Section 2 | lire la Section 2 / l'App. B du papier ; à défaut, la reconstruction est **cohérente** avec l'équation (2) et avec `D_p = D_b + adaptateurs` |
| H2 | Le facteur `(1−λ)` s'applique à l'**ensemble** des adaptateurs (decodeur + encodeur + self-conditioning) | **[INFERENCE]** — l'App. A.2 dit que les adaptateurs « stay unmerged and are scaled », sans préciser module par module | idem ; le cas self-conditioning est explicitement « no AR counterpart », ce qui implique qu'il reçoit le facteur `(1−λ)` et rien d'autre |
| H3 | Le papier met à l'échelle `lora_alpha` plutôt que les poids `B` | **[NON VÉRIFIÉ]** — le texte ne le dit pas | sans impact sur le résultat (les deux sont numériquement équivalents à l'arrondi BF16 près) ; **V5** tranche |
| H4 | À l'inférence, **les deux** jeux de l'adaptateur (decodeur et encodeur) sont appliqués simultanément | **[INFERENCE forte]** — l'adaptateur publié contient les deux et le modèle a des poids partagés ; l'App. A.2 mentionne « separate encoder and decoder updates » | charger l'adaptateur sur le modèle DG et mesurer `‖ΔW_total‖` ; comparer à `‖ΔW_dec + ΔW_enc‖` |
| H5 | Le « D_b » du merge DG est bien le checkpoint IT (et non un base public) | **VÉRIFIÉ** — Table 7/8 (« *D_b denotes the instruction-tuned reference checkpoint* ») + App. A.1 (« *the two instruction-tuned checkpoints as the pre-task references* ») | — |
| H6 | Le mapping AR → DG applique chaque mise à jour AR **une fois** sur le poids partagé | **VÉRIFIÉ** (App. A.2) **et** confirmé par l'index du checkpoint (une seule copie des poids) | — |
| H7 | Les `self_conditioning` sont adaptés par l'adaptateur de diffusion | **VÉRIFIÉ** (`modules/a2d/docs/VP_ADAPTERS.md` §1.3 : 3 modules `self_conditioning.{gate,up,down}_proj`) | — |
| H8 | Le récepteur DG de l'adaptateur publié est bien `diffusiongemma-26B-A4B-it` (le checkpoint IT local) | **VÉRIFIÉ** (`adapter_config.json` : `base_model_name_or_path: google/diffusiongemma-26B-A4B-it`) | — |
| H9 | L'adaptateur `ar-*` (Gemma 4) et l'adaptateur `diffusion-*` ont été entraînés sur le **même** jeu (SLAKE / VQA-RAD / VQA-Med) | **VÉRIFIÉ** par le nommage des sous-dossiers (`ar-slake` ↔ `diffusion-slake`, …) | — |
| H10 | Les poids `experts` de DG sont bien gelés par l'adaptateur (aucun module LoRA) | **VÉRIFIÉ** — la regex `target_modules` ne matche pas `gate_up_proj`/`down_proj` d'experts, et les experts sont des paramètres empilés 3-D | — |

## 7. Décision

**Le blocage bibliothèque est levé (J8, phase B).** Le mode
`merge-dg-scaled` (option (b) + vérification V5, alias `adapter-scale`) est
implémenté, testé (formule, non-fusion de τ_D, échelle, mutation, recharge
`from_pretrained`) et documenté. La porte **G2 reste non franchie** pour une
raison de **moyens de calcul** : le protocole App. A.1 (décodeur DG 26B en
BF16 ≈ 48 GiB + juge 27B) ne tient pas sur une RTX 5090 32 Go sans
quantification, qui changerait le protocole et devrait être déclarée.

**Ordre de travail** :

1. J7 (Dream) : transfert α=0,9 et merge λ=0,6, évalués selon
   `modules/a2d/docs/design/G1_PROTOCOL.md` → **G1** ;
2. implémentation du mode « adaptateur à l'échelle » + V1–V8 → déblocage
   **G2** ;
3. G2 : évaluation DG sur SLAKE / VQA-RAD / VQA-Med-2019 avec le protocole
   App. A.1 et les ancres α = 1,0 / 0,7 et λ = 0,1 / 0,3 ;
4. G1b (Nemotron-3B) : faisabilité déjà instruite dans
   `modules/a2d/docs/design/G1B_NEMOTRON3B.md` ; à placer entre G1 et G2 selon la
   disponibilité des poids.

## 8. Implémentation (J8, phase B)

| Élément | Emplacement |
|---|---|
| Mise à l'échelle + vérification | `modules/a2d/a2d/adapters.py` (`scale_adapter`, `verify_scaled_adapter`) |
| Mode de fusion | `modules/a2d/a2d/merge.py` (`MODES`, `mode_coefficients`, `required_sources`, `validate_config`, `merge_checkpoint`) |
| CLI | `modules/a2d/cli/merge_a2d.py --mode merge-dg-scaled` (alias `adapter-scale`), `--adapter-dtype`, `--adapter-output-name` |
| Vérification post hoc | `modules/a2d/cli/verify_merge.py --mode merge-dg-scaled --merged-adapter ...` |
| Manifeste | `merge_config.json` : `diffusion_adapter`, `diffusion_adapter_verification`, `tau_d_merged_into_weights` |
| Tests | `modules/a2d/tests/test_a2d_library.py` (formule, non-fusion, échelle, recharge) ; `modules/a2d/tests/test_a2d_mutation.py` (facteur erroné, refus d'écrasement) |

Limites **non vérifiées** contre le PDF (inchangées) : H1–H4 du §6. En
particulier, le facteur `(1−λ)` appliqué à **toutes** les branches (décodeur,
encodeur, self-conditioning) reste une **inférence** cohérente avec l'eq. (2),
mais l'App. A.2 ne détaille pas module par module.
