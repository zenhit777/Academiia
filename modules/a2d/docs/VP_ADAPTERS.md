# Inspections externes — adaptateurs médicaux et dépôt de référence

Inspection **LECTURE SEULE** du 2026-10-20, dans le cadre de la mission J6 (tâche A3).
Rien n'a été écrit dans les dépôts externes ; aucun adaptateur n'a été téléchargé en
intégralité (voir §4, justification).

---

## 1. HuggingFace — `gevaertlab/diffusiongemma-radiology-vqa`

* **Public**, non gated, licence `apache-2.0`, dernière modification **2026-07-04**.
* Tags : `peft`, `safetensors`, `medical`, `radiology`, `vqa`, `medical-imaging`, `lora`,
  `diffusion-llm`, `image-text-to-text`, `arxiv:2607.01436`,
  `base_model:google/diffusiongemma-26B-A4B-it`, `base_model:adapter:google/diffusiongemma-26B-A4B-it`.
* `base_model` (carte) : `google/diffusiongemma-26B-A4B-it` **et** `google/gemma-4-26B-A4B-it`.
* Article associé (carte) : *Discrete Diffusion Language Models for Interactive Radiology Report
  Drafting*, arXiv:2607.01436 (Van Puyvelde et al.).

### 1.1 Arborescence (API `/tree`, tailles exactes)

| Sous-dossier | `adapter_config.json` | `adapter_model.safetensors` | `README.md` |
|---|---|---|---|
| `ar-slake` | 1 196 o | 297 430 856 o (284 MiB) | 5 174 o |
| `ar-vqamed` | 1 196 o | 297 430 856 o | 5 174 o |
| `ar-vqarad` | 1 196 o | 297 430 856 o | 5 174 o |
| `diffusion-slake` | 1 222 o | 598 647 960 o (571 MiB) | 5 188 o |
| `diffusion-vqamed` | 1 222 o | 598 647 960 o | 5 188 o |
| `diffusion-vqarad` | 1 222 o | 598 647 960 o | 5 188 o |

Total si tout était téléchargé : **≈ 2,68 Go** (non téléchargé, cf. §4).

### 1.2 `adapter_config.json` (identique pour les 6 sous-dossiers, sauf `base_model_name_or_path`)

| Champ | Valeur (mesurée) |
|---|---|
| `peft_type` | `LORA` |
| `r` | **64** |
| `lora_alpha` | **128** (⇒ facteur d'échelle `alpha/r` = **2.0**) |
| `use_rslora` | `false` |
| `lora_dropout` | `0.0` |
| `bias` | `none` |
| `init_lora_weights` | `true` |
| `inference_mode` | `true` |
| `peft_version` | `0.19.1` |
| `target_modules` | `(?!.*vision_tower).*\.(q_proj\|k_proj\|v_proj\|o_proj\|gate_proj\|up_proj\|down_proj)` |
| `base_model_name_or_path` (ar-*) | `google/gemma-4-26B-A4B-it` |
| `base_model_name_or_path` (diffusion-*) | `google/diffusiongemma-26B-A4B-it` |
| `auto_mapping.base_model_class` (ar-*) | `Gemma4ForConditionalGeneration` (`transformers.models.gemma4.modeling_gemma4`) |
| `auto_mapping.base_model_class` (diffusion-*) | `DiffusionGemmaForBlockDiffusion` (`transformers.models.diffusion_gemma.modeling_diffusion_gemma`) |

**Vérification de l'affirmation « rank 64 / α 128 »** : confirmée par lecture directe des six
`adapter_config.json` (valeurs ci-dessus). **Attention** : `lora_alpha/r = 2.0`, le facteur
appliqué n'est donc pas 1 — c'est le facteur `scale` de PEFT.

### 1.3 Modules réellement adaptés (en-têtes safetensors lus par requêtes HTTP Range)

`lora_A` : `[64, in_features]`, `lora_B` : `[out_features, 64]` ⇒ `ΔW = (128/64)·B·A = 2·B·A`,
exactement la convention PEFT (`ΔW = scale · B @ A`).

**Adaptateurs AR (`ar-*`)** — 410 tenseurs = **205 modules** × {A, B} :

```
base_model.model.model.language_model.layers.{0..29}.self_attn.{q_proj,k_proj,o_proj}   (30 de chaque)
base_model.model.model.language_model.layers.{0..29}.self_attn.v_proj                   (25 de 30)
base_model.model.model.language_model.layers.{0..29}.mlp.{gate_proj,up_proj,down_proj}   (30 de chaque)
```

**Adaptateurs diffusion (`diffusion-*`)** — 826 tenseurs = **413 modules** × {A, B} :

```
base_model.model.model.decoder.layers.{0..29}.{self_attn.*, mlp.*}        (205 modules, identiques aux AR)
base_model.model.model.encoder.language_model.layers.{0..29}.{...}         (205 modules, MÊMES POIDS DE BASE)
base_model.model.model.decoder.self_conditioning.{gate_proj,up_proj,down_proj}  (3 modules)
```

### 1.4 Ce qui est **gelé** (vérifié, pas recopié)

Aucune clé `expert*`, `router*`, `vision*`, `embed*`, `norm*`, `lm_head*` dans les en-têtes
des 6 adaptateurs ⇒ **experts, routeur, embeddings, normes, tête et tour vision sont gelés**.
Le mécanisme (à ne pas confondre avec la seule expression régulière) :

* la regex `target_modules` ne retient que des noms finissant par `q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj` ;
* les experts de Gemma 4 / DiffusionGemma sont des **paramètres empilés 3-D**
  (`model.language_model.layers.N.experts.gate_up_proj`, `…experts.down_proj` — vérifié sur
  l'index de `checkpoints/gemma4_it`), donc ce ne sont pas des `nn.Linear` : PEFT ne peut pas
  les adapter ;
* le routeur (`router.proj.weight`) ne correspond pas aux suffixes listés ; la tour vision est
  exclue par la négation `vision_tower`.

### 1.5 Convention de nommage (question de la mission)

| Côté | Convention |
|---|---|
| AR (Gemma 4) | `model.language_model.layers.N.…` — c'est **la convention Gemma 4** (identique à celle des checkpoints `google/gemma-4-26B-A4B[-it]` du dépôt) |
| Diffusion (DiffusionGemma) | `model.decoder.layers.N.…` **et** `model.encoder.language_model.layers.N.…` — convention **DiffusionGemma** |

→ Le mapping `Gemma4 → DiffusionGemma` déjà présent dans `academiia/core/common.py`
(`model.language_model.layers.` → `model.decoder.layers.`) est **exactement** celui qu'il faut
pour ces adaptateurs, et les formes des modules AR et diffusion sont **identiques**
(q_proj `[64, 2816]→[4096, 64]`, gate/up `[64,2816]→[2112,64]`, down `[64,2112]→[2816,64]`) :
le `ΔW` se transpose sans rechargement.

### 1.6 Résultatats publiés par la carte du dépôt

| sous-dossier | backbone | dataset | LLM-judge acc |
|---|---|---|---|
| `diffusion-vqarad` | discrete-diffusion | VQA-RAD | 0.649 |
| `ar-vqarad` | autoregressive | VQA-RAD | 0.649 |
| `diffusion-slake` | discrete-diffusion | SLAKE | 0.863 |
| `ar-slake` | autoregressive | SLAKE | 0.817 |
| `diffusion-vqamed` | discrete-diffusion | VQA-Med | 0.666 |
| `ar-vqamed` | autoregressive | VQA-Med | 0.631 |

---

## 2. GitHub — `mxvp/discrete_diffusion_RRG`

* Public, branche `main`, dernier push **2026-07-01**, sans licence, 0 étoile.
* Description : *Image-conditioned discrete-diffusion CXR report drafting* — « *Anonymized for
  double-blind review* ».
* Contenu : `README.md`, `config.py`, `data.py`, `utils.py`, `configs/`, `models/`, `scripts/`.
* Le README indique : `pip install torch transformers>=5.11 peft accelerate numpy pillow pyyaml` ;
  « *Requires a GPU (bf16, ~80 GB for the diffusion backbone)* ».
* Méthode décrite : « *LoRA on the text/decoder linears* » ; **AR = baseline contrôlée**
  (« the controlled baseline ») ; infill any-order via un *sampler hook*.
* Scripts : `scripts/train.py` (LoRA), `scripts/infer.py` (`--mode vqa`, `--mode infill_ablation`),
  `scripts/eval.py` (accuracy fermée/ouverte + test de significativité apparié),
  `scripts/llm_judge_vqa.py` et `scripts/llm_judge_infill.py` (juges **API** : anthropic /
  openai / google-genai), `scripts/api_vqa_baseline.py`, `scripts/bench_speed.py`.
* Les identifiants HF des backbones sont dans les configs, pas dans le README.

**Différence avec le protocole A2D** : l'article A2D (App. A.1, PDF p. 18) dit avoir utilisé
**toutes** les questions de test (1 061 SLAKE / 451 VQA-RAD / 500 VQA-Med) avec un **juge
ouvert** (Qwen3.8-27B, greedy, thinking désactivé), là où Van Puyvelde et al. ont évalué
350 questions tirées au hasard avec Claude Sonnet-4.6 : « *so their results serve only as
reference* ».

---

## 3. Conséquences pour la bibliothèque A2D (`modules/a2d/a2d/`)

1. **Source LoRA** : `academiia.a2d.sources.LoraAdapterSource` lit exactement ce format
   (`base_model.model.` + suffixe `.lora_A.weight` / `.lora_B.weight`, `r`/`lora_alpha` de
   `adapter_config.json`, facteur `alpha/r`). L'équivalence avec `peft.merge_and_unload` est
   vérifiée par test (`modules/a2d/tests/test_a2d_library.py::test_lora_source_matches_peft_merge_and_unload`).
2. **Mapping** : `academiia.a2d.naming.GEMMA4_TO_DIFFUSIONGEMMA` (qui réutilise `academiia/core/common.py`)
   s'applique tel quel aux adaptateurs `ar-*` → projections du decoder DiffusionGemma.
3. **Piège « double comptage » (à traiter à la porte G2)** : les adaptateurs `diffusion-*`
   contiennent **deux jeux d'updates sur des poids partagés** (`model.decoder.layers.*` et
   `model.encoder.language_model.layers.*`) plus les projections `self_conditioning`.
   L'article (App. A.2) tranche : « *The diffusion fine-tuning adapters, which hold separate
   encoder and decoder updates, **stay unmerged and are scaled according to the experimental
   formula*** ». Une fusion de `τ_D` dans les poids, qui additionnerait les deux jeux, ne
   serait **pas** conforme au papier. La bibliothèque expose donc `τ_D` comme une source
   distincte : c'est à l'appelant (protocole G2) de décider entre fusion et mise à l'échelle
   des adapters.
4. **Le vecteur AR médical** (`ar-*`) est, lui, un delta de fine-tuning de domaine
   (IT → médical) : c'est le `τ_A` du setup DiffusionGemma de l'article (Tableaux 6 et 19),
   et non le vecteur « base → IT général » que le dépôt utilisait jusqu'ici.

---

## 4. Ce qui n'a **pas** été téléchargé, et pourquoi

Les 6 fichiers `adapter_model.safetensors` (284–571 MiB chacun, ≈ 2,68 Go au total) **n'ont
pas** été téléchargés. La mission A3 demandait de ne télécharger les adaptateurs que si leur
taille était raisonnable, et de s'arrêter sinon ; ici la taille est *raisonnable* mais
l'intérêt est nul à ce stade :

* la porte **G2** (reproduction DiffusionGemma médical) est **hors périmètre** de la mission
  J6 (trajectoire par portes : G1 mécanique → G2 médical → G3 raisonnement) ;
* l'information demandée par A3 (rank, alpha, target_modules, taille, **noms de modules et
  convention de nommage**) est **complète** dans les en-têtes safetensors, lus par requêtes
  HTTP Range — sans écrire 2,68 Go sur disque.

Commandes pour rejouer l'inspection (lecture seule) :

```bash
# arborescence et tailles
python3 -c "import json,urllib.request as u; d=json.load(u.urlopen('https://huggingface.co/api/models/gevaertlab/diffusiongemma-radiology-vqa/tree/main?recursive=true&expand=true')); [print(e['path'], e.get('size')) for e in d]"

# adapter_config.json
python3 -c "import json,urllib.request as u; print(json.dumps(json.load(u.urlopen('https://huggingface.co/gevaertlab/diffusiongemma-radiology-vqa/resolve/main/diffusion-slake/adapter_config.json')), indent=1))"

# en-tête safetensors (noms de modules) sans télécharger les poids
python3 -c "
import json,struct,urllib.request as u
url='https://huggingface.co/gevaertlab/diffusiongemma-radiology-vqa/resolve/main/ar-slake/adapter_model.safetensors'
n=struct.unpack('<Q', u.urlopen(u.Request(url,headers={'Range':'bytes=0-7'})).read(8))[0]
h=json.loads(u.urlopen(u.Request(url,headers={'Range':f'bytes=8-{8+n-1}'})).read())
print(len(h)-1, 'tenseurs'); print('\n'.join(sorted(h)[:5]))
"
```
