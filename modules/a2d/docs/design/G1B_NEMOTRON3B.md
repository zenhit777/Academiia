# G1b — Faisabilité : Nemotron-Labs-Diffusion-3B ↔ Ministral-3-3B-2512

**Date** : 2026-10-20 · **Mission** : J6bis E1–E4 · **Nature** : **lecture seule**
(aucun poids téléchargé, aucun fichier de poids écrit) · **Statut de la porte
G1b** : **non démarrée**

> **Conclusion en une phrase : GO pour la partie mécanique (géométrie de la
> Table 21, ~23 Go de téléchargement, ~10 min de calcul), NO-GO pour la
> partie évaluation complète (protocole à 8 192 tokens, ≈ 30 GPU-h par
> modèle et par point de grille, hors budget de la machine). Détails et
> chiffres : §6.

---

## 1. E1 — Identifiants HF, licences, tailles

Source primaire : `https://huggingface.co/api/models/<id>` et
`…/tree/main?recursive=true` (relevés du 2026-10-20).

| # | Identifiant HF exact | Statut | `gated` | Licence | Poids publiés | `safetensors.total` (paramètres) |
|---|---|---|---|---|---:|---:|
| 1 | `mistralai/Ministral-3-3B-Base-2512` | existe | non | `apache-2.0` | 7 698 241 056 o (2 shards + doublon `consolidated.safetensors`) | 3 849 090 048 |
| 2 | `mistralai/Ministral-3-3B-Instruct-2512` | existe | non | `apache-2.0` | 4 672 099 184 o (fichier unique, FP8) | 3 849 090 048 |
| 3 | `nvidia/Nemotron-Labs-Diffusion-3B-Base` | existe | non | `nvidia-nemotron-open-model-license` | 7 663 347 000 o (fichier unique) | 3 831 659 520 |
| 4 | `nvidia/Nemotron-Labs-Diffusion-3B` | existe — **c'est le checkpoint « Instruct »** | non | `nvidia-nemotron-open-model-license` | 7 663 347 000 o + `linear_spec_lora/` 95 427 600 o | 3 831 659 520 |
| 5 | `nvidia/Nemotron-Labs-Diffusion-3B-Instruct` | **N'EXISTE PAS** | — | — | — | — |

### 1.1 L'identifiant « Instruct » inexistant

`https://huggingface.co/api/models/nvidia/Nemotron-Labs-Diffusion-3B-Instruct`
→ **HTTP 401**, strictement identique à la réponse du dépôt témoin
`nvidia/This-Repo-Does-Exist-xyz123` (HF renvoie le même code pour un dépôt
inexistant et pour un dépôt privé). La recherche
`…/api/models?search=Nemotron-Labs-Diffusion-3B` ne retourne **aucun** dépôt
`nvidia/…-Instruct` : la famille publie `-3B`, `-3B-Base`, `-8B`, `-8B-Base`,
`-14B`, `-14B-Base`, `-VLM-8B`.

⇒ **Règle à figer** : le checkpoint post-entraîné (« Instruct ») est
`nvidia/Nemotron-Labs-Diffusion-3B` **sans suffixe**. Ne jamais écrire
`Nemotron-Labs-Diffusion-3B-Instruct` dans un script.

Sa qualification comme « Instruct » est **[INFERENCE]**, étayée par trois
sources concordantes : `generation_config.json` (`eos_token_id: 11` =
`<|im_end|>` contre `2` = `</s>` pour le Base), `config.json`
(`max_position_embeddings: 262144` + RoPE yarn `factor: 16.0` contre `4096` +
`factor: 0.25`), et le gabarit ChatML de `chat_template.jinja`.

### 1.2 Points d'attention

* **Aucune `model.safetensors.index.json`** n'est publiée pour les deux dépôts
  diffusion (fichier `model.safetensors` unique) : la bibliothèque A2D sait
  gérer ce cas (`load_weight_map` tombe sur le mono-fichier) — vérifié.
* `Nemotron-Labs-Diffusion-3B-Base` **documente dans son README** un
  `PeftModel.from_pretrained(..., subfolder="linear_spec_lora")` alors que le
  dossier `linear_spec_lora/` **n'est pas publié** sur ce dépôt (il l'est sur
  `-3B`). Le snippet échouerait.
* Révisions à figer pour une reproduction exacte : Base AR `6f9c4b12…`, Instruct
  AR `b35d4dfe…`, Base diffusion `4bd78816…`, diffusion Instruct `0d51902d…`.
* Le `linear_spec_lora/` de `-3B` est une LoRA `r=128, α=512`,
  `target_modules=["o_proj"]` (auto-spéculation linéaire) — **hors périmètre
  A2D**.

## 2. E2 — Noms de paramètres et brouillon de mapping (App. A.2)

### 2.1 Inventaire du modèle de diffusion (237 tenseurs, 3 831 659 520 paramètres)

`config.json` : `model_type: nemotron_labs_diffusion`, `hidden_size: 3072`,
`intermediate_size: 9216`, `num_hidden_layers: 26`, `num_attention_heads: 32`,
`num_key_value_heads: 8`, `head_dim: 128`, `vocab_size: 131072`,
`tie_word_embeddings: false`, `torch_dtype: bfloat16`, `mask_token_id: 100`,
`dlm_paradigm: "bidirectional"`, `block_size: 32`, `use_cache: false`.

| Motif de tenseur | Occ. | Forme | Paramètres |
|---|---:|---|---:|
| `encoder.embed_tokens.weight` | 1 | `[131072, 3072]` | 402 653 184 |
| `encoder.layers.{0..25}.self_attn.{q,k,v,o}_proj.weight` | 104 | `[4096,3072]`, `[1024,3072]`, `[3072,4096]` | 817 889 280 |
| `encoder.layers.{0..25}.mlp.{gate,up,down}_proj.weight` | 78 | `[9216,3072]`, `[3072,9216]` | 2 208 301 056 |
| `encoder.layers.{0..25}.{input,post_attention}_layernorm.weight` | 52 | `[3072]` | 159 744 |
| `encoder.norm.weight` | 1 | `[3072]` | 3 072 |
| `diffusion_head.weight` | 1 | `[131072, 3072]` | 402 653 184 |
| **Total** | **237** | | **3 831 659 520** ✔ = `safetensors.total` |

**Familles** (pour la bibliothèque) : `embed` 1 · `head` 1 · `norms` 53 ·
`attn` 104 · `mlp` 78 · `router` **0** · `experts` **0** · `other` 0.

Points structurants :

* **Il n'y a pas de décodeur séparé** (contrairement à DiffusionGemma) : un
  seul encodeur bidirectionnel sert aux trois modes (AR, diffusion,
  auto-spéculation) ; le basculement se fait par un flag par couche.
* **La tête de sortie est un tenseur unique et distinct**,
  `diffusion_head.weight`, **non lié** à l'embed (`tie_word_embeddings: false`)
  ⇒ « *All measurements exclude output heads* » (App. B.7) s'applique par
  exclusion d'**un seul** tenseur (402 653 184 paramètres).
* **Pas de self-conditioning** dans le modèle 3B publié (contrairement à DG) :
  `other` = 0.

### 2.2 Inventaire du côté AR (458 tenseurs)

`mistralai/Ministral-3-3B-Base-2512/model.safetensors.index.json` :

| Famille | Noms | Effectif |
|---|---|---:|
| embed | `language_model.model.embed_tokens.weight` | 1 |
| norms | `language_model.model.layers.{0..25}.{input,post_attention}_layernorm.weight` + `language_model.model.norm.weight` | 53 |
| attn | `language_model.model.layers.{0..25}.self_attn.{q,k,v,o}_proj.weight` | 104 |
| mlp | `language_model.model.layers.{0..25}.mlp.{gate,up,down}_proj.weight` | 78 |
| **vision** | `vision_tower.*` | **218** |
| **projecteur** | `multi_modal_projector.*` | **4** |
| **Total** | | **458** |

Pas de `lm_head.weight` : la tête AR est **liée** à l'embed
(`text_config.tie_word_embeddings: true`).

### 2.3 Isométrie vérifiée

| | AR (Ministral-3-3B) | Diffusion (Nemotron-LD-3B) |
|---|---:|---:|
| embed | 402 653 184 | 402 653 184 |
| 26 couches (attn + mlp + norms) | 3 026 347 008 | 3 026 347 008 |
| norm final | 3 072 | 3 072 |
| **« squelette »** | **3 429 006 336** | **3 429 006 336** |
| tête de sortie | 0 (liée) | 402 653 184 (`diffusion_head`) |
| vision + projecteur | 420 083 712 | 0 |
| **Total** | 3 849 090 048 ✔ | 3 831 659 520 ✔ |

⇒ Les deux backbones sont **isométriques** (26 couches, mêmes formes) : le plan
de fusion est **bien défini et non ambigu**.

### 2.4 Brouillon de mapping (règles de l'App. A.2)

Règle du papier (App. A.2, « Nemotron / Ministral ») : « *Nemotron needs an
explicit mapping from Mistral's language-model parameters to the diffusion
backbone. The vocabulary sizes agree, but 14 token IDs denote different tokens
in the two families, and IDs 34–35 additionally differ between Ministral Base
and Instruct. […] For protected Instruct merges, we keep these 16
input-embedding rows at their values in the Nemotron Instruct recipient,
excluding them from both the AR-vector addition and the diffusion-vector
subtraction. For corrected Base-transfer experiments, we additionally freeze
the diffusion mask-token embedding (ID 100), keeping all 17 protected rows at
their Nemotron Base values. The diffusion output head is likewise kept at its
value in the recipient checkpoint.* »

| # | Tenseur AR | Tenseur diffusion | Règle |
|---|---|---|---|
| M1 | `language_model.model.embed_tokens.weight` | `encoder.embed_tokens.weight` | mappée **ligne à ligne**, **sauf lignes protégées** (§2.5) |
| M2–M5 | `…layers.{N}.self_attn.{q,k,v,o}_proj.weight` | `encoder.layers.{N}.self_attn.{q,k,v,o}_proj.weight` | fusionnée (τ_A appliqué, τ_D soustrait) |
| M6–M8 | `…layers.{N}.mlp.{gate,up,down}_proj.weight` | `encoder.layers.{N}.mlp.{gate,up,down}_proj.weight` | fusionnée |
| M9–M10 | `…layers.{N}.{input,post_attention}_layernorm.weight` | `encoder.layers.{N}.{input,post_attention}_layernorm.weight` | fusionnée |
| M11 | `language_model.model.norm.weight` | `encoder.norm.weight` | fusionnée |
| M12 | *(aucun — tête liée)* | `diffusion_head.weight` | **tête conservée** (valeur du receveur) : exclue de `+τ_A` et de `−τ_D` |
| M13 | `vision_tower.*` (218) | — | **sans équivalent** ⇒ écarté de la géométrie |
| M14 | `multi_modal_projector.*` (4) | — | **sans équivalent** ⇒ écarté |

Bilan : **236 paires** (M1–M11), **1 tenseur diffusion sans donneur** (M12),
**222 tenseurs AR sans receveur** (M13–M14). 236 + 1 = 237 ✔ (côté diffusion) ;
236 + 222 = 458 ✔ (côté AR).

**Implémentation à prévoir** : un nouveau mapping `MINISTRAL_TO_NEMOTRON_LD`
dans `modules/a2d/a2d/naming.py`, avec :
* `rename` : `language_model.model.` → `encoder.` (+ `layers.N.` conservé) ;
* `embedding_policy` : **lignes** (intersection) **avec masque de lignes
  protégées** — extension nécessaire de `MergePolicy` (la bibliothèque ne
  connaît aujourd'hui que « rows » / « strict ») ;
* famille `head` **non fusible** (la tête de sortie doit être conservée) ;
* familles `router` / `experts` : sans objet (0 tenseur).

### 2.5 Lignes d'embedding protégées — 16 / 17, **vérifié sur les tokenizers**

| ID | Ministral Base | Ministral Instruct | Nemotron (Base et Instruct) |
|---|---|---|---|
| 10 | `[IMG]` | `[IMG]` | `<\|im_start\|>` |
| 11 | `<pad>` | `<pad>` | `<\|im_end\|>` |
| 12 | `[IMG_BREAK]` | `[IMG_BREAK]` | `<think>` |
| 13 | `[IMG_END]` | `[IMG_END]` | `</think>` |
| 14 | `[PREFIX]` | `[PREFIX]` | `<tool_call>` |
| 15 | `[MIDDLE]` | `[MIDDLE]` | `</tool_call>` |
| 16 | `[SUFFIX]` | `[SUFFIX]` | `<tool_response>` |
| 17 | `[SYSTEM_PROMPT]` | `[SYSTEM_PROMPT]` | `</tool_response>` |
| 18 | `[/SYSTEM_PROMPT]` | `[/SYSTEM_PROMPT]` | `<SPECIAL_18>` |
| 19 | `[TOOL_CONTENT]` | `[TOOL_CONTENT]` | `<SPECIAL_19>` |
| 24 | `[AUDIO]` | `[AUDIO]` | `<SPECIAL_24>` |
| 25 | `[BEGIN_AUDIO]` | `[BEGIN_AUDIO]` | `<SPECIAL_25>` |
| 32 | `[ARGS]` | `[ARGS]` | `<SPECIAL_32>` |
| 33 | `[CALL_ID]` | `[CALL_ID]` | `<SPECIAL_33>` |
| 34 | `[THINK]` | `<SPECIAL_34>` | `<SPECIAL_34>` |
| 35 | `[/THINK]` | `<SPECIAL_35>` | `<SPECIAL_35>` |
| 100 | `<SPECIAL_100>` | `<SPECIAL_100>` | `<SPECIAL_100>` (**token de masque** côté diffusion) |

⇒ **14 ids divergents** (10–19, 24, 25, 32, 33) **+ 2 ids qui diffèrent aussi
entre Base et Instruct** (34, 35) = **16 lignes protégées** en merge Instruct ;
**17** en Base-transfer (ajout de la ligne 100, le token de masque). **Le compte
du papier est confirmé à l'identique.**

Les deux familles n'utilisent pas le même moteur de tokenisation
(`MistralCommonBackend` + `mistral-common` côté Mistral ;
`PreTrainedTokenizerFast` côté Nemotron) ⇒ le mappage doit se faire sur les
**IDs de token**, jamais sur les chaînes.

### 2.6 Différence de configuration positionnelle (point d'attention)

| champ | AR Base | AR Instruct | Diffusion Base | Diffusion Instruct |
|---|---:|---:|---:|---:|
| `max_position_embeddings` | 262 144 | 262 144 | **4 096** | 262 144 |
| rope `factor` | 16,0 | 16,0 | **0,25** | 16,0 |
| `tie_word_embeddings` | true | true | false | false |
| `use_cache` | true | true | false | false |
| `eos_token_id` | 2 | 2 | 2 | 11 |

Le RoPE n'a **aucun paramètre appris** (`inv_freq` est un buffer non
persistant) ⇒ rien à fusionner pour le positionnel. Mais le **réglage
positionnel natif diffère** entre le Base AR et le Base diffusion : cohérent
avec le protocole du papier (« *We evaluate it on the raw problem text at its
native positional setting* », App. A.1).

## 3. E3 — Pile d'évaluation (NeMo-Skills) et décodage AR

### 3.1 NeMo-Skills

| Élément | Valeur | Source |
|---|---|---|
| Licence | Apache-2.0 | `api.github.com/repos/NVIDIA-NeMo/Skills` |
| Installation | `pip install git+https://github.com/NVIDIA-NeMo/Skills.git` (ou `nemo-skills-core` pour un chemin sans orchestration cluster) | `docs/basics/installation.md` |
| **Docker requis ?** | **NON** pour le chemin local (`--cluster=local`) ; **OUI** pour le chemin SLURM (`eval.sh` exige « *SLURM + enroot/pyxis container required* ») | idem + `NVlabs/Nemotron-Labs-Diffusion` |
| Taille du dépôt | ≈ 390 MiB (historique git) | API GitHub |
| Image SGLang documentée | `lmsysorg/sglang:v0.5.10.post1` = **17,1 Go** (amd64) | Docker Hub |
| Commande | `ns prepare_data <benchmarks>` puis `ns eval --cluster=local --server_type=openai --model=… --server_address=… --benchmarks=…` | `nvidia-nemo.github.io/Skills/evaluation/` |
| Benchmarks couverts | math (aime24/25, hmmt), code (livecodebench, swe-bench, bird), sciences (**gpqa**), instruction (**ifbench, ifeval**), long-contexte, tool-calling, multilingue, speech, VLM, SD | idem |

> **Attention** : l'URL `docs.nvidia.com/nemo/skills` demandée par la mission
> renvoie **404**. Les docs canoniques sont
> `https://nvidia-nemo.github.io/Skills/`.

### 3.2 Le `ns eval` stock ne connaît pas la diffusion

Le client NeMo-Skills ne dialogue qu'avec un endpoint OpenAI
chat-completions standard ; les modes de décodage Nemotron-LD exigent des
knobs hors schéma (`steps`, `block_length`, `threshold`, `ar_weight`,
`linear_speculation`, `max_thinking_tokens`).

**Voie officielle** (fournie par NVIDIA/NVlabs) :
`https://github.com/NVlabs/Nemotron-Labs-Diffusion` →
`xp/nemo-skills/eval_dlm.py` (client patché : `--generation-algorithm`,
`--steps`, `--block-length`, `--threshold`, `--ar-weight`, …) +
`xp/dlm_api/dlm_batch_server.py` (serveur OpenAI-compatible, multi-GPU +
load-balancer).

**Trois chemins possibles** :

| Chemin | Docker | SLURM | Benchmarks | Commentaire |
|---|---|---|---|---|
| **A.** `dlm_batch_server.py` + `eval_dlm.py`, `--cluster=local` | non | non | les 10 du papier | **recommandé** (chemin workstation) |
| **B.** `evaluate.py` (évaluateur autonome du dépôt NVlabs) | non | non | gsm8k + math-500 seulement | dépannage ; pas de NeMo-Skills |
| **C.** `eval.sh` | oui (`.sqsh` NeMo-Skills, enroot/pyxis) | **oui** | les 10 du papier | impossible ici (pas de SLURM) |

### 3.3 Décodage AR avec cache : **faisable, natif**

Dans `modeling_nemotron_labs_diffusion.py` :

* `ar_generate()` : « *Bypasses NemotronLabsDiffusionModel.forward() … Calls
  self.encoder (Ministral3Model) with explicit cache_position, position_ids, and
  use_cache so the KV cache and causal masking behave identically to
  MistralForCausalLM / vLLM* » → `DynamicCache`, `cache_position` explicite,
  `use_cache=True` ;
* `linear_spec_generate()` : auto-spéculation linéaire, un **seul** `DynamicCache`
  partagé entre la phase de brouillon et la phase de vérification, avec
  `_crop_dynamic_cache` pour revenir en arrière sur les jetons rejetés ;
* `config.json` déclare `use_cache: false` (défaut de config) mais les méthodes
  de génération forcent `use_cache=True` ⇒ **ce n'est pas un obstacle** ;
* `generate()` (diffusion) exige `max_new_tokens % block_length == 0`,
  `block_length=32` par défaut, `threshold=0.9` (dévoilement par seuil).

Les checkpoints 3B publiés sont en `dlm_paradigm: "bidirectional"` ⇒ le chemin
`sdpa` standard est utilisé (pas de `flex_attention`, pas de contrainte de
compilateur stricte).

### 3.4 Protocole d'évaluation du papier (App. A.1)

> « *We use NeMo-Skills for all evaluations. For the Ministral donor
> checkpoints, we sample one response per problem with **temperature 0.7,
> top-p 0.95**, and a maximum of **8 192 output tokens**. The donor evaluations
> use native Mistral chat formatting and benchmark-provided system
> instructions.* »

Points opérationnels :

* **Donneur AR** : servi par **vLLM** (recommandation de la carte Mistral) —
  `vllm serve mistralai/Ministral-3-3B-Base-2512 --tokenizer_mode mistral
  --config_format mistral --load_format mistral`, `vllm>=1.12.0` +
  `mistral-common>=1.8.6` ;
* **Récepteur diffusion** : `transformers>=5.0.0` + `trust_remote_code=True`
  (le code distant importe `transformers.modeling_layers`,
  `transformers.masking_utils`, `torch.nn.attention.flex_attention` ⇒
  **sensible à la version de `transformers`** ; `config.json` déclare
  `transformers_version: 5.0.0`) ;
* **Base diffusion** : protocole « raw » (texte brut, EOS = id 2) ; **Instruct
  diffusion** : gabarit ChatML Nemotron, 8 192 tokens max ;
* **Transfert Base** : prompts au format du donneur `[INST]…[/INST]`, pas de
  prompt système, pas de marqueur de thinking.

## 4. Coût par échantillon : AR vs diffusion

> **Tout ce qui suit est une projection**, dérivée de (i) nos **mesures** sur
> Dream-7B (cf. `modules/a2d/docs/design/EVAL_BUDGET.md`), (ii) les **comptes de
> paramètres** lus sur les `config.json`, (iii) le **TPF = 5,89** rapporté par
> l'outil `evaluate.py` de NVlabs (mesuré sur **8B**, pas sur 3B). Aucun de ces
> chiffres n'est une mesure sur le 3B.

### 4.1 Bande passante mémoire (le régime limitant)

| | AR (Ministral-3-3B) | Diffusion (Nemotron-LD-3B) |
|---|---:|---:|
| Poids lus par forward | 7,70 Go | 7,66 Go |
| Tokens produits par forward | 1 | **5,89** [INFERENCE pour 3B ; mesuré sur 8B] |
| **Trafic mémoire par token généré** | **7,70 Go** | **1,30 Go** |
| Gain | — | **≈ 5,9×** |

### 4.2 Coût de calcul par token

* AR : `2 × 3,429e9 ≈ 6,9 GFLOP/token` (hors tête) ; ≈ 7,7 GFLOP/token avec
  la tête non liée.
* Diffusion : chaque forward couvre 32 positions (bloc) ⇒ `32 × 0,17 forward ×
  7,7 GFLOP ≈ 41 GFLOP/token`.
* Sur un GPU à ~1,8 To/s HBM et ~700 TFLOP/s BF16 : 1,30 Go ⇒ **0,72 ms** de
  mémoire contre 0,06 ms de calcul ⇒ le chemin diffusion **reste memory-bound** :
  le gain de 5,9× n'est pas amorti par le calcul.
* **[INFERENCE]** Si le batch monte (régime compute-bound), l'avantage se réduit.

### 4.3 Cache KV

`2 × 8 têtes KV × 128 head_dim × 2 o × 26 couches` = **106 496 o/token
≈ 104 KiB/token** ⇒ 8 192 tokens = **0,87 Go** ; 262 144 tokens = 27,9 Go.
En diffusion, le cache est un **cache de préfixe** (le bloc courant n'est pas
mis en cache) : la mémoire KV est dominée par le préfixe commun.

### 4.4 Coût par échantillon de benchmark, protocole papier (8 192 tokens max)

| Scénario | AR | Diffusion |
|---|---:|---:|
| GSM8K, longueur **médiane** constatée (~300-500 tokens) | 1,1 – 1,9 GPU-h pour 1 319 problèmes | 0,2 – 0,3 GPU-h |
| Pire cas (8 192 tokens saturés, 1 319 problèmes) | **30 GPU-h** | **5,1 GPU-h** |
| Sweep de grille (λ sur 1IFEval : 541 prompts × 6-8 points) | **12 – 35 GPU-h** | **2 – 6 GPU-h** |

**Corroboration par un fait primaire du papier** : le budget de 8 192 tokens sur
541 prompts IFEval a été jugé « *too expensive* » et **remplacé par 1 280
tokens** (Table 8, sweep de transfert Nemotron-LD) — c'est le papier lui-même
qui constate le problème.

### 4.5 Report sur notre machine

En transposant la mesure Dream (326 s/prompt à 1 280 tokens de débruitage) au
3B avec le gain de bande passante 5,9× et le budget de 8 192 tokens (× 6,4) :

| Mode | Tokens | s/prompt [INFERENCE] | 541 prompts (IFEval) |
|---|---:|---:|---:|
| AR (Ministral) | 8 192 | ≈ 2 100 s | **≈ 315 h** |
| Diffusion (Nemotron-LD) | 8 192 | ≈ 354 s | **≈ 53 h** |

Ces ordres de grandeur sont **cohérents** avec le tableau 4.4 (le Dream-7B est
7 B, pas 3 B, et son budget de 1 280 tokens était saturé) — ils sont donc
**indicatifs**, pas prédictifs.

## 5. Cibles chiffrées de l'article (Tableaux 1c, 17, 21)

| Cible | Valeur | Source | Vérifiable sur les configs lues ? |
|---|---:|---|---|
| ‖τ_A‖ | **43,1** | Table 21 | **NON** — propriété de tenseurs |
| ‖τ_D‖ | **17,1** | Table 21 | **NON** |
| ‖Δ_b‖ | **280,9** | Table 21 | **NON** |
| ‖Δ_p‖ | **271,4** | Table 21 | **NON** |
| cos(Δ_b, Δ_p) | **+0,987** | Table 21 | **NON** |
| cos(τ_A, τ_D) | **+0,075** | Table 21 | **NON** |
| Table 1c (scores) | AR donor 56,7 · Base DLM 38,0 · transfer DLM 44,9 · Instruct DLM 51,0 · merge DLM 54,0 · Instruct AR 52,8 · merge AR 54,2 | Table 1c | **NON** — nécessite l'exécution |

**Ce qui est vérifiable, en revanche, et qui conditionne la validité des six
chiffres** :

1. ✅ le **plan de tenseurs est compatible** : les 236 paires ont des formes
   identiques ⇒ la concaténation des différences (protocole App. B.7 : « *we
   concatenate parameter differences over backbone tensors (attention, MLP, and
   normalization layers)* ») est bien définie ;
2. ✅ les **lignes protégées sont exactement 16 / 17**, vérifiées sur les
   tokenizers (§2.5) ;
3. ✅ la **tête de sortie est identifiable et séparable** (`diffusion_head`,
   unique, non liée) ;
4. ✅ les **222 tenseurs multimodaux AR** sont identifiables comme « unmatched » ;
5. ✅ la **ligne 100** est réservée dans les deux familles et sert de token de
   masque côté diffusion ;
6. ✅ la **différence positionnelle du Base** est documentée et sans effet sur
   la géométrie (RoPE = buffer non persistant).

⇒ **Les six chiffres de la Table 21 sont structurellement reproductibles, mais
non vérifiables a priori** : ils exigent les poids. La porte G1b doit être
franchie par **exécution**, pas par inspection de configuration.

## 6. Conclusion : **GO / NO-GO**

### 6.1 Partie mécanique (géométrie Table 21 + fusion) — **GO**

| Poste | Coût |
|---|---|
| Téléchargement | **≈ 23 Go** (7,70 + 4,67 + 7,66 + 7,66) |
| Temps de calcul | **≈ 10 min** (4 checkpoints lus en FP32, le calcul Dream-7B prend 4 min pour 4 × 15 Go) |
| RAM | ≤ 30 Go (streaming tenseur par tenseur, comme `measure_geometry.py`) |
| Code à écrire | un mapping `MINISTRAL_TO_NEMOTRON_LD` (§2.4) + le **masque des lignes protégées** (extension de `MergePolicy`) |
| Risque | **faible** : le plan de fusion est isométrique et vérifié ; les 6 chiffres de la Table 21 deviennent mesurables |

**Décision : GO.** C'est l'élément le plus rentable de la porte G1b : il
transforme 6 « non vérifiable » en 6 mesures, pour ~23 Go et ~10 min.

### 6.2 Partie évaluation (Table 1c / Table 17) — **NO-GO sur cette machine**

| Blocage | Gravité | Détail |
|---|---|---|
| **Coût du protocole papier** | bloquant | 8 192 tokens de sortie ⇒ ≈ 315 h (AR) ou ≈ 53 h (diffusion) **par benchmark** de 541 prompts, et **10 benchmarks** dans la Table 1c. Le budget de la mission est de 2 h par évaluation (`modules/a2d/docs/design/G1_PROTOCOL.md`) |
| **Pile d'évaluation** |bloquant | le `ns eval` stock **ne supporte pas la diffusion** : il faut le client patché de NVlabs (`xp/nemo-skills/eval_dlm.py`) + un serveur DLM à construire et à déboguer (charge de travail non chiffrable ici) |
| **Dépendance `transformers`** | moyen | le code distant de Nemotron-LD exige `transformers>=5.0.0` et importe `transformers.modeling_layers` / `masking_utils` ⇒ un environnement dédié (le venv du harnais Dream est en `transformers==4.46.2`, incompatible) |
| **Juge / métriques** | moyen | les 10 benchmarks de la Table 1c exigent le scoring NeMo-Skills complet (math, code, IFEval, GPQA…) ; le chemin B (`evaluate.py`) ne couvre que gsm8k + math-500 |
| **Faible nombre de paramètres** | faible | 3 B ⇒ un « squelette » de 3,43 Go : l'effet mémoire du transfert est **plus faible** que sur Dream-7B, donc les gains attendus seront **plus petits** que les gains de **transfert** observés sur Dream-7B (MATH-500 **+19,4** points ; GSM8K **+15,2** points) |

**Décision : NO-GO pour l'évaluation complète en l'état.** Condition de
réouverture :

1. réduire le budget de sortie à **1 280 tokens** (comme le papier l'a fait pour
   IFEval) ⇒ coût divisé par 6,4 ;
2. ne retenir que **2 benchmarks** (math + instruction) au lieu de 10 ;
3. construire le chemin A (serveur DLM + client patché) et vérifier qu'il
   produit une génération correcte sur 8 prompts ;
4. re-mesurer le **TPF sur 3B** (le 5,89 est mesuré sur 8B) avant tout
   engagement de planning.

À ces conditions, l'évaluation de 2 benchmarks en diffusion tient dans
≈ 2 × 541 × 354 s ≈ **106 h** : encore hors budget. ⇒ **la porte G1b, en tant
qu'évaluation, reste NO-GO sur cette machine ; seule sa partie mécanique est
GO.**

### 6.3 Ordre recommandé

1. **G1** (Dream) — `modules/a2d/docs/design/G1_PROTOCOL.md` —.transfert α=0,9 et merge
   λ=0,6, 4 évaluations, ≈ 4 h ;
2. **G1b-m** (Nemotron-3B, mécanique) — GO : 23 Go + 10 min, 6 mesures de la
   Table 21 ;
3. **G2** (DiffusionGemma) — bloquée par le mode « adaptateur à l'échelle »
   (`modules/a2d/docs/design/G2_adapter_scaling.md`) ;
4. **G1b-e** (Nemotron-3B, évaluation) — NO-GO tant que (1)-(4) du §6.2 ne
   sont pas levés.

## 7. Ce qui n'est **pas** vérifié

| # | Point | Statut |
|---|---|---|
| 1 | Les 6 valeurs de la Table 21 | `[NON VÉRIFIÉ]` — nécessitent les poids |
| 2 | Les scores des Tables 1c et 17 | `[NON VÉRIFIÉ]` — nécessitent l'exécution |
| 3 | Les formes exactes des tenseurs de diffusion | `[INFERENCE]` — dérivées de `config.json` (jamais lues dans un en-tête safetensors) ; le **total** est validé contre `safetensors.total` |
| 4 | Les formes du `vision_tower` et du `multi_modal_projector` | `[NON VÉRIFIÉ]` — absentes de `config.json` |
| 5 | La qualification « Instruct » de `Nemotron-Labs-Diffusion-3B` | `[INFERENCE]` étayée (eos, position, gabarit) |
| 6 | Le TPF sur 3B | `[NON VÉRIFIÉ]` — le 5,89 est mesuré sur 8B |
| 7 | La taille de l'image Docker NeMo-Skills | `[NON VÉRIFIÉ]` — non publiée |
| 8 | Le support SGLang pour le 3B | `[INFERENCE]` — le guide NVlabs est écrit pour 8B |
| 9 | Le périmètre exact d'App. B.7 (quels tenseurs entrent dans ‖τ‖) | le papier dit « *backbone tensors (attention, MLP, and normalization layers)* » **+** « *compatible input embeddings after key mapping* » pour Nemotron : l'**inclusion des embeddings est spécifique à Nemotron** et doit être respectée |
| 10 | La symétrie exacte des deux jeux LoRA (encodeur/décodeur) de l'adaptateur DG | **[INFERENCE]** — sans rapport avec Nemotron-3B, qui n'a pas d'adaptateur publié de ce type |
