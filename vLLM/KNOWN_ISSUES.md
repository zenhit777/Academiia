# vLLM — Problèmes connus

## Incident du 2026-10-18 (WSL2) — crash de la 1re génération avec le runner V1 forcé

**Contexte :** image `vllm/vllm-openai:gemma` (vLLM 0.22.1rc1.dev357), WSL2,
compose avec `VLLM_USE_V2_MODEL_RUNNER=0` (runner V1 forcé) + `--enforce-eager`
+ `--cpu-offload-gb 30`.

**Symptômes (observés, log du 2026-10-18) :**

- Chargement OK : 11 shards, ~14,5 min jusqu'à "Application startup complete".
- `GET /v1/models` OK.
- **Première** `POST /v1/chat/completions` : EngineCore meurt avec
  `ScatterGatherKernel.cu:163 Assertion idx_dim >= 0 && idx_dim < index_size
  ... index out of bounds`, puis `torch.AcceleratorError "device-side assert
  triggered"` (l'emplacement affiché `uva.py:119` est trompeur : erreur CUDA
  asynchrone).
- Dump du scheduler : `scheduled_spec_decode_tokens` = 256 valeurs `-1`
  (canevas de diffusion).

**Hypothèses (NON PROUVÉES) :**
- Forcer le runner V1 rend le chemin de génération par diffusion incompatible
  (`diffusion_gemma.py` utilise `req_states`/`draft_tokens`, structures du
  runner V2) ; des `-1` seraient utilisés comme indices dans un gather.
- Cause d'origine de l'erreur UVA : sous WSL2 la mémoire épinglée est
  désactivée → UVA indisponible → le runner V2 échoue à l'init
  (`RuntimeError: "UVA is not available"`). Dans vLLM main,
  `VLLM_WSL2_ENABLE_PIN_MEMORY=1` réactive la mémoire épinglée sous WSL2 ;
  **inconnu si notre image connaît cette variable** (à vérifier via
  `./diagnose_image.sh`).

**Prouvé :** chargement + démarrage + `/v1/models` avec runner V1 forcé ;
crash à la 1re génération (assertion GPU scatter/gather).
**NON prouvé :** qualité du modèle ; rôle éventuel des poids fusionnés Jainius
(un témoin `D_p` avec la même config est nécessaire pour les distinguer).

**Correctif appliqué (à valider sur GPU) :** le compose ne définit plus
`VLLM_USE_V2_MODEL_RUNNER` (runner V2 = défaut de vLLM) ; expose
`VLLM_WSL2_ENABLE_PIN_MEMORY=1` et `CUDA_LAUNCH_BLOCKING=0` avec défauts
explicites ; le runner V1 ne se force que via `VLLM_USE_V2_MODEL_RUNNER_OVERRIDE=0`
dans `.env` (déconseillé). Voir l'arbre de décision dans `README.md`.
## 2026-10-19 (WSL2, RTX 5090 32 Go) — runner V2 : faits nouveaux

**Diagnostiqué puis exécuté (sorties brutes dans le rapport
J4Bis-bis : `vLLM/RAPPORT_J4BIS-BIS.md`, sources `_artefact/vllm_test/`) :**
- `gemma` (vLLM 0.22.1rc1.dev357) : `VLLM_WSL2_ENABLE_PIN_MEMORY` est
  **INCONNUE** de l'image — log serveur : `WARNING [envs.py:2111] Unknown
  vLLM environment variable detected: VLLM_WSL2_ENABLE_PIN_MEMORY` (le
  défaut=1 du compose est donc sans effet).
- `gemma` + runner V2 (compose par défaut) : échec à l'init en ~22 s —
  `gpu_worker.py:333 → GPUModelRunnerV2 → states.py:33 → buffer_utils.py:48 :
  RuntimeError("UVA is not available")`. L'UvaBuffer (offload des poids)
  exige UVA, indisponible sous WSL2 (pin_memory désactivé :
  `WARNING [interface.py:757] Using 'pin_memory=False' as WSL is detected`).
- `gemma` + `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1` : testé, ne contourne
  pas (même erreur UVA).
- `latest` (vLLM 0.24.0, présente localement) : la variable pin memory est
  **CONNUE** (`envs.py` + `platforms/cuda.py`), l'init UVA **passe**, mais
  échec à `load_model` → `RuntimeError: CUDA driver error: device not ready`
  (`routed_experts.py:168`, allocation MoE) — **reproduit 2 fois**, non résolu.
- `req_states` présent dans `diffusion_gemma.py` des deux images (lignes
  954/1083/1153/... : `prepare_inputs(self, input_batch, req_states)`,
  `req_states.draft_tokens[...]`) — le chemin diffusion utilise bien les
  structures du runner V2.

**Non prouvé :** cause racine du `device not ready` (0.24.0) — piste non
testée : désactiver l'offload CPU (`VLLM_CPU_OFFLOAD_GB=0`) exigerait
≥ 48,10 GiB de VRAM, impossible sur 32 Go ; tag plus récent ; Linux natif.
Le témoin D_p n'a pas été exécuté : les trois échecs surviennent à l'init /
au chargement, indépendamment du checkpoint (l'incident V1 du 2026-10-18
reste le seul crash de génération).

## WSL2 : UVA indisponible avec le runner V2

- **Symptôme (observé) :** `RuntimeError: UVA is not available` à l'init du
  runner V2 sous WSL2.
- **Piste (non vérifiée dans notre image) :** `VLLM_WSL2_ENABLE_PIN_MEMORY=1`
  (introduite dans vLLM main). Lancer `./diagnose_image.sh` pour savoir si
  l'image la connaît ; si elle est absente de `envs.py`/`platforms/cuda.py`,
  la définir n'a aucun effet → image plus récente ou Linux natif.

## `--enforce-eager` sous WSL2

- **Prouvé (2026-10-18) :** nécessaire avec le runner V1 forcé + offload CPU
  sous WSL2 (contourne un blocage `torch.compile` au démarrage) ; débit réduit
  (pas de CUDA graphs), 1re requête plus lente (JIT Triton).
- Le compose l'active par défaut (`VLLM_ENFORCE_EAGER=1`, flag injecté
  seulement si =1). On ne sait pas s'il est encore nécessaire avec le runner
  V2 — à tester en le désactivant après stabilisation.

## Durées observées (WSL2, `/mnt/d` en 9P, 2026-10-18)

- ~14,5 min jusqu'à "Application startup complete", dont ~3 min de lecture
  des poids via 9P et ~6 min de profilage/autotune ; 1re requête plus lente
  (JIT Triton). Copier le checkpoint dans le FS natif WSL accélère la lecture
  (voir `README.md`).

## Limites (non corrigées)

- Aucun test d'inférence GPU automatisé (aucune CI ne remplace un GPU) :
  `smoke_chat.py` + témoin `D_p` restent à exécuter sur machine GPU.
- Rôle des poids Jainius dans le crash : non déterminé (témoin manquant).

## 2026-10-19 (agent, WSL2 / Ubuntu 24.04, RTX 5090) — E0 : config A reproduite, opération fautive identifiée

**Environnement mesuré :** WSL2, distribution **Ubuntu** 24.04 (nom de machine
hôte « Morpheus » ≠ nom de distribution), `docker` 29.6.1 / Compose v5.1.4,
`nvidia-smi` → RTX 5090, 32 607 MiB (31,84 GiB), `free -h` → **30 GiB de RAM au
total, ~29 GiB disponibles**, swap 8 GiB.

**E0 — reproduction de la config A** (image `vllm/vllm-openai:gemma`,
ID `sha256:9c719fc0c869…`, vLLM 0.22.1rc1.dev357 ; runner V1 forcé via
`VLLM_USE_V2_MODEL_RUNNER_OVERRIDE=0` ; `--enforce-eager` ; `--cpu-offload-gb 30` ;
`--gpu-memory-utilization 0.90` ; `CUDA_LAUNCH_BLOCKING=1` ;
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) :

- Chargement **OK** : `Model loading took 18.08 GiB memory and 389.63 s` ;
  `Available KV cache memory: 8.56 GiB` ; `GPU KV cache size: 52,547 tokens` ;
  `init engine … took 449.20 s` ; « Application startup complete » à ~14,5 min
  (14:13:16 → 14:27:54). `GET /v1/models` OK (`id: jainius`,
  `max_model_len: 4096`).
- `POST /v1/chat/completions` → **HTTP 500 en 174,3 s** :
  `EngineCore encountered an issue` / `EngineDeadError`.
- **Opération réellement fautive** (CUDA_LAUNCH_BLOCKING=1, traceback) :
  ```
  vllm/v1/worker/gpu_model_runner.py:1797  in _prepare_input_ids
      self.input_ids.gpu.scatter_(
  torch.AcceleratorError: CUDA error: device-side assert triggered
  ```
  cause CUDA : `ScatterGatherKernel.cu:163 Assertion idx_dim >= 0 &&
  idx_dim < index_size && "scatter gather kernel index out of bounds"`.
- **Chaîne causale identifiée** (source de l'image `gemma`,
  `gpu_model_runner.py` lignes 1730-1800) : `_prepare_input_ids` calcule
  `draft_len = len(scheduled_spec_decode_tokens.get(req_id, ()))` puis
  `sample_flattened_indices.append(flattened_index - draft_len)`.
  Pour la diffusion, le scheduler remplit `scheduled_spec_decode_tokens` avec
  **256 valeurs `-1`** (canvas de diffusion, `canvas_length: 256` dans
  `config.json`) et `num_scheduled_tokens = 256` → `flattened_index = 255`,
  `draft_len = 256` → **index `-1`** dans le tenseur d'indices du `scatter_` →
  assertion CUDA.
  Autrement dit : le runner V1 interprète la liste de placeholders diffusion
  comme une liste de *draft tokens* ; l'hypothèse « incompatibilité runner V1
  / chemin diffusion » (H1) est **confirmée sur le mécanisme**, dans cette
  image et avec ce runner.

**Conséquence pour la RAM (fait mesuré, nouveau) :** `--cpu-offload-gb 30`
sur 30 GiB de RAM WSL2 fonctionne mais **sature** la machine : pendant le
service, `free -h` → 30 GiB utilisés, 371 MiB libres, **swap 7,9 GiB
utilisés**. Le couple (offload 30 Go, RAM 30 Go) est à la limite de cette
machine.

## 2026-10-19 (agent, WSL2 / Ubuntu 24.04) — E1 : config C avec `expandable_segments` désactivé

**E1** — image `vllm/vllm-openai:latest` (ID `sha256:251eba5cc7c1…`,
vLLM **0.24.0**), runner **V2** (défaut, pas d'override), `--enforce-eager`,
`--cpu-offload-gb 30`, `--max-model-len 4096`,
`--gpu-memory-utilization 0.90`, `CUDA_LAUNCH_BLOCKING=1`,
**`PYTORCH_CUDA_ALLOC_CONF=` (vide → `expandable_segments` désactivé)**.

- RAM disponible mesurée **avant** le lancement : `free -h` → 30 GiB total,
  29 GiB disponibles, swap 8 GiB. RAM vue par vLLM pendant le chargement :
  `weight_utils.py:849 — Filesystem type for checkpoints: 9P. Checkpoint size:
  48.10 GiB. Available RAM: 26.79 GiB.`
- `Model loading took 48.58 GiB and 305.4 seconds` → **le chargement
  RÉUSSIIT** (contrairement à la config C).
- Puis : `Available KV cache memory: -1.71 GiB` →
  `ValueError: No available memory for the cache blocks. Try increasing
  gpu_memory_utilization…` → `EngineCore failed to start` → conteneur arrêté
  après **362 s**.
- **L'erreur `CUDA driver error: device not ready` (config C) ne s'est PAS
  reproduite** : le mode de défaillance a changé (chargement OK, puis plus de
  place pour le cache KV).

**Différence de comptabilité GPU, V1 vs V2 (fait mesuré, même checkpoint,
même `--cpu-offload-gb 30`) :**

| Runner | Image | Ligne de log | Poids comptés contre le budget VRAM |
|---|---|---|---|
| V1 forcé | `gemma` 0.22.1rc1 | `gpu_model_runner.py:5182` | **18,08 GiB** (seule la part GPU) |
| V2 (défaut) | `latest` 0.24.0 | `model_runner.py:302` | **48,58 GiB** (modèle entier, offload CPU inclus) |

Conséquence : avec le runner V2, l'offload CPU **n'allège pas** le budget
`--gpu-memory-utilization` (28,66 GiB à 0,90) — d'où le budget KV cache
négatif. C'est la raison mécanique pour laquelle « offload 30 Go + runner V2 »
ne peut pas fonctionner sur 32 Go de VRAM, indépendamment de `expandable_segments`.

**H2 (`expandable_segments` → `device not ready`) : NI CONFIRMÉE NI INFIRMÉE.**
Un seul essai, avec un facteur parasite non contrôlé (RAM disponible :
26,79 GiB vus par vLLM, variable selon l'état de la machine). Le symptôme
visé n'est pas réapparu, mais la config échoue toujours — pour une raison
différente et **déterminée** (budget KV cache négatif, V2 comptant 48,58 GiB).

## 2026-10-19 (agent) — E2 : sans offload CPU, FP8 à la volée → **PREMIÈRE GÉNÉRATION**

**E2** — image `vllm/vllm-openai:latest` (ID `sha256:251eba5cc7c1…`, vLLM
0.24.0), runner V2, `--cpu-offload-gb 0`, `--quantization fp8` (quantification
**à la volée** des poids BF16), `--max-model-len 2048`, `--max-num-seqs 1`,
`--gpu-memory-utilization 0.92`, `--enforce-eager`,
`PYTORCH_CUDA_ALLOC_CONF=` (vide), `CUDA_LAUNCH_BLOCKING=0`.

- **Chargement OK** : `Model loading took 25.85 GiB and 304.26 s` (vs 48,10 GiB
  en BF16) ; `Available KV cache memory: 2.57 GiB` ; `GPU KV cache size:
  12,137 tokens` ; `init engine … took 14.67 s` ; « Application startup
  complete » après **~6 min** (14:39:11 → 14:45:03).
- Noyaux FP8 retenus : `Selected CutlassFP8ScaledMMLinearMethod for
  Fp8PerTensorOnlineLinearMethod` et `Using TRITON Fp8 MoE backend`
  (FP8 **par tenseur**, statique).
- VRAM pendant le chargement (échantillons 10 s) : 762 MiB → 9 721 → 14 457 →
  19 170 → 24 660 → 29 368 → **31 995 MiB** (193 MiB libres) ; RAM hôte
  stabilisée à ~4-6 GiB (pas d'offload).
- **Génération : HTTP 200** (2,7 s), `finish_reason: length`, **texte non vide
  (301 caractères)** → `eval/smoke_jainius_20261019_164553.json`.
- **MAIS le texte est dégénéré** : `smoke_chat.py` → `ÉCHEC : vocabulaire
  dégénéré (12 mots uniques / 64)`. Le texte produit ressemble à
  « … popularity dissatisfaction lapse rough lapse dissatisfaction popularity
  popularity popularity popularity own dissatisfaction own own own … ».
  Le même résultat (texte vide de sens, répétitif) a été observé avec
  `max_tokens` ∈ {64, 128, 256} et `temperature` ∈ {0, 0,8, 1,0}, et avec
  trois prompts différents (« Hello », « What is the capital of France? »,
  « 2+2= ») → ce n'est **pas** un artefact de prompt ni de température.

**E2b — E2 sur l'image `gemma` (FP8, sans offload) : ÉCHEC À L'INIT.**
`RuntimeError: UVA is not available` (`gpu_worker.py:333 → GPUModelRunnerV2 →
states.py:33 → buffer_utils.py:142 → buffer_utils.py:48`), conteneur arrêté
après 50 s. Cause : dans l'image `gemma`, `platforms/interface.py:758`
(`is_pin_memory_available()`) renvoie **False inconditionnellement sous
WSL** — aucune variable d'environnement ne peut réactiver la mémoire
épinglée, donc pas d'UVA, donc pas de runner V2. L'image `gemma` est donc
**inutilisable sur cette machine** (V2 : pas d'UVA ; V1 : crash de
`_prepare_input_ids`, voir E0).

**Comparaison des implémentations DiffusionGemma des deux images** (extraction
statique, `docker create` + `docker cp`, aucun conteneur lancé) : le code de
dénoïsation est équivalent (`argmax_canvas`, `entropy_bound`, `t_min`/`t_max`,
`stability_threshold`, `confidence_threshold` lus de `generation_config.json`
dans les deux images ; `router(residual)` dans les deux). La seule
différence fonctionnelle relevée (`router_uses_prenormed_input`, présent
seulement dans `gemma`) a le **même effet** dans les deux images.
→ L'hypothèse « implémentation DiffusionGemma cassée dans `latest` » n'est
**pas** étayée par le code ; l'hypothèse « quantification FP8 par tenseur
trop grossière pour un MoE à 128 experts » reste ouverte (cf. E3).

## 2026-10-19 (agent) — E3 (variante d'E2, consignée) : FP8 **par canal de sortie**

Écart explicite par rapport à la liste E2 du cahier des charges, consigné ici
et dans le rapport : `--quantization fp8` (E2) applique une **échelle FP8
statique par tenseur** à chaque expert du MoE (128 experts/layer,
`moe_intermediate_size: 704`) ; l'image `latest` (0.24.0) propose un mode
`fp8_per_channel` (`Fp8PtpcOnlineMoEMethod` : « Quantizes each expert's
weights per output channel during loading. Activations are quantized
dynamically per token at runtime. »). Hypothèse testée : l'échelle par
tenseur est trop grossière → routage MoE perturbé → texte dégénéré.

**Résultat d'E3** : **inchangé** — même texte dégénéré avec l'échelle par canal
de sortie qu'avec l'échelle par tenseur (`eval/smoke_jainius_20261019_170010.json`,
verdict FAIL). L'hypothèse « granularité de l'échelle FP8 » est donc **écartée** :
la cause n'est pas la quantification.

## 2026-10-19 (agent) — E4/E5/E6 : cause racine = le checkpoint était fusionné à **α=0.8**

**Diagnostic (CPU, sans GPU).** Le checkpoint servi pendant E2/E3
(`checkpoints/Jainius`) n'était **pas** le résultat de la commande documentée
(`merge_jainius.py --alpha 0.3`) mais d'un ancien build à **α=0.8** : la mesure
`α_eff = ||Jainius − D_p|| / ||τ_A||` donne **0,799 ± 0,002** sur les
projections (q/k/v/o/mlp), 0,79 sur les norms, 0,80 sur `layer_scalar`, 0,70 sur
`embed_tokens` → embeddings et norms avaient été fusionnés (variante
`--include-norms-embed`). Le build livré ne contenait **aucun**
`merge_config.json` (provenance absente) et sa model card annonçait
explicitement « α=0.8 ».

Contrôle d'intégrité de τ_A (indépendant du merge) : `max|τ_A − (A_p − A_b)| = 0`
(exact) sur 6 tenseurs de contrôle ; `||τ_A|| / ||D_p|| ≈ 0,55–0,72` sur les
projections → à α=0.8, le delta atteint **~50 % de la norme** des projections
d'attention (cohérent avec le texte dégénéré), à α=0.3 il atteint ~18 %.

**E4 — refusion à α=0.3 (défaut du script : norms/embeddings exclus,
355 tenseurs fusionnés / 692 inchangés), même config de service que E2 :**

- Chargement OK (`Model loading took 25.85 GiB`), « Application startup complete ».
- `smoke_chat.py` : **4/4 PASS**, texte cohérent
  (« 7 × 8 font 56. », « The capital of France is Paris. », photosynthèse,
  « Un modèle de diffusion génère du contenu en apprenant à transformer du
  bruit aléatoire en données structurées. ») → `eval/smoke_jainius_a03_*.json`.
- **PASS** : le verdict « texte dégénéré » d'E2/E3 portait sur le build α=0.8,
  pas sur α=0.3.

**E5 — refusion à α=0.3 avec `--include-norms-embed` (656 fusionnés / 391
inchangés)** : chargement OK, **3/3 PASS**, texte cohérent mais avec plus
d'artefacts lexicaux que la variante par défaut (« The capital of **way** is
Paris ») → la variante par défaut (embeddings/norms inchangés) est livrée.

**E6 — témoin `D_p` (`checkpoints/diffusiongemma_it`), config identique à E2/E3 :**
**3/3 PASS**, texte cohérent (« 7 × 8 font 56. ») → la pile de service
(image 0.24.0, FP8 à la volée, runner V2, `--enforce-eager`) **n'est pas en
cause**. L'image `gemma` (0.22.1rc1) reste inutilisable sur cette machine
(UVA), mais `latest` épinglée par digest fonctionne.

**Conclusion (vLLM) :** l'incident « texte dégénéré » était un problème de
**checkpoint périmé** (α=0.8), pas de configuration de service. Avec le
checkpoint livré (α=0.3), la génération est saine. La config par défaut du
compose (FP8 à la volée, sans offload CPU, `--max-model-len 2048`) est **validée
sur GPU** ; toute comparaison se fait en FP8, pas en BF16.

**Évaluation complète (103 prompts, `scripts/eval_compare.py`)** : Jainius
α=0.3 → 0 % dégénéré, 88,1 % de justesse, 75,0 % de format, 21,1 tok/s ;
D_p → 0 % dégénéré, 100 % de justesse, 93,8 % de format, 116,4 tok/s.
Détail : `eval/RESULTS.md`, `eval/results_*.json`.
