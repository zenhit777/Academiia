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
