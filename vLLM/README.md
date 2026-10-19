# vLLM — servir Jainius (image officielle)

Servez le checkpoint `checkpoints/Jainius` avec l'image **vLLM officielle**,
qui supporte nativement DiffusionGemma (PR `#45163`, blog vLLM du 10-06-2026) :
scheduling asynchrone diffusion, pas de synchro CPU-GPU, sampler
`EntropyBound` + adaptive stopping gérés côté moteur.

Chiffres du Technical Report (1× H100, FP8, batch 1) : ~1500 tok/s,
~20 tokens/forward, ~12 steps effectifs (max 48). Avantage diffusion jusqu'à
~32 requêtes concurrentes ; au-delà, l'AR repasse devant.

**Statut (2026-10-18, WSL2) :** chargement et démarrage validés ; la 1re
génération a crashé (assertion GPU scatter/gather) avec le runner V1 forcé.
Cause probable (non prouvée) : incompatibilité runner V1 avec la diffusion.
Voir `KNOWN_ISSUES.md` et l'arbre de décision ci-dessous.

## Prérequis

- WSL2 + Docker avec runtime NVIDIA (`nvidia-smi` OK sous WSL).
- Image `${VLLM_IMAGE}` = `vllm/vllm-openai:gemma` (support natif DiffusionGemma, blog vLLM 2026-06-10). CUDA ≥ 12.9 (Blackwell/5090).
- Checkpoint présent (`VLLM_MODEL_HOST_PATH`, défaut : `../checkpoints/Jainius`).

## Démarrage

```bash
cd vLLM
cp .env.template .env   # ajuster si besoin
docker compose up       # Ctrl+C pour arrêter ; ~14,5 min de chargement (WSL2, /mnt/d en 9P)
```

Le serveur écoute `http://localhost:8000` (API OpenAI-compatible,
modèle `${VLLM_SERVED_MODEL_NAME}`, défaut `jainius`). Test de génération :

```bash
python smoke_chat.py    # autre terminal ; stdlib uniquement
```

## Arbre de décision (dépannage WSL2)

1. **`./diagnose_image.sh`** (sans GPU, sans chargement) : version de vLLM
   dans l'image, présence de `VLLM_WSL2_ENABLE_PIN_MEMORY`,
   `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA`, `VLLM_USE_V2_MODEL_RUNNER` dans
   `envs.py`, de `WSL2_ENABLE_PIN_MEMORY` dans `platforms/cuda.py`, usage de
   `req_states` dans `diffusion_gemma.py`.
2. **Si `VLLM_WSL2_ENABLE_PIN_MEMORY` est connue de l'image :** lancer le
   compose par défaut (runner V2 = défaut de vLLM, pin memory activée).
   N'utilisez `VLLM_USE_V2_MODEL_RUNNER_OVERRIDE=0` (runner V1) qu'en dernier
   recours — le test du 2026-10-18 a crashé à la 1re génération avec lui.
3. **Sinon :** la variable n'a aucun effet dans cette image → tester une image
   plus récente (tag à déterminer via `docker search`/Docker Hub, ne pas
   inventer de tag) ou une machine Linux native / GPU ≥ 80 Go sans offload CPU.
4. **Si le serveur démarre :** `python smoke_chat.py` sur Jainius **puis sur
   `D_p`** (même config, un seul modèle à la fois : ~52 Go chacun) ; comparer.
   Un `D_p` qui plante pareil = problème de configuration, pas du merge.
5. **En cas de nouveau crash :** mettre `CUDA_LAUNCH_BLOCKING=1` dans `.env`
   pour localiser l'opération fautive (les erreurs CUDA sont asynchrones —
   le point affiché par défaut est trompeur), puis
   `docker logs --tail 200 jainius-vllm`.

## Variables (`.env`)

| Variable | Défaut | Rôle |
|---|---|---|
| `VLLM_IMAGE` | `vllm/vllm-openai:gemma` | Image officielle epinglee (support dLLM). `latest` interdit (non reproductible) |
| `VLLM_MODEL_HOST_PATH` | `../checkpoints/Jainius` | Checkpoint hôte vu depuis WSL |
| `VLLM_SERVED_MODEL_NAME` | `jainius` | Nom exposé à l'API |
| `VLLM_PORT` | `8000` | Port hôte |
| `HF_TOKEN` | (vide) | Optionnel, contenus gatés uniquement |
| `VLLM_DTYPE` | `bfloat16` | `fp8` exige un checkpoint pré-quantifié (non fourni) |
| `VLLM_CPU_OFFLOAD_GB` | `30` | MoE délestés en RAM (51.6 Go vs ~31 Go VRAM) |
| `VLLM_MAX_MODEL_LEN` | `4096` | Contexte max (baisser à 2048 si OOM) |
| `VLLM_MAX_NUM_SEQS` | `1` | 1 pour l'eval reproductible |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.90` | Marge VRAM pour sampler diffusion |
| `VLLM_WSL2_ENABLE_PIN_MEMORY` | `1` | Réactive la mémoire épinglée sous WSL2 (précondition UVA pour le runner V2). Variable de vLLM main : sans effet si absente de l'image (voir `diagnose_image.sh`) |
| `CUDA_LAUNCH_BLOCKING` | `0` | `1` = debug GPU (localise le crash, ralentit) |
| `VLLM_ENFORCE_EAGER` | `1` | `1` = `--enforce-eager` (nécessaire avec runner V1 + offload CPU sous WSL2, 2026-10-18 ; nécessité avec runner V2 non testée) |
| `VLLM_USE_V2_MODEL_RUNNER_OVERRIDE` | (vide) | Force le runner V1 si `0`. **Déconseillé** : crash à la 1re génération (2026-10-18). Vide = runner V2 (défaut vLLM) |
| `VLLM_EXTRA_ARGS` | (vide) | Flags supplementaires, decoupes par shell (multi-flags OK) |

## Entrypoint (pourquoi `entrypoint: ["bash","-c"]`)

L'image `vllm/vllm-openai` définit `ENTRYPOINT ["vllm","serve"]`. Un `command:`
simple serait donc **concaténé** à cet entrypoint (`vllm serve bash -c '…'`) et
vLLM rejetterait ces arguments. Le compose neutralise l'entrypoint avec
`entrypoint: ["bash","-c"]` et fournit tout le script comme **UN SEUL élément**
de `command` (avec `bash -c`, seul le premier argument est lu comme script).

**Aucun backslash dans le bloc `>-`** : le repli YAML transforme les sauts de
ligne en espaces, bash recevrait un backslash littéral suivi d'un espace
(bug corrigé en 686c5a8, regression testée par `scripts/tests/test_compose.py`).

Vérification de l'entrypoint de l'image :

```bash
docker image inspect vllm/vllm-openai:gemma --format '{{json .Config.Entrypoint}}'
# -> ["vllm","serve"]
```

## Tests sans Docker

`scripts/tests/test_compose.py` (pytest + PyYAML) valide le compose :
structure, pas de backslash, simulation `bash -c` du script (aucun argument
vide, `--enforce-eager` conditionnel, override runner). C'est ce test qui
tourne dans la CI.

## Outils de ce dossier

- `smoke_chat.py` — client HTTP stdlib : `POST /v1/chat/completions`,
  contrôles objectifs (HTTP 200, texte non vide, pas de répétition dégénérée
  évidente), sauvegarde `eval/smoke_<modèle>_<horodatage>.json`. Ces fichiers
  sont de petites preuves (~1 Ko) : ils ne sont PAS ignorés par git.
- `diagnose_image.sh` — diagnostic de l'image Docker sans GPU ni chargement
  (version vLLM, variables `envs.py`, `platforms/cuda.py`, `req_states`).
- `smoke.sh` — smoke minimal via curl (remplacé par `smoke_chat.py` pour la
  génération ; conservé pour le health-check rapide).

## Notes

- **VRAM** : pas de `load_in_8bit` ici (dépannage `transformers` uniquement).
  Le fit tient via `--cpu-offload-gb` ; la RAM hôte (~67 Go) l'absorbe.
- **Montage Windows** : via `/mnt/d` (9P), ~3 min de lecture des 52 Go —
  copier le checkpoint dans le FS natif WSL (`~/models/Jainius`) et ajuster
  `VLLM_MODEL_HOST_PATH` accélère le démarrage (~14,5 min observés dont
  ~6 min de profilage/autotune ; 1re requête plus lente, JIT Triton).
- **Débit max** : batch 1, sampler par défaut (adaptive stopping ON),
  mode no-think pour le volume. Mesurer TPF/steps effectifs/TPS
  (éq. 10/12 du rapport) avant toute allégation — faiblesse n°2 de `AUDIT.md`.
- **Dépannage** : `restart: "no"` volontaire (un OOM doit se voir).
  `docker compose logs vllm` pour le détail.
