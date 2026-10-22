# vLLM — servir Jainius (image officielle)

Servez le checkpoint `checkpoints/Jainius` avec l'image **vLLM officielle**,
qui supporte nativement DiffusionGemma (PR `#45163`, blog vLLM du 10-06-2026) :
scheduling asynchrone diffusion, pas de synchro CPU-GPU, sampler
`EntropyBound` + adaptive stopping gérés côté moteur.

Chiffres du Technical Report (1× H100, FP8, batch 1) : ~1500 tok/s,
~20 tokens/forward, ~12 steps effectifs (max 48). Avantage diffusion jusqu'à
~32 requêtes concurrentes ; au-delà, l'AR repasse devant.

**Statut (2026-10-19, WSL2, RTX 5090 32 Go) :**
- `gemma` (vLLM 0.22.1rc1.dev357) : runner V2 → échec à l'init
  (`RuntimeError: UVA is not available`, `buffer_utils.py:48`) ; runner V1
  forcé (2026-10-18) → démarrage OK mais crash à la 1re génération (assert GPU
  scatter/gather). `VLLM_WSL2_ENABLE_PIN_MEMORY` est INCONNUE de cette image
  (log : « Unknown vLLM environment variable detected »).
- `latest` (vLLM 0.24.0) : `VLLM_WSL2_ENABLE_PIN_MEMORY` CONNUE, plus d'erreur
  UVA, mais échec à `load_model` : `CUDA driver error: device not ready`
  (reproduit 2 fois) — non résolu à ce jour.
- Voir `KNOWN_ISSUES.md` et l'arbre de décision ci-dessous.

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
3. **Sinon :** la variable n'a aucun effet dans cette image. **Testé le
   2026-10-19 :** `vllm/vllm-openai:latest` (vLLM 0.24.0, déjà présente
   localement) CONNAÎT la variable et passe l'init UVA, mais échoue ensuite à
   `load_model` (`CUDA driver error: device not ready`, reproduit 2 fois —
   non résolu). `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1` sur `gemma` : testé,
   ne contourne pas (même erreur UVA). Reste à tester : tag plus récent que
   `latest`, ou Linux natif / GPU ≥ 80 Go sans offload CPU.
4. **Si le serveur démarre :** `python smoke_chat.py` sur Jainius **puis sur
   `D_p`** (même config, un seul modèle à la fois : 48,10 GiB (~52 Go) chacun) ; comparer.
   Un `D_p` qui plante pareil = problème de configuration, pas du merge.
5. **En cas de nouveau crash :** mettre `CUDA_LAUNCH_BLOCKING=1` dans `.env`
   pour localiser l'opération fautive (les erreurs CUDA sont asynchrones —
   le point affiché par défaut est trompeur), puis
   `docker logs --tail 200 jainius-vllm`.

## Qui exécute quoi (WSL / Windows)

- **Exécution GPU (agent de dev, ops vLLM) : dans WSL2**, distribution
  **Ubuntu** 24.04 (le nom de machine hôte « Morpheus » n'est pas le nom de la
  distribution). Mesuré le 2026-10-19 depuis ce shell : `docker --version`
  (29.6.1, Compose v5.1.4) et `nvidia-smi` (RTX 5090, 32 607 MiB) répondent
  **directement**, sans passer par Windows.
- **Revue côté Windows (PowerShell/CMD) : sans Docker** — la relecture
  documentaire et les diffs git n'exigent ni GPU ni runtime Docker. Les
  commandes Linux restent invocables depuis Windows via `wsl.exe -e <cmd>`
  (p. ex. `wsl.exe -e docker compose -f infra/vllm/docker-compose.yml config`).
  Depuis WSL, `wsl.exe -e …` échoue (`execvpe(nvidia-smi) failed`) : on est
  déjà dans WSL, les commandes se lancent directement.

> **RAM WSL2 : 30 GiB au total, ~29 GiB disponibles** (`free -h`, mesuré le
> 2026-10-19). Le checkpoint fait 48,14 GiB : en BF16, `--cpu-offload-gb 30`
> est **impossible** ici (30 GiB de RAM pour les poids seuls, hors tampons).
> Voir `../modules/a2d/legacy/docs/GUIDE_RESTANT.md` (les trois chiffres : poids / budget vLLM /
> VRAM).

## Règle VRAM (LM Studio) — pour les agents

Avant TOUTE opération GPU : mesurer
`nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv`
(`memory.used` inclut ce qu'occupe LM Studio). Besoin vLLM ≈
`VLLM_GPU_MEMORY_UTILIZATION × total` (défaut 0,90 → ≈ 28,8 GiB sur 32 GiB).
Si un modèle LM Studio est chargé ET que `memory.free < besoin + 1 GiB` :
STOPPER et demander la permission à l'utilisateur (citer mémoire libre
mesurée, besoin estimé, modèle détecté). Ne JAMAIS décharger un modèle LM
Studio, tuer un processus ou modifier ses réglages. Après chaque série de
tests GPU : `docker compose down`, re-mesurer et rapporter `memory.used`.
Opérations CPU seules (pytest, merge, verify) : aucune permission requise.

## Variables (`.env`)

| Variable | Défaut | Rôle |
|---|---|---|
| `VLLM_IMAGE` | `vllm/vllm-openai:gemma` | Image officielle epinglee (support dLLM). `latest` interdit (non reproductible) |
| `VLLM_MODEL_HOST_PATH` | `../checkpoints/Jainius` | Checkpoint hôte vu depuis WSL |
| `VLLM_SERVED_MODEL_NAME` | `jainius` | Nom exposé à l'API |
| `VLLM_PORT` | `8000` | Port hôte |
| `HF_TOKEN` | (vide) | Optionnel, contenus gatés uniquement |
| `VLLM_DTYPE` | `bfloat16` | `fp8` exige un checkpoint pré-quantifié (non fourni) |
| `VLLM_CPU_OFFLOAD_GB` | `30` | MoE délestés en RAM (48,10 GiB lu par vLLM vs ~31 Go VRAM) |
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
(bug corrigé en 686c5a8, regression testée par `modules/a2d/tests/test_compose.py`).

Vérification de l'entrypoint de l'image :

```bash
docker image inspect vllm/vllm-openai:gemma --format '{{json .Config.Entrypoint}}'
# -> ["vllm","serve"]
```

## Tests sans Docker

`modules/a2d/tests/test_compose.py` (pytest + PyYAML) valide le compose :
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
  (éq. 10/12 du rapport) avant toute allégation - faiblesse n°2 de `modules/a2d/legacy/docs/AUDIT.md`.
- **Dépannage** : `restart: "no"` volontaire (un OOM doit se voir).
  `docker compose logs vllm` pour le détail.
