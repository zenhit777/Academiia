# vLLM — servir Jainius (image officielle)

Servez le checkpoint `checkpoints/Jainius` avec l'image **vLLM officielle**,
qui supporte nativement DiffusionGemma (PR `#45163`, blog vLLM du 10-06-2026) :
scheduling asynchrone diffusion, pas de synchro CPU-GPU, sampler
`EntropyBound` + adaptive stopping gérés côté moteur.

Chiffres du Technical Report (1× H100, FP8, batch 1) : ~1500 tok/s,
~20 tokens/forward, ~12 steps effectifs (max 48). Avantage diffusion jusqu'à
~32 requêtes concurrentes ; au-delà, l'AR repasse devant.

## Prérequis

- WSL2 + Docker avec runtime NVIDIA (`nvidia-smi` OK sous WSL).
- Image `${VLLM_IMAGE}` = `vllm/vllm-openai:gemma` (support natif DiffusionGemma, blog vLLM 2026-06-10). CUDA ≥ 12.9 (Blackwell/5090).
- Checkpoint présent (`VLLM_MODEL_HOST_PATH`, défaut : `../checkpoints/Jainius`).

## Démarrage

```bash
cd vLLM
cp .env.template .env   # déjà fait : ajuster si besoin
docker compose up       # Ctrl+C pour arrêter
./smoke.sh              # test de génération (autre terminal)
```

Le serveur écoute `http://localhost:8000` (API OpenAI-compatible,
modèle `${VLLM_SERVED_MODEL_NAME}`, défaut `jainius`).

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
| `VLLM_EXTRA_ARGS` | (vide) | Flags supplementaires, decoupes par shell (multi-flags OK) |

## Entrypoint (pourquoi `entrypoint: ["bash","-c"]`)

L'image `vllm/vllm-openai` définit `ENTRYPOINT ["vllm","serve"]`. Un `command:`
simple serait donc **concaténé** à cet entrypoint (`vllm serve bash -c '…'`) et
vLLM rejetterait ces arguments. Le compose neutralise l'entrypoint avec
`entrypoint: ["bash","-c"]` et fournit tout le script comme **UN SEUL élément**
de `command` (avec `bash -c`, seul le premier argument est lu comme script).

Vérification de l'entrypoint de l'image :

```bash
docker image inspect vllm/vllm-openai:gemma --format '{{json .Config.Entrypoint}}'
# -> ["vllm","serve"]
```

C'est cette forme qui permet à `VLLM_EXTRA_ARGS` de contenir **plusieurs** flags
(découpés par le shell du conteneur). Attention : avec cet entrypoint,
`docker compose run --rm vllm --version` **ne fonctionne pas** (bash reçoit
`--version` comme script) — pour un smoke du conteneur, passer une commande
complète en un seul argument, p.ex. :

```bash
docker compose run --rm -T --no-deps vllm 'echo OK; vllm --version'
```

## Notes

- **VRAM** : pas de `load_in_8bit` ici (dépannage `transformers` uniquement).
  Le fit tient via `--cpu-offload-gb` ; la RAM hôte (~67 Go) l'absorbe.
- **Montage Windows** : via `/mnt/d`, le chargement des ~52 Go est lent.
  Pour du bench : copier le checkpoint dans le FS WSL et ajuster `VLLM_MODEL_HOST_PATH`.
- **Débit max** : batch 1, sampler par défaut (adaptive stopping ON),
  mode no-think pour le volume. Mesurer TPF/steps effectifs/TPS
  (éq. 10/12 du rapport) avant toute allégation — faiblesse n°2 de `AUDIT.md`.
- **Dépannage** : `restart: "no"` volontaire (un OOM doit se voir).
  `docker compose logs vllm` pour le détail.
