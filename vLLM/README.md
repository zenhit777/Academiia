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
- Image `${VLLM_IMAGE}` postérieure au 2026-06-10, CUDA ≥ 12.9 (Blackwell/5090).
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
| `VLLM_IMAGE` | `vllm/vllm-openai:latest` | Image officielle (doit inclure le support dLLM) |
| `VLLM_MODEL_HOST_PATH` | `../checkpoints/Jainius` | Checkpoint hôte vu depuis WSL |
| `VLLM_SERVED_MODEL_NAME` | `jainius` | Nom exposé à l'API |
| `VLLM_PORT` | `8000` | Port hôte |
| `HF_TOKEN` | (vide) | Optionnel, contenus gatés uniquement |
| `VLLM_DTYPE` | `bfloat16` | `fp8` exige un checkpoint pré-quantifié (non fourni) |
| `VLLM_CPU_OFFLOAD_GB` | `30` | MoE délestés en RAM (51.6 Go vs ~31 Go VRAM) |
| `VLLM_EXTRA_ARGS` | (vide) | Passés tels quels à `vllm serve` |

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
