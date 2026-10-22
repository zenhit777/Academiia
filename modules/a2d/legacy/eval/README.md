# eval/ — preuves d'inférence

Contenu de ce dossier : les **preuves brutes** des générations de texte
réalisées le 2026-10-19 sur vLLM (RTX 5090 32 Go, WSL2). Rien n'est calculé
à la main : tout vient de `infra/vllm/smoke_chat.py` (smoke) et de
`modules/a2d/eval/eval_compare.py` (évaluation).

| Fichier | Provenance |
|---|---|
| `prompts.jsonl` | Jeu de 103 prompts (faits FR/EN, maths, formats, questions libres) — champs `id`, `category`, `lang`, `prompt`, `expected` (attendu), `format` (json/list/one_line) |
| `results_Dp.json` | Évaluation complète de `D_p` (`checkpoints/diffusiongemma_it`), 103 prompts, même config que le checkpoint livré |
| `results_jainius_a03.json` | Évaluation complète de **Jainius α=0.3** (checkpoint livré, 355 tenseurs fusionnés) |
| `smoke_jainius_20261019_164553.json` | Smoke du **build α=0.8** (ancien checkpoint livré) — verdict FAIL, texte dégénéré |
| `smoke_jainius_20261019_170010.json` | Smoke du build α=0.8 en FP8 par canal de sortie (E3) — verdict FAIL, même texte dégénéré |
| `smoke_diffusiongemma_it_20261019_170756.json` | Smoke du témoin `D_p` — verdict PASS, texte cohérent |
| `smoke_jainius_a03_*.json` | Smokes du checkpoint α=0.3 après refusion — verdicts PASS |
| `smoke_jainius_a03_ne_*.json` | Smokes de la variante α=0.3 `--include-norms-embed` (656 tenseurs fusionnés) — verdicts PASS |
| `RESULTS.md` | Rapport de comparaison (protocole, chiffres, ce qu'on peut conclure, limites) |

> Les deux `smoke_jainius_2026*.json` (16:45 et 17:00) correspondent à l'ancien
> build **α=0.8** : c'est ce checkpoint périmé — et non la configuration de
> service — qui produisait le texte dégénéré. Voir `docs/KNOWN_ISSUES.md`
> (« Inférence GPU vLLM (2026-10-19) »).

Reproductible :

```bash
cd vLLM && docker compose up -d      # config FP8 : voir infra/vllm/.env.template
python modules/a2d/eval/eval_compare.py \
    --base http://localhost:8000/v1/chat/completions \
    --model jainius --prompts modules/a2d/legacy/eval/prompts.jsonl \
    --output eval/results_jainius_a03.json
```
