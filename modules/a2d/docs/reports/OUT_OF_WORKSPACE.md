# Inventaire des écritures hors du dépôt — et commandes de suppression PROPOSÉES

> **Document historique (J6–J8).** Les chemins cités dans ce rapport sont
> ceux de l'époque : la table de migration des chemins est
> [`docs/adr/path-migration.md`](../../../../docs/adr/path-migration.md).

**Date** : 2026-10-20 · **Mission** : J6bis, tâche C5 · **Mode** : **LECTURE SEULE**

> **Aucune suppression n'a été effectuée.** Les commandes de la section 3 sont des
> **propositions** : elles ne seront exécutées qu'après **accord écrit de
> l'utilisateur** (règle de conduite R2 de la mission J6bis). Chaque chemin
> listé ici est un chemin **hors du dépôt** ; les écritures ont été produites par
> les missions précédentes (J4Bis, J5, J6) et par l'outillage tiers
> (`pip`, `uv`, HuggingFace, Docker, NLTK).
>
> La règle désormais en vigueur : **toute écriture doit vivre dans le dépôt**
> (`AGENTS.md` règle 9). Les harnais d'évaluation redirigent leurs caches vers
> `harness/<famille>/.cache/` (`harness/dream/setup.sh`).

## 1. Inventaire

Tailles mesurées par `du -sb` (octets, contenu *réel* des fichiers, pas les
blocs alloués) ; propriétaire et date par `stat`.

### 1.1 Le harnais Dream hors dépôt (le cas J6 principal)

| Chemin | Taille | Propriétaire | Modifié | Contenu |
|---|---:|---|---|---|
| `/home/sjteb/dream-harness/` | **7 588 593 361** (7,07 Go) | `sjteb:sjteb` | 2026-10-20 00:24 | harnais J6 (V1 + V2) |
| ↳ `venv/` | 7 536 885 618 | `sjteb:sjteb` | 2026-10-19 | Python 3.12 + torch 2.8.0+cu128 + fork lm-eval |
| ↳ `Dream/` | 51 169 450 | `sjteb:sjteb` | 2026-10-19 | clone `DreamLM/Dream`, commit `31f94a60d187e3fd481fee3bbc2c732eb94a879c` |
| ↳ `logs/` | 351 398 | `sjteb:sjteb` | 2026-10-20 | 9 logs d'exécution (≈ 343 Ko) |
| ↳ `tasks_a2d/` | 104 123 | `sjteb:sjteb` | 2026-10-19 | tâches lm_eval (déjà dupliquées dans `harness/dream/tasks/`) |
| ↳ `valid_b1/`, `valid_b4/` | 43 720 + 24 964 | `sjteb:sjteb` | 2026-10-19 | 2 runs de validation (hors dépôt) |
| ↳ `run_dream_evals.sh`, `run_dream_evals2.sh`, `setup_dream_venv.sh` | 3 263 + 2 675 + 1 373 | `sjteb:sjteb` | 2026-10-20 | scripts (récupérés et reparamétrés dans `harness/dream/`) |
| ↳ `smoke_dream.py`, `speed_dream.py`, `speed2.py`, `setup_dream_venv.log` | 1 268 + 1 734 + 1 616 + 2 159 | `sjteb:sjteb` | 2026-10-19 | sondes de débit, log de setup |

**Récupération effectuée avant toute proposition de suppression** : la logique de
`run_dream_evals2.sh` (entrelacement par tâche, garde `SKIP`) est intégrée et
paramétrée dans `harness/dream/run_evals.sh` ; les tâches sont dans
`harness/dream/tasks/` ; l'environnement est reconstruit dans
`harness/dream/.venv` (fork épinglé sur le même commit). Le dossier hors dépôt
n'est donc plus une source unique.

### 1.2 Caches utilisateur

| Chemin | Taille | Propriétaire | Modifié | Contenu |
|---|---:|---|---|---|
| `/home/sjteb/.cache/huggingface/` | **92 586 706 003** (86,2 Go) | `sjteb:sjteb` | 2026-10-19 23:39 | `hub/` 92 365 570 737 · `datasets/` 209 968 796 · `xet/` 10 847 378 · `modules/` 313 152 |
| ↳ `hub/models--Qwen--Qwen3.6-27B-FP8` | 30 890 046 708 | | | poids de modèle (hors missions A2D) |
| ↳ `hub/models--nvidia--Qwen3.8-27B-NVFP4` | 21 944 958 547 | | | idem |
| ↳ `hub/models--RadixArk--Qwen3.8-27B-NVFP4` | 21 944 885 563 | | | idem |
| ↳ `hub/blobs` | 11 160 328 816 | | | blobs partagés |
| ↳ `hub/models--incoai--Qwen3.8-27B-DFlash2` | 4 519 511 949 | | | idem |
| ↳ `hub/models--Qwen--Qwen3-0.6B` | 1 519 182 817 | | | idem |
| ↳ `hub/datasets--*`, `modules/transformers_modules/dream_*` | — | | | datasets + code `trust_remote_code` des checkpoints Dream |
| `/home/sjteb/.cache/uv/` | **40 329 711 647** (37,6 Go) | `sjteb:sjteb` | 2026-10-12 17:08 | caches `uv` (hors missions A2D) |
| `/home/sjteb/.cache/pip/` | **8 869 011 825** (8,26 Go) | `sjteb:sjteb` | 2026-10-12 05:56 | wheels + cache HTTP pip |
| ↳ complément **J10 (2026-10-22)** | — | — | — | quelques entrées `http-v2` ajoutées par les `pip install` de la mission J10 (venv de développement `_artefact/venv-dev`, installation editable) : **écart à la règle 9 déclaré**, aucune suppression hors dépôt |
| `/home/sjteb/.cache/vllm/` | **3 641 214 554** (3,39 Go) | `sjteb:sjteb` | 2026-04-29 | `modelinfos/`, `torch_compile_cache/` (campagne vLLM) |
| `/home/sjteb/.cache/electron/` | 115 050 193 | `sjteb:sjteb` | | sans rapport avec le projet |
| `/home/sjteb/.cache/node-gyp/` | 58 885 896 | `sjteb:sjteb` | | sans rapport avec le projet |
| `/home/sjteb/nltk_data/` | **15 204 406** | `sjteb:sjteb` | 2026-10-20 00:00 | `tokenizers/punkt_tab` (téléchargé par l'évaluation IFEval) |

### 1.3 `/tmp`

| Chemin | Taille | Propriétaire | Modifié | Contenu |
|---|---:|---|---|---|
| `/tmp/tau_ref/` | **4 013 949 312** (3,74 Go) | `sjteb:sjteb` | 2026-10-19 18:33 | 3 `.npy` de τ (extraction manuelle, **hors** pipeline conforme) |
| ↳ `model_language_model_embed_tokens_weight.npy` | 2 952 790 144 | | | |
| ↳ `model_language_model_layers_15_experts_down_proj.npy` | 1 015 021 696 | | | |
| ↳ `model_language_model_layers_0_self_attn_q_proj_weight.npy` | 46 137 472 | | | |
| `/tmp/pytest-of-sjteb/` | **13 095 569** (12,49 Mo) | `sjteb:sjteb` | 2026-10-21 10:20 | `pytest-*` (J6bis + audit N1 + **J7**) — taille **périmée** dans l'inventaire initial (591 207 o) ; la croissance 8 483 981 → 13 095 569 o vient des relances `pytest` de **J7** (2026-10-21), lancées avec le `TMPDIR` **par défaut** (donc hors dépôt) : **écart à la règle 9**, déclaré ici |
| `/tmp/SUMMARY_test.json` | **48 867** | `sjteb:sjteb` | 2026-10-20 13:03 | résumé de test écrit **hors dépôt** (jamais inventorié jusqu'ici) |
| `/tmp/SUMMARY_test.md` | **2 814** | `sjteb:sjteb` | 2026-10-20 13:03 | idem (`SUMMARY_test.json`) |
| `/tmp/a2d_smoke_transfer/` | **absent du disque** | — | — | checkpoint de **transfert α=0,9** de la mission J6 (poids 7B BF16, ≈ 15 Go), cité dans `docs/reports/J6.md` (lignes ~63 et ~309) : il **n'existe plus** ; conservé ici pour tracer ce qui a disparu (aucune suppression n'est imputable à une mission) |
| `/tmp/dgsrc/` | 56 479 | `sjteb:sjteb` | 2026-10-19 | copie de `diffusion_gemma.py` |
| `/tmp/a2d/` | 17 813 | `sjteb:sjteb` | 2026-10-20 13:12 | fixtures de test CI (J6bis) |
| `/tmp/dummy_dg/`, `/tmp/dummy_tau/`, `/tmp/dummy_out/` | 422 + 534 + 3 306 | `sjteb:sjteb` | 2026-10-20 12:5x | fixtures de test CI (J6bis) |
| `/tmp/measure.json` | 1 240 | `sjteb:sjteb` | | mesure J6 |
| `/tmp/torchinductor_sjteb/` | 0 | `sjteb:sjteb` | 2026-10-19 | cache du compilateur torch (vide) |
| `/tmp/systemd-private-*`, `/tmp/snap-private-tmp` | 0 | — | | sockets systemd : **illisibles** (permission denied) |

**Note d'inventaire (transparence sur l'outillage)** : la croissance de
`/tmp/pytest-of-sjteb/` (591 207 o → 8 483 981 o) s'explique par les
**relances de `pytest` de l'audit N1** du 2026-10-20. Ces relances
redirigent `TMPDIR` vers `_artefact/audit_j6bis/pytest_tmp/` (donc **dans le
dépôt**, conformément à la règle 9) ; le volume restant sous
`/tmp/pytest-of-sjteb/` est l'accumulation des runs antérieurs à cette
redirection. La valeur 591 207 o de l'inventaire initial était **périmée** : la
mesure à jour est **8 483 981 o** (2026-10-20 16:40).

**Complément J7 (2026-10-21)** : la même zone est repassée à **13 095 569 o**
(12,49 Mo), à cause des relances `pytest` de la mission J7, elles lancées avec
le `TMPDIR` **par défaut** (hors dépôt) — **écart à la règle 9**, déclaré ici.
Une seule écriture hors dépôt a été produite par l'agent lui-même pendant J7 :
`/tmp/tmpjyhmcic4/` (176 o, deux fichiers `*.safetensors` d'une démonstration
de contre-régression du bloc `__metadata__`) — **supprimée** le 2026-10-21
(échafaudage jetable de l'agent, pas une donnée préexistante). Aucune donnée
préexistante n'a été supprimée.

> **Correctif de la recette (mesuré le 2026-10-21)** : rediriger `TMPDIR` vers le
> dépôt **casse pytest** quand le dépôt est sur **DrvFs** (`/mnt/d`), qui ne supporte
> pas `O_TMPFILE` : `tempfile.TemporaryFile()` retombe sur un fichier nommé et la
> capture pytest échoue en `FileNotFoundError` (`capture.py:594`, au `truncate()` de
> fin de session) — **après** l'exécution des tests, donc sans message « passed ».
>
> ```bash
> # conforme à la règle 9 ET fonctionnel sur DrvFs :
> TMPDIR="$PWD/_artefact/tmp" TEMP="$PWD/_artefact/tmp" TMP="$PWD/_artefact/tmp" \
>     python3 -m pytest scripts/tests -q --capture=sys     # 73 verts
> ```
>
> `--capture=fd` échoue comme la capture par défaut sur DrvFs ; le `TMPDIR` par défaut
> (`/tmp`, ext4) fonctionne mais écrit hors dépôt.

### 1.4 Docker

`docker system df` (lecture seule) :

| Type | Total | Actif | Taille | Reclaimable |
|---|---:|---:|---:|---:|
| Images | 18 | 4 | 258,6 Go | 232,2 Go (89 %) |
| Containers | 5 | 1 | 581,6 ko | 557,1 ko (95 %) |
| Local Volumes | 1 | 1 | 2,249 Go | 0 o |
| Build Cache | 37 | 0 | 81,28 Go | 48,45 Go |

Volume : `qdrant_qdrant_data` (2,249 Go, actif). Les images vLLM 0.24.0
épinglée par digest servent à la porte J4Bis ; le volume Qdrant **n'a aucun
lien** avec ce projet et n'est pas attribuable à une mission A2D.

### 1.5 Ce qui a été déplacé dans le dépôt (bilan)

| Écriture | Devient | Taille |
|---|---|---:|
| harnais J6 (scripts, tâches, environnement) | `harness/dream/` (`.venv` 7 537 409 029 o · `.fork` 51 106 809 o · `.cache` 1 241 003 141 o) | ~8,8 Go |
| résultats J6 (dossiers à chemins absolus) | `eval/dream_a2d/<modèle>/<tâche>/` (noms portables) | ~2,6 Mo |

Le cache pip du harnais (`harness/dream/.cache/pip`, 1,23 Go) est désormais la
**seule** utilisation de `pip` du projet ; `~/.cache/pip` n'est plus alimenté par
nos commandes (`PIP_CACHE_DIR` est redirigé dans `setup.sh` et `run_evals.sh`).

## 2. Ce qui a été fait, et ce qui ne l'a pas été

* **Fait** : inventaire complet, en lecture seule, avec tailles et propriétaires.
* **Fait (complément, 2026-10-20, après l'audit N1)** : l'inventaire a été
  **complété** de trois entrées manquantes dans `/tmp` —
  `/tmp/SUMMARY_test.json` (48 867 o), `/tmp/SUMMARY_test.md` (2 814 o), écrits
  hors dépôt le 2026-10-20 13:03 et jamais inventoriés jusqu'ici, et
  `/tmp/a2d_smoke_transfer/` (checkpoint de transfert α=0,9 de la mission J6,
  ≈ 15 Go), **désormais absent du disque** — ainsi que de la taille à jour de
  `/tmp/pytest-of-sjteb/` (8 483 981 o au lieu de 591 207 o). **Toujours
  aucune suppression** : ces ajouts sont documentaires.
* **Fait** : toute la logique utile du harnais hors dépôt est **récupérée et
  versionnée** dans `harness/dream/` ; le harnais a été **reconstruit de zéro**
  et **rejoué** (smoke GSM8K n=8, mêmes items que la passe J6 à 300).
* **NON fait** : **aucune suppression**. Les commandes ci-dessous sont des
  propositions.

## 3. Commandes de suppression PROPOSÉES (à n'exécuter qu'avec accord écrit)

> Toutes les commandes sont **proposées**. Elles sont destructrices : les
> spécifier, les relire, puis les exécuter **une par une** après validation de
> l'utilisateur. Chemin par chemin.

### 3.1 `/tmp` ( Gain ≈ 3,75 Go ) — sans risque pour le projet

```bash
# 3,74 Go de .npy de tau extraits manuellement (hors pipeline conforme)
rm -rf /tmp/tau_ref
# fixtures et caches de test
rm -rf /tmp/pytest-of-sjteb /tmp/a2d /tmp/dummy_dg /tmp/dummy_tau /tmp/dummy_out
rm -f  /tmp/measure.json
rm -rf /tmp/torchinductor_sjteb /tmp/dgsrc
# résumés de test écrits hors dépôt (jamais inventoriés jusqu'ici)
rm -f  /tmp/SUMMARY_test.json /tmp/SUMMARY_test.md
```

> **`/tmp/a2d_smoke_transfer/` : aucune commande nécessaire.** Ce checkpoint de
> transfert α=0,9 de la mission J6 (≈ 15 Go) est **déjà absent du disque** ;
> son gain est donc de **0 o** et il n'apparaît pas dans les commandes
> proposées. Le gain réel de `/tmp` n'augmente que d'environ **8,5 Mo** par
> rapport à l'inventaire initial (croissance de `/tmp/pytest-of-sjteb/` due aux
> relances de `pytest` de l'audit N1 + 51,7 Ko de `SUMMARY_test.*`).

### 3.2 Le harnais hors dépôt (Gain ≈ 7,07 Go ) — après accord

```bash
# Le harnais est reconstruit dans harness/dream/ (setup.sh) et rejoué le
# 2026-10-20 : la suppression n'entraîne aucune perte de méthode.
rm -rf /home/sjteb/dream-harness
```

> **Attention** : `harness/dream/.venv` est sur DrvFs et son import torch prend
> 15,8 s (contre 1,05 s en local). Si le harnais doit être relancé souvent,
> reconstruire le venv prend ~4 min (`bash harness/dream/setup.sh`).

### 3.3 Caches utilisateur (Gain ≈ 133 Go ) — hors périmètre du projet

```bash
# 86 Go : caches HuggingFace (modeles Qwen3.6/3.8, blobs, datasets).
# ATTENTION : ces poids ne servent pas aux portes A2D ; ils sont however
# peut-être réutilisables par d'autres travaux sur cette machine.
rm -rf /home/sjteb/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B-FP8
rm -rf /home/sjteb/.cache/huggingface/hub/models--nvidia--Qwen3.8-27B-NVFP4
rm -rf /home/sjteb/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4
rm -rf /home/sjteb/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2
rm -rf /home/sjteb/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B
# 37,6 Go : caches uv
rm -rf /home/sjteb/.cache/uv
# 8,3 Go : cache pip (le harnais utilise désormais harness/dream/.cache/pip)
rm -rf /home/sjteb/.cache/pip
# 3,4 Go : caches vLLM (campagne J4Bis)
rm -rf /home/sjteb/.cache/vllm/torch_compile_cache
# 15 Mo : données NLTK (re-téléchargeables à la demande)
rm -rf /home/sjteb/nltk_data
```

### 3.4 Docker (Gain ≈ 280 Go )

```bash
# ATTENTION : l'image vLLM 0.24.0 epinglee par digest sert a la porte J4Bis ;
# la conserver. Ne nettoyer que le cache de build et les images non utilisees.
docker builder prune -f                 # ~48 Go reclaimables
docker image prune -f --filter "until=720h"
# Le volume qdrant_qdrant_data (2,2 Go) n'a AUCUN rapport avec ce projet :
# ne pas le supprimer sans confirmation explicite de l'utilisateur.
# docker volume rm qdrant_qdrant_data
```

### 3.5 Ce qu'il ne faut PAS supprimer

| Chemin | Raison |
|---|---|
| `/home/sjteb/.cache/huggingface/datasets/` | datasets GSM8K / MATH / IFEval (re-téléchargeables, mais réseau lent) |
| `/home/sjteb/.cache/huggingface/modules/transformers_modules/dream_*` | code `trust_remote_code` des checkpoints Dream ; régénéré à la volée |
| `harness/dream/.venv`, `.fork`, `.cache` | environnement du harnais versionné (non versionné par git, mais **dans** le dépôt) |
| `checkpoints/` | les 4 checkpoints Dream/Qwen (60 Go) : nécessaires à J7 |

## 4. Gain total potentiel

| Catégorie | Gain |
|---|---:|
| `/tmp` | 3,75 Go |
| harnais hors dépôt | 7,07 Go |
| caches utilisateur | ~133 Go |
| Docker (hors volume Qdrant) | ~280 Go |
| **Total** | **~424 Go** |

> **Note** : `/tmp/a2d_smoke_transfer/` (≈ 15 Go) **n'est plus sur disque** ;
> le gain potentiel de `/tmp` est donc plus faible que ce qu'indiquait
> l'inventaire initial (seulement ~8,5 Mo de plus qu'alors, liés à la
> croissance de `/tmp/pytest-of-sjteb/` et aux `SUMMARY_test.*`). Le total
> ci-dessus reste ≈ 424 Go : les ~15 Go non récupérables ne sont pas
> comptabilisés.

Les volumes Docker ne sont pas comptés dans le gain de `/home/sjteb/.cache`
(puits) : ils sont mesurés séparément par `docker system df`.
