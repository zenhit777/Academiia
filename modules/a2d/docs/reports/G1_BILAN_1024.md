# G1 — addendum : diagnostic MATH à 1 024 tokens (J8, phase A2)

> **Document historique (J6–J8).** Les chemins cités dans ce rapport sont
> ceux de l'époque : la table de migration des chemins est
> [`docs/adr/path-migration.md`](../../../../docs/adr/path-migration.md).

**Statut : annulé sur coût mesuré. Aucun résultat, aucun échantillon produit.**
Le verdict G1 (`docs/reports/G1_BILAN.md` §1) et les mesures pré-enregistrées à 512 tokens
restent **inchangés**. Ce document consigne le protocole prévu, la mesure de débit qui a
motivé l'arrêt, et ce qu'il faudrait pour rendre le diagnostic exploitable.

## 1. Protocole prévu

Rappel de `docs/reports/G1_BILAN.md` §5, inchangé :

| Paramètre | Valeur |
|---|---|
| Tâche | `math_trunc_a2d` (67 items tronqués au budget 512, construits depuis M2) |
| Modèles | **M0** (`dream_7b_base`) et **M2** (`dream_transfer_a09`) — M1 et M3 non rejoués |
| Budget de génération | `max_new_tokens = 1024`, `diffusion_steps = 1024` |
| Lot | 4 (protocole de tous les runs G1) |
| Graines | `0,1234,1234,1234` |
| Métrique | `math_verify` |
| Sortie | `eval/dream_a2d_trunc/<modèle>/math_trunc_a2d/` |

L'objectif était de vérifier si le déficit de M2 sur les 67 items tronqués
(`M2 − M0 = −2,99 pts [−11,94 ; +5,97]` à 512 tokens) disparaît quand le raisonnement peut
se terminer. C'était une **analyse de sensibilité**, pas un second test de G1.

## 2. Déroulement mesuré

Lancé le **2026-10-21 à 13:48:24** (WSL2, RTX 5090, aucun autre modèle chargé) :

| Horodatage | État |
|---|---|
| 13:48:24 | démarrage du harnais |
| 13:50:41 | modèle chargé, début de `generate_until` |
| 14:11:36 | **4/67 items**, 20 min 55 s écoulées, `313,93 s/item`, reste estimé **5 h 29** |
| 14:32:30 | **toujours 4/67** — aucun item supplémentaire en ~21 min |
| 14:32:40 | arrêt du processus, GPU libéré (1,4 GiB résiduels, 0 % d'utilisation) |

Le débit du premier lot n'était donc pas un artefact de démarrage : le deuxième lot de 4 items
n'a pas terminé en 21 minutes. Le débit soutenu est de **~314 s/item**, soit :

| Projection | Durée |
|---|---|
| 67 items × 314 s | **5,84 h par modèle** |
| M0 + M2 | **11,68 h** |
| Avec les 43 min déjà consommées | **12,4 h** |

Le plafond GPU de la mission est de **5 h** : même un seul modèle le dépasse (1,17×), et les
deux modèles demandent 2,3× le plafond. Le diagnostic a été arrêté avant d'écrire quoi que ce
soit : lm-eval n'écrit `results_*.json` et `samples_*.jsonl` qu'en fin de tâche, le dossier de
sortie est donc **vide** et aucun résultat partiel n'est exploitable.

Écart avec l'estimation de préparation (**108 s/item**, soit 4,0 h pour deux modèles) :
**facteur 2,9**. L'hypothèse la plus probable est que le pilote ne fixait pas
`diffusion_steps = 1024` (le lanceur fixe `max_new_tokens` et `diffusion_steps` à la même
valeur) ; aucun journal de ce pilote n'a été conservé, l'hypothèse n'est donc **pas vérifiée**.
La mesure à 67 items fait foi.

## 3. Conséquences

* **Aucune analyse de sensibilité à 1 024 tokens** n'est disponible. La troncature reste
  l'explication **la plus plausible** du déficit de C1 sur MATH (82/175 générations épuisent
  512 tokens, dont 67 sans réponse finale), mais elle n'est **pas démontrée** par une
  contre-mesure à budget supérieur.
* **Le verdict G1 ne change pas** : C1 est défini sur MATH-175 à 512 tokens, avec les seuils
  pré-enregistrés. Un diagnostic annulé ne peut ni confirmer ni infirmer C1.
* **Le budget GPU de J8 reste sous le plafond** : 43 min consommées (plus ~5 min de pilote),
  contre 5 h autorisées. Le reliquat n'a pas été réaffecté à un sous-ensemble plus petit :
  avec 314 s/item, 20 items pour deux modèles coûteraient encore 3,5 h et donneraient un test
  apparié de puissance très faible (n = 20, quelques gains au plus), sans valeur probante.
* **M1 et M3 ne sont pas mesurés** à 1 024 tokens, comme prévu : le diagnostic ne peut pas
  comparer M2 à M1 à budget égal.

## 4. Ce qu'il faudrait pour rendre le diagnostic exploitable

1. **Budget dédié d'au moins 12 h** (M0 + M2, 67 items, protocole inchangé), ou **6 h** pour
   un seul modèle — mais un seul modèle ne répond pas à la question appariée.
2. **Plan réduit pré-enregistré** : n items tirés avant mesure, hypothèse et seuil de décision
   fixés d'avance, puissance calculée sur l'effet observé à 512 tokens (+17,6 pts sur le
   complément). Avec un taux de récupération attendu de 20 %, n ≈ 40–50 items seraient
   nécessaires pour un McNemar à 80 % de puissance ; le coût reste de l'ordre de 7–9 h.
3. **Matériel plus rapide** ou décodeur plus efficace : le coût vient de 1 024 pas de diffusion
   par item (un pas par token), pas du chargement. Aucune optimisation de type vLLM n'est
   disponible pour Dream-7B sur cette machine.
4. **Changer `diffusion_steps`** réduirait le coût mais **change le protocole** : à déclarer
   comme un nouveau protocole, pas comme une reproduction du run à 512 tokens.

## 5. Reproduire la mesure de coût

```bash
# 43 min, 4/67 items, puis arrêt manuel — le journal brut est versionné
MODEL_DIR=checkpoints/dream_7b_base,checkpoints/dream_transfer_a09 \
TASK=math_trunc LIMIT=67 BATCH=4 OUTDIR=eval/dream_a2d_trunc \
bash harness/dream/run_evals.sh
```

## 6. Pièces

| Fichier | Contenu |
|---|---|
| `eval/dream_a2d_trunc/ATTEMPT_1024.json` | horodatages, débit, projection, motif d'arrêt |
| `eval/dream_a2d_trunc/ATTEMPT_1024.log` | journal brut du harnais (séquences ANSI de tqdm incluses) |
| `docs/reports/G1_BILAN.md` §5 | protocole d'origine et analyse de troncature à 512 tokens |
| `eval/dream_a2d/TRUNCATION_512.json` | mesure de troncature (82/175 pour M2, 67 sans réponse) |
