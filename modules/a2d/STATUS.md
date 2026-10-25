# État du projet Academiia — 2026-10-21 (fin de la mission J8)

> **Mise à jour J9 (2026-10-22)** : réorganisation du dépôt en **pipeline
> modulaire** (module A2D sous `modules/a2d/`, code générique sous
> `academiia/core/`, paquets importables `academiia.core` / `academiia.a2d`).
> **Aucun résultat, aucun seuil et aucun verdict n'ont été modifiés** ; le
> verdict G1 recalculé est **strictement identique** à celui stocké. Voir
> `docs/ARCHITECTURE.md`, `docs/adr/path-migration.md` et
> `modules/a2d/docs/reports/J9.md`.
>
> **Mise à jour J10 (2026-10-22)** : nettoyage du workspace (harnais reconstruit
> sous `modules/a2d/harness/dream/`, ancien dossier racine `harness/` supprimé),
> README racine allégé (détail déplacé dans `modules/a2d/docs/QUICKSTART.md`),
> entry-points CLI (`merge-a2d`, `verify-merge`, `measure-geometry`,
> `verify-checkpoint`, `download-models`). **Aucun résultat, aucun seuil et aucun
> verdict n'ont été modifiés.** Voir `modules/a2d/docs/reports/J10.md`.
>
> **Mise à jour J12 (2026-10-23)** : le module **d-OPD** (arXiv 2609.35362) a
> désormais un **squelette** sous `modules/dopd/` (paquet `academiia.dopd`,
> cible future-aware NumPy/CPU, tests). Il n'importe pas A2D et **ne touche à
> aucun résultat, seuil ou verdict G1** ; voir `modules/dopd/docs/DOPD_SPEC.md`
> et `modules/dopd/docs/reports/J12.md`.
>
> **Mise à jour J13 (2026-10-23)** : les **contrats d'interface** de d-OPD sont
> formalisés (professeur, rollout, vérificateur, lignée ADR-002) et un **smoke
> de chargement Qwen3-0.6B** a réussi (CPU et GPU, **aucun entraînement**).
> d-OPD reste **indépendant** d'A2D : il n'importe pas `academiia.a2d` et
> **aucun résultat, seuil ou verdict G1 n'a été modifié**. Voir
> `modules/dopd/docs/reports/J13.md`.
>
> **Mise à jour J14 (2026-10-23)** : la **porte scientifique Annexe D** de
> d-OPD est **franchie, avec réserves de lecture** sur Qwen3-0.6B : la cible future-aware à un seul futur
> échantillonné réduit la KL appariée de **0,1623** (IC 95 % bootstrap
> `[0,0742 ; 0,2252]`), sur 3 056 marginaux et 3 configurations. Run GPU
> **déclaré**, inférence seule : **aucun entraînement, aucun backward, aucun
> poids modifié**, et **aucun résultat, seuil ou verdict G1 touché**. La voie
> d-'entraînement locale est **chiffrée sur mesures** : échelle **0,6B**
> faisable en **16–21 h** sur une RTX 5090 (scoring professeur mesuré à
> 51,2 ms par job avec cache de préfixe) ; l'échelle du papier (4B–8B) reste
> **hors de portée** d'une carte unique. Voir
> `modules/dopd/docs/BUDGET_RTX5090.md`, `modules/dopd/docs/reports/J14.md`.
>
> **Mise à jour J15 (2026-10-25)** : la **porte scientifique Annexe D** de d-OPD
> est renforcée par un **diagnostic multi-prompt** (20 prompts variés, 61 120
> marginaux) et une **mesure de l'écart cible exacte vs top-k = 16**. L'effet
> future-aware reste réel et significatif (`ΔKL = +0,0802`, IC 95 % `[0,064 ;
> 0,096]`) mais **moyenne-porté-par-queue** (médiane nulle : 42,8 % de marginaux
> sans futur visible) : la porte est désormais qualifiée **« franchie, avec
> réserves de lecture »**, décision **GO conditionnel** pour un `TeacherScorer`
> réel. Troncature top-k fidèle en moyenne (TV 1,48 %) mais hétérogène (masse
> perdue max 37,3 %). Runs GPU **déclarés**, inférence seule : **aucun
> entraînement, aucun backward, aucun poids modifié**, et **aucun résultat, seuil
> ou verdict G1 touché**. Résidus J14 corrigés (formulation, règle 12,
> `importorskip`, CI citée sur le HEAD d'arrivée). Voir
> `modules/dopd/docs/reports/J15.md`.

Ce document est le **point d'entrée** pour savoir ce qui est démontré, ce qui ne l'est pas, et
où se trouve chaque chose. Il remplace les statuts dispersés dans les rapports de mission ;
les rapports eux-mêmes restent la source des chiffres.

## 1. Les portes, en une page

| Porte | Question | État | Preuve |
|---|---|---|---|
| **G1** | L'A2D-Transfer / A2D-Merge améliorent-ils Dream-7B avec un vecteur Qwen2.5-7B ? | **NON FRANCHIE — porte ouverte.** C2 atteint (GSM8K **+14,33** pts, IC [+9,33 ; +19,67]) ; C1 non atteint (MATH-175 **+9,71** pts, IC [+2,86 ; +16,57] — **0,29 pt** sous +10) ; C3/C4 non établis (merge indiscernable de D_p). | `modules/a2d/docs/reports/J7.md`, `modules/a2d/docs/reports/G1_BILAN.md`, `modules/a2d/eval/dream_a2d/G1_VERDICT.json` |
| **G2** | Reproduire le setup médical DiffusionGemma du papier | **Débloquée côté bibliothèque, NON évaluée.** Le mode `merge-dg-scaled` existe et est testé ; l'évaluation exige un GPU où le décodeur DG 26B et le juge 27B tiennent ensemble. | `modules/a2d/docs/design/G2_adapter_scaling.md`, `modules/a2d/a2d/adapters.py`, §5 |
| **G1b** | Faisabilité Nemotron-Labs-Diffusion-3B | **Non démarrée** (étude en lecture seule). | `modules/a2d/docs/design/G1B_NEMOTRON3B.md` |
| **G3** | Raisonnement (hors papier) | **Non démarrée** (conception d'abord). | — |
| **J5** | Campagne d'évaluation 103 prompts (vLLM FP8) | **Non conforme** au protocole du papier — contrôle de non-régression seulement. | `modules/a2d/legacy/eval/RESULTS.md` |

**Lecture du verdict G1.** Le protocole pré-enregistré (`modules/a2d/docs/design/G1_PROTOCOL.md` §4.1,
amendement M1) n'a **pas** été modifié après les résultats. L'échec de C1 pour 0,29 point est
documenté comme un **problème de résolution et de budget de génération** : M2 épuise le plafond
de 512 tokens sur 82/175 générations MATH, et sur les 67 items qu'il ne peut pas terminer il
n'est pas meilleur que la base, alors que sur les 108 autres il gagne **+17,59** points. Cela
**n'annule pas** le verdict : C1 est défini sur MATH-175 à 512 tokens. Le diagnostic à
1 024 tokens qui devait tester directement cette hypothèse a été **annulé sur coût mesuré**
(314 s/item ⇒ 5,8 h par modèle, au-delà du plafond de 5 h) : la troncature reste l'explication
la plus plausible, mais elle n'est **pas démontrée** (`modules/a2d/docs/reports/G1_BILAN_1024.md`).

## 2. Ce qui est validé mécaniquement

* **Bibliothèque A2D** (`modules/a2d/a2d/`, CLI `modules/a2d/cli/merge_a2d.py`) : τ recalculé à la volée
  (jamais persisté), arithmétique FP32 + un seul cast BF16, écriture atomique, **vérification
  post-écriture v2 obligatoire** avec `max_ulp = 0` par défaut (1 ULP = échec), manifeste
  complet (coefficients demandés/effectifs signés, hashes d'index et empreintes de contenu,
  couverture, garde-fou coefficient nul, versions).
* **Chargeabilité** : les shards fusionnés portent le bloc `__metadata__` (`format: pt`) repris
  de la source ; un test `from_pretrained` sur un modèle-jouet fait partie de la suite.
* **Preuves G1** : M2 (`α = 0,9`) et M3 (`λ = 0,6`) vérifiés (0 ULP, `verify_merge` OK) puis
  évalués sur les mêmes items (empreintes identiques sur 4 modèles, lot 4) ; plancher de bruit
  de décodage **nul** au lot 4 sur GSM8K.
* **G2 (bibliothèque)** : mode `merge-dg-scaled` / `adapter-scale` — `base' = D_b + λ·τ_A`,
  adaptateur τ_D **non fusionné**, `lora_B × (1−λ)`, `lora_alpha` inchangé, vérification V5/V6
  (L2 relative ≤ 1e-2 BF16 / 1e-3 FP32, cosinus ≥ 1 − 1e-4), manifeste `diffusion_adapter`.
* **Suite de tests** : `python -m pytest modules/a2d/tests tests infra/vllm/scripts/tests -q`
  (la suite pytest du dépôt ; le nombre de tests n'est **pas** figé dans la prose). Le code
  générique a ses propres tests : `tests/test_core.py`.

## 3. Ce qui n'est PAS démontré

* **C1, C3, C4** (voir §1) ; aucune **équivalence** avec la Table 16 du papier (MATH-175 ≠
  MATH-500 ; un IC qui contient une cible n'est pas une preuve d'équivalence).
* **Aucune mesure à 1 024 tokens** : le diagnostic a été annulé sur coût mesuré. L'effet de la
  troncature sur MATH n'est donc **pas isolé expérimentalement**.
* **Aucune évaluation DiffusionGemma** (G2) : ni le mode `merge-dg-scaled` en conditions
  réelles, ni les adaptateurs publiés, ni le juge.
* **IFEval complet** (541 prompts) : hors budget ; seul un contrôle n = 8 existe.
* **Généralisation** à une autre paire AR, à un autre coefficient α/λ, ou à Nemotron (G1b).
* **Le plancher de bruit de décodage** n'est mesuré que sur GSM8K/256 tokens avec M1 ; pas sur
  MATH/512 ni à 1 024 tokens.

## 4. Contraintes de la machine

* GPU : **RTX 5090, 32 607 MiB**, WSL2. Pas de GPU > 32 Go.
* DiffusionGemma : décodeur 26B en BF16 ≈ 48 GiB **plus** le juge 27B ⇒ **impossible sans
  quantification** sur cette machine. Toute évaluation DG devra **déclarer** la quantification
  et re-mesurer la baseline D_b dans les mêmes conditions.
* Budget GPU de J8 : **≤ 5 h**. Utilisé : **43 min** — diagnostic MATH 1 024 tokens lancé puis
  **annulé** après 4/67 items (314 s/item ⇒ 5,8 h/modèle projetées), plus ~5 min de pilote.
  Aucun résultat n'a été écrit : `modules/a2d/eval/dream_a2d_trunc/ATTEMPT_1024.{json,log}`.

## 5. Où est quoi

| Chemin | Contenu |
|---|---|
| `modules/a2d/a2d/` | bibliothèque : `naming.py`, `sources.py`, `adapters.py`, `merge.py` |
| `modules/a2d/cli/merge_a2d.py` | CLI conforme (transfert, merge, DG, ablations) |
| `modules/a2d/cli/verify_merge.py` | vérification post hoc (coefficients, ULP, adaptateur) |
| `modules/a2d/cli/measure_geometry.py` | géométrie des vecteurs (Table 21) |
| `modules/a2d/harness/dream/` | harnais Dream-7B (setup, tâches, `run_evals.sh`, synthèse) |
| `modules/a2d/harness/dream/tasks/math_trunc_a2d.*` | sous-ensemble des 67 items tronqués (diagnostic) |
| `modules/a2d/eval/dream_a2d/` | résultats G1 versionnés + verdict + empreintes + troncature |
| `modules/a2d/eval/dream_a2d_trunc/` | tentative de diagnostic 1 024 tokens (annulée : débit mesuré) |
| `modules/a2d/docs/design/` | protocoles et conceptions (G1, G2, budget d'évaluation) |
| `modules/a2d/docs/QUICKSTART.md` | détail opérationnel déplacé depuis le README racine (J10) : garanties de la bibliothèque, protocole App. A.1 et tableaux de scores, géométrie, workflow legacy, scripts vLLM |
| `modules/a2d/docs/reports/` | rapports de mission (J6, J6bis, J7, **G1_BILAN**, audits, **G1_BILAN_1024**) |
| `modules/a2d/legacy/scripts/`, `modules/a2d/legacy/docs/` | pipeline Jainius **déprécié** et archives (hors papier) |
| `checkpoints/` | poids binaires, **non versionnés** |

## 6. Reproduire les vérifications

```bash
# tests (DrvFs : rediriger TMPDIR dans le depot et garder la capture par defaut)
TMPDIR="$PWD/_artefact/tmp" python -m pytest modules/a2d/tests tests infra/vllm/scripts/tests -q --capture=sys   # tout vert (DrvFs : --capture=sys requis)

# verdict G1 (recalcule a partir des echantillons versionnes)
python modules/a2d/eval/g1_verdict.py            # ecrit modules/a2d/eval/dream_a2d/G1_VERDICT.json

# troncature MATH et sous-ensemble de diagnostic (CPU)
python modules/a2d/eval/measure_math_truncation.py \
    --samples-root modules/a2d/eval/dream_a2d --tokenizer checkpoints/dream_7b_instruct \
    --budget 512 --output modules/a2d/eval/dream_a2d/TRUNCATION_512.json

# mode G2 (exemple ; exige les adaptateurs publies et un GPU adapte pour l'evaluation)
python modules/a2d/cli/merge_a2d.py --mode merge-dg-scaled \
    --diffusion checkpoints/diffusiongemma_it \
    --tau-a-post checkpoints/gemma4_it --tau-a-base checkpoints/gemma4_base \
    --tau-d-adapter checkpoints/dg_adapters/diffusion-vqarad \
    --lambda 0.3 --mapping gemma4-diffusiongemma \
    --output checkpoints/diffusiongemma_merge_l03
```

## 7. Recommandation DiffusionGemma (quand un GPU adapté sera disponible)

1. **Matériel** : viser **≥ 80 Go** (A100/H100 80 Go) pour tenir le décodeur 26B en BF16 et le
   juge 27B **sans quantification**, ou assumer une quantification et la déclarer.
2. **Ne pas réutiliser `checkpoints/Jainius`** : c'est une ablation hors papier (τ_A général,
   α non sélectionné, aucun gain mesuré).
3. **Respecter le rôle des vecteurs du papier** : pour DiffusionGemma, `D_b` = checkpoint **IT**
   (pas de base diffusion public), `τ_A` = delta **de domaine** (LoRA IT → médical) et `τ_D` =
   delta LoRA de diffusion. Le mode `merge-dg-scaled` accepte des sources LoRA
   (`--tau-a-adapter`, `--tau-d-adapter`).
4. **Vérifier la disponibilité des adaptateurs** dans `modules/a2d/docs/VP_ADAPTERS.md` avant de figer λ :
   λ vient de la Table 6/7 du papier, pas d'un balayage sur cette machine.
5. **Protocole d'évaluation** : VQA-RAD / SLAKE / VQA-Med avec juge LLM, tel que App. A.1 ;
   baseline `D_b` et `D_b + λ·τ_A` mesurées dans les mêmes conditions. Sans juge, on ne
   reproduit pas le chiffre du papier — on produit un contrôle de non-régression.
6. **Ordre** : d'abord un smoke `merge-dg-scaled` (vérification V5/V6, chargement
   `from_pretrained` + adaptateur), puis l'évaluation.

## 8. Décisions ouvertes

* **Rouvrir C1** : soit augmenter `n` sur MATH (coût ~48 h/modèle), soit pré-enregistrer un
  **TOST** sur une marge d'acceptabilité, soit pré-enregistrer un protocole **1 024 tokens**
  pour les quatre modèles (change le protocole du papier, à déclarer).
* **G1b** : étude de faisabilité Nemotron-3B (lecture seule) avant tout téléchargement.
* **G3** : conception écrite avant code (hors papier).
* **G2** : dépend d'un accès GPU ; aucune décision scientifique ne peut être prise sur 32 Go
  sans quantification déclarée.
