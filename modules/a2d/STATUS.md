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
> future-aware reste réel et significatif (`ΔKL = +0,0802`, IC 95 % `[0,0638 ;
> 0,0962]`) mais **moyenne-porté-par-queue** (médiane nulle : 42,788 % de marginaux
> sans futur visible) et avec une queue nuisible (13,30 % des marginaux à futur
> visible dégradés) : la porte est qualifiée **« franchie, avec réserves de
> lecture »**, décision **GO conditionnel** pour un `TeacherScorer` réel. Réduction
> relative **8,58 %** (le chiffre 16,12 % est celui de J14). Troncature top-k
> fidèle en moyenne (TV corrigée 2,9534 %) mais hétérogène (masse perdue max
> 37,3190 %). Runs GPU **déclarés**, inférence seule : **aucun entraînement,
> aucun backward, aucun poids modifié**, et **aucun résultat, seuil ou verdict G1
> touché**. Voir `modules/dopd/docs/reports/J15.md`.
>
> **Mise à jour J16 (2026-10-25)** : l'audit `AUDIT_J15.md` a relevé cinq écarts
> de communication/métrique ; ils sont corrigés dans `J15.md`, `DOPD_SPEC.md`
> §6.9.9 et `derived_summary.json`. **Aucun résultat, seuil ni verdict G1 n'a été
> modifié** : la porte Annexe D reste franchie avec réserves de lecture, décision
> **GO conditionnel** confirmée (critère primaire atteint, médiane = 0). J16 ne
> met en œuvre **ni `TeacherScorer` réel, ni entraînement**, et ne pousse rien.
> Voir `modules/dopd/docs/reports/J16.md`.
>
> **Mise à jour J17 (2026-10-25)** : le **`TeacherScorer` réel** de d-OPD est
> livré (Qwen3-0.6B, BF16), **isolé sous `modules/dopd/harness/`** : le paquet
> `academiia.dopd` reste **pur** (aucun import `torch`/`transformers`). Le
> cache de préfixe de l'App. B est **exact dans la tolérance pré-enregistrée
> (C3)** : le même job rejoué en FP32 donne un écart de **1,1444e-05** (cellule
> de référence), **6,1989e-05** (état réel, GPU) et **2,7e-05** (run CPU FP32),
> contre **0,395069** en BF16 — écart imputé à la précision BF16 du protocole
> de comparaison, **pas** à la mécanique du cache. Latence cache : médiane
> **52,75 ms** sur les 12 jobs à futur visible (géométrie homogène, seule
> comparable au rejeu), **53,905 ms** à 16 branches et **68,187 ms** à 33 en
> cellule de référence (J14 : 51,2 / 65,6 ms) ; le passage en FP32 coûte
> **+0,68 ms** de latence (**+1,3 %**) et **+1 161 MiB** de pic mémoire
> (mesure complémentaire non versionnée). Sur les 14 critères pré-enregistrés,
> **13 sont tenus** ; seul **C2** échoue sur les états réels BF16 (seuil 0,35,
> max 0,395069), sans atteindre les conditions de NO-GO ; C7 est tenu avec une
> **réserve de géométrie** (16 branches réelles, 17 au protocole). Runs GPU
> **déclarés**, inférence seule : **aucun entraînement, aucun backward, aucun
> poids modifié**, et **aucun résultat, seuil ou verdict G1 touché**. Une
> **revue N2 indépendante** a rendu le verdict « Incorrect » (3 majeurs,
> 4 mineurs) : **les 7 constats ont été vérifiés puis corrigés** — contrainte
> `transformers>=4.56.2` avec garde-fou explicite, empreintes **de fichier**
> du checkpoint dans `parent.hashes` (ADR-002), tests rendus sensibles au
> préfixe/aux positions/masque avec **contrôle sur un vrai Qwen3**, échec dur
> hors dépôt, révision du snapshot et positions effectives consignées. La
> sensibilité de la suite a été prouvée par **injection de fautes** (6, 4 et 3
> tests échouent selon la faute : masque, positions, préfixe). Une **seconde
> revue indépendante** a confirmé 6/7 constats corrigés et signalé un écart
> résiduel sur F3 (le fake enregistrait le masque sans l'utiliser) : corrigé —
> les tokens masqués sont exclus de l'agrégat du fake, propriété **verrouillée
> par un test** (écart des **scores** de suffixe, masque plein vs nul :
> 0,0 → 0,377). Décision J18 : **GO conditionnel** — réserve
> explicite : §6.10.7 subordonne ce GO à « C1–C14 tenues », clause non
> littéralement remplie ; la décision est une **interprétation** (un seul
> critère échoué, hors liste NO-GO) et **J18 doit la trancher
> explicitement**. Conditions : professeur en FP32, protocole d'entraînement
> pré-enregistré distinct, référence de comparaison en rejeu intégral FP32,
> prise en compte du coût du cache aux préfixes courts.
> **Réserve de gouvernance** : un push non autorisé a porté `origin/main` de
> `08d62bd` à `50464eb` (2026-10-25 14:10:29 +0100) ; l'attribution est
> indéterminée (aucun appel `git push` dans les transcripts OMP, aucun hook),
> le contenu poussé est sans secret, et les 8 commits suivants sont **locaux**.
> Mise à jour du distant : **accord écrit requis**. Voir
> `modules/dopd/docs/reports/J17.md` §6.3 et §8.1.
> **R1 (2026-10-26) — Décision architecte sur C2 / §6.10.7.** Le professeur est en **FP32** comme précision de production, et le rejeu intégral FP32 est la référence d'exactitude. **On autorise la poursuite** vers J18. La règle §6.10.7 est amendée pour l'avenir : tout critère hors liste NO-GO qui échoue impose une **décision explicite écrite** (ADR ou STATUS) ; le silence n'est pas un GO. Cette décision est **propre à J17** et ne préjuge pas des critères de J18.
>
> **Audit J18 (2026-10-26, corrigé le 2026-10-27)** : le journal
> `modules/dopd/docs/reports/AUDIT_J18.md` **ne certifie pas** la décision GO de
> J18. L'audit initial (2026-10-26) avait certifié le GO au sens des critères
> pré-enregistrés ; une revue d'opposition indépendante, puis un contre-test
> reproduit sur le vrai Qwen3-0.6B, ont établi un **écart critique** : dans la
> géométrie J18 (un seul masque, toutes les autres positions du bloc visibles),
> la cible future-aware **coïncide avec la posterior exacte** (4,5e-10 près),
> donc `future_kl ≡ 0` et `ΔKL ≡ KL(exact‖causal) ≥ 0` **par construction**.
> Les critères C1–C4 sont des **identités**, non des observations ; l'amplitude
> ΔKL ≈ 3,05 mesure le désaccord exact/causal sur un bloc atypique (tirage
> uniforme au lieu de pondéré par `log_joint`), pas un gain de la correction
> future-aware. **Le GO de J18 n'est pas reportable sur J19** : la mesure doit
> être reconstruite (futur échantillonné séparément, convention J15) puis
> re-lancer. S'y ajoutent **onze écarts** de traçabilité et de communication
> (référence d'exactitude déclarée « rejeu intégral FP32 » alors que le chemin
> exécuté est celui du cache de préfixe ; C4 mesuré sur les marginaux token ;
> `producer.command` non rejouable ; « sans réserve » et « 120 blocs/config »
> non étayés ; `mean_causal_kl`/`mean_future_kl` absents du manifeste
> (correctif **différé à J19**, `diagnostic.py` étant hors périmètre et protégé
> par C10) ; `artifact.hashes` vide et C7 infalsifiable ; `correctness_gate`
> déclaré actif sans vérificateur ; `arithmetic_dtype` faux (float64) ;
> suite de tests ne pinne ni la construction future-aware ni l'unité de
> bootstrap (prouvé par injection de fautes) ; horloges de mesure mélangées).
> **Aucun résultat, seuil ni verdict G1 n'est modifié** ; le pré-enregistrement
> J18 reste intact. Voir `modules/dopd/docs/reports/AUDIT_J18.md`.

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
