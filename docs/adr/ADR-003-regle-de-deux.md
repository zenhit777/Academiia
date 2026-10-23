# ADR-003 — Règle de deux

- **Statut** : accepté (J9, 2026-10-22)
- **Décide par** : mission J9, phase 3
- **Complète** : ADR-001 (dépendance), ADR-002 (manifeste de lignée)

## Contexte

Le dépôt a produit deux implémentations de la même chose (fusion
« D_p + α·τ_A ») : la **bibliothèque A2D** (`modules/a2d/a2d/`) et le
**wrapper legacy Jainius** (`modules/a2d/legacy/scripts/merge_jainius.py`).
Le wrapper est aujourd'hui un proxy mince, mais la divergence a déjà produit
un incident : le build livré à α = 0,8 étiqueté 0,3 — attrapé par la
vérification post hoc, qui compare les coefficients *effectifs* aux
coefficients *demandés*.

Le symptôme n'est pas le double code : c'est l'**absence de règle** qui
autorise deux implémentations d'une même règle scientifique, donc deux
sources de vérité sur α, λ, τ et la façon de fusionner.

## Décision — la règle de deux

> **Un module ne duplique pas une règle d'un autre module.**
> Si un second module a besoin de la même règle, il **appelle** le premier.
> Une copie est admise **uniquement** si le premier module est *déprécié* et
> voué à disparaître, et **seulement** avec :
> 1. un marqueur explicite dans le fichier et dans son README
>    (« déprécié — voir `modules/a2d/a2d/` ») ;
> 2. un **test d'égalité** qui compare les deux implémentations sur des
>    entrées fixes et échoue si elles divergent ;
> 3. une **date de suppression** inscrite dans le README du module déprécié.

### Application au dépôt

| Règle | Implémentation canonique | Copie / wrapper | Statut |
|---|---|---|---|
| fusion D_b + α·τ_A, D_p + λ(τ_A − τ_D) | `modules/a2d/a2d/merge.py` | `modules/a2d/legacy/scripts/merge_jainius.py` (wrapper : délègue à la bibliothèque, garde l'interface historique et le plafond α ∈ [0,1] **côté legacy**) | déprécié, **test d'égalité** dans `modules/a2d/tests/test_pipeline.py` |
| calcul de τ_A persisté (`.npy`) | — (interdit dans le pipeline conforme) | `modules/a2d/legacy/scripts/compute_tau_a.py` | chemin legacy assumé : il persiste τ, le pipeline conforme **jamais** |
| statistiques / IC | `academiia/core/stats.py` | — | source unique |
| empreintes / E/S safetensors | `academiia/core/common.py` | — | source unique |
| verdict de porte | `modules/a2d/a2d/verdict.py` | — | source unique |

### Cas du module d-OPD (squelette J12)

* d-OPD **n'appelle pas** la bibliothèque A2D s'il applique une méthode
  différente : dans ce cas il ne duplique pas une *règle*, il implémente une
  *autre méthode*, et son manifeste doit le dire (`producer.module =
  "dopd"`, ADR-002).
* C'est le cas actuel : d-OPD est une distillation on-policy (cible
  future-aware), pas une fusion de poids. Le squelette `modules/dopd/`
  n'importe pas `academiia.a2d` et ne duplique aucune équation A2D.
* Si d-OPD devait réimplémenter une règle A2D (même équation, même convention
  de mapping), il **appelle** `academiia.a2d` et ne duplique pas.
* Le test d'égalité devient alors une **comparaison inter-modules** : le
  résultat de d-OPD et celui de A2D sur les mêmes entrées doivent coïncider
  à la tolérance BF16.

## Conséquences

* La règle est **opposable** : tout nouveau module passe cette question
  (« est-ce que je duplique ? ») avant d'écrire du code.
* Un wrapper qui **diverge** est un bug, pas une variante : le test d'égalité
  le signale ; on corrige le wrapper, pas le test.
* La règle de deux ne justifie **pas** de garder le legacy « au cas où » : le
  legacy a une date de suppression (J8 phase C l'a déjà.reduce à un wrapper ;
  la suppression reste ouverte).
* Le dépôt n'a **pas** d'exception « troisième copie » : toute occurrence
  supplémentaire est à justifier dans un ADR.
