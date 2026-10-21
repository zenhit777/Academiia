#!/usr/bin/env python3
"""Verdict de la porte G1 — application mécanique des critères amendés (J7, Phase 4).

La porte G1 est **franchie** si, et seulement si (`docs/design/G1_PROTOCOL.md` §1) :

1. les quatre modèles (D_b, D_p, Transfer α=0,9, Merge λ=0,6) sont évalués sur
   les **mêmes items** (preuve par empreinte) et avec le **même protocole** ;
2. chaque fusion a passé la vérification v2 (`post_write_verification.ok` et
   `verify_merge` ⇒ VERDICT OK) ;
3. les **critères d'écart** de la §4 sont atteints ;
4. les **IC appariés** sont rapportés.

Ce script mechanise les points 1, 3 et 4 :

* contrôle d'identité des items (bloquant) : `items_sha256` et `prompts_sha256`
  identiques pour les quatre modèles, sur chaque tâche ;
* écarts appariés + **IC bootstrap apparié 95 %** (même algorithme et même
  graine que `scripts/paired_bootstrap.py`) ;
* formulation **amendée** (M1) :
  - **C1** (Transfer − Base, MATH-175) : point estimate ≥ +10 **et** borne
    basse de l'IC > 0 ;
  - **C2** (Transfer − Base, GSM8K) : point estimate ≥ +8 **et** borne basse
    de l'IC > 0 ;
  - **C3** (Merge − Instruct, MATH-175) et **C4** (Merge − Instruct, GSM8K) :
    « non-régression non réfutée » si borne basse ≥ −2 ; « non-régression
    établie » seulement si borne basse ≥ 0 ; « régression » en dessous.

Usage (le métrique `math_verify` impose le venv du harnais) :

    harness/dream/.venv/bin/python scripts/g1_verdict.py --root eval/dream_a2d

Code de sortie : **0** = porte franchie, **1** = porte **non** franchie (verdict négatif,
calcul sans erreur), **2** = erreur de contrôle (items différents, run manquant).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_item_fingerprint import fingerprint  # noqa: E402
from paired_bootstrap import bootstrap_ci95, per_item, wilson  # noqa: E402

#: (critère, modèle A, modèle B, dossier de tâche, métrique, filtre, seuil)
CRITERIA = (
    ("C1", "M0", "M2", "minerva_math_algebra", "math_verify", "none", 10.0,
     "Transfer - Base, MATH-175"),
    ("C2", "M0", "M2", "gsm8k_cot_a2d", "exact_match", "flexible-extract", 8.0,
     "Transfer - Base, GSM8K"),
    ("C3", "M1", "M3", "minerva_math_algebra", "math_verify", "none", -2.0,
     "Merge - Instruct, MATH-175"),
    ("C4", "M1", "M3", "gsm8k_cot_a2d", "exact_match", "flexible-extract", -2.0,
     "Merge - Instruct, GSM8K"),
)

MODELS = {
    "M0": "dream_7b_base",
    "M1": "dream_7b_instruct",
    "M2": "dream_transfer_a09",
    "M3": "dream_merge_l06",
}


def compare(run_a: Path, run_b: Path, metric: str, filter_name: str,
            iters: int, seed: int):
    """Écart apparié + IC bootstrap 95 % entre deux runs (mêmes items)."""
    a = per_item(run_a, metric, filter_name)
    b = per_item(run_b, metric, filter_name)
    if set(a) != set(b):
        raise SystemExit(f"ERREUR: jeux d'items differents entre {run_a} et {run_b}")
    keys = sorted(a)
    n = len(keys)
    diffs = [b[k] - a[k] for k in keys]
    lo, hi = bootstrap_ci95(diffs, iters=iters, seed=seed)
    hits_a = sum(a[k] for k in keys)
    hits_b = sum(b[k] for k in keys)
    return {
        "n_items": n,
        "acc_a": hits_a / n,
        "acc_b": hits_b / n,
        "difference": (hits_b - hits_a) / n,
        "ci95": [lo, hi],
        "discordant_b_only": sum(1 for k in keys if b[k] == 1 and a[k] == 0),
        "discordant_a_only": sum(1 for k in keys if a[k] == 1 and b[k] == 0),
        "wilson_a": wilson(hits_a, n),
        "wilson_b": wilson(hits_b, n),
    }


def verdict_gain(criterion, threshold: float, diff: float, lo: float, hi: float):
    """C1/C2 : point estimate >= seuil ET borne basse > 0."""
    ok_point = diff >= threshold
    ok_ci = lo > 0.0
    if ok_point and ok_ci:
        state = "ATTEINT"
    elif not ok_point and ok_ci:
        state = "NON ATTEINT (point estimate sous le seuil, IC exclut 0)"
    elif ok_point and not ok_ci:
        state = "NON ATTEINT (borne basse <= 0)"
    else:
        state = "NON ATTEINT (point estimate sous le seuil et IC contient 0)"
    return {
        "criterion": criterion,
        "type": "gain",
        "seuil": threshold,
        "point_estimate": diff,
        "borne_basse": lo,
        "borne_haute": hi,
        "condition_point_estimate": ok_point,
        "condition_borne_basse": ok_ci,
        "etat": state,
        "atteint": ok_point and ok_ci,
    }


def verdict_non_regression(criterion, threshold: float, diff: float, lo: float, hi: float):
    """C3/C4 : non-regression « non refutee » / « etablie » / « regression »."""
    if lo < threshold:
        state = "REGRESSION (borne basse < -2)"
        atteint = False
    elif lo >= 0.0:
        state = "non-regression ETABLIE (borne basse >= 0)"
        atteint = True
    else:
        state = "non-regression NON REFUTEE (borne basse >= -2)"
        atteint = True
    return {
        "criterion": criterion,
        "type": "non-regression",
        "seuil": threshold,
        "point_estimate": diff,
        "borne_basse": lo,
        "borne_haute": hi,
        "etat": state,
        "atteint": atteint,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="eval/dream_a2d",
                        help="racine des resultats (<root>/<modele>/<tache>)")
    parser.add_argument("--iters", type=int, default=10000,
                        help="nombre de re-echantillonnages bootstrap")
    parser.add_argument("--seed", type=int, default=0, help="graine du bootstrap")
    parser.add_argument("--json", action="store_true", help="sortie JSON")
    args = parser.parse_args()

    root = Path(args.root)
    # --- point 1 : identite des items sur les quatre modeles ---------------
    identity = {}
    for task in ("gsm8k_cot_a2d", "minerva_math_algebra", "ifeval_a2d"):
        prints = {}
        for label, alias in MODELS.items():
            run_dir = root / alias / task
            if not run_dir.is_dir():
                print(f"ERREUR: run manquant : {run_dir}")
                return 2
            fp = fingerprint(run_dir)
            if "error" in fp:
                print(f"ERREUR: {run_dir} : {fp['error']}")
                return 2
            prints[label] = (fp["items_sha256"], fp["prompts_sha256"], fp["n_items"])
        same_items = len({v[0] for v in prints.values()}) == 1
        same_prompts = len({v[1] for v in prints.values()}) == 1
        identity[task] = {
            "items_sha256": prints["M0"][0],
            "prompts_sha256": prints["M0"][1],
            "n_items": prints["M0"][2],
            "memes_items_4_modeles": same_items,
            "memes_prompts_4_modeles": same_prompts,
        }
        if not (same_items and same_prompts):
            print(f"ERREUR: items/prompts differents sur {task} : {prints}")
            return 2

    # --- points 3 et 4 : ecarts apparies, IC et verdicts ------------------
    verdicts = []
    for (criterion, model_a, model_b, task, metric, filter_name, threshold,
         label) in CRITERIA:
        run_a = root / MODELS[model_a] / task
        run_b = root / MODELS[model_b] / task
        res = compare(run_a, run_b, metric, filter_name, args.iters, args.seed)
        if threshold > 0:
            v = verdict_gain(criterion, threshold, res["difference"],
                             res["ci95"][0], res["ci95"][1])
        else:
            v = verdict_non_regression(criterion, threshold, res["difference"],
                                       res["ci95"][0], res["ci95"][1])
        v.update({
            "modele_a": model_a, "modele_b": model_b, "tache": task,
            "metrique": metric, "filtre": filter_name, "libelle": label,
            "precision_a": res["acc_a"], "precision_b": res["acc_b"],
            "n_items": res["n_items"],
            "discordant_b_only": res["discordant_b_only"],
            "discordant_a_only": res["discordant_a_only"],
            "wilson_a": res["wilson_a"], "wilson_b": res["wilson_b"],
        })
        verdicts.append(v)

    all_met = all(v["atteint"] for v in verdicts)
    report = {
        "root": str(root),
        "identite_items": identity,
        "criteres": verdicts,
        "tous_les_criteres_atteints": all_met,
        "conclusion": ("G1 FRANCHIE" if all_met else
                       "G1 NON FRANCHIE (porte ouverte)"),
        "bootstrap_iters": args.iters,
        "bootstrap_seed": args.seed,
    }
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print("=== Identite des items (bloquant) ===")
    for task, info in identity.items():
        print(f"  {task:24s} n={info['n_items']:3d} items={info['items_sha256'][:16]} "
              f"prompts={info['prompts_sha256'][:16]} "
              f"4 modeles identiques : {info['memes_items_4_modeles'] and info['memes_prompts_4_modeles']}")
    print("\n=== Criteres (formulation amendee) ===")
    for v in verdicts:
        print(f"  {v['criterion']} — {v['libelle']}")
        print(f"      {v['modele_a']} {100 * v['precision_a']:.2f} %  ->  "
              f"{v['modele_b']} {100 * v['precision_b']:.2f} %  (n={v['n_items']})")
        print(f"      ecart = {100 * v['point_estimate']:+.2f} pts   "
              f"IC 95 % [{100 * v['borne_basse']:+.2f} ; {100 * v['borne_haute']:+.2f}]")
        print(f"      -> {v['etat']}")
    print(f"\n=== Conclusion : {report['conclusion']} ===")
    return 0 if all_met else 1


if __name__ == "__main__":
    raise SystemExit(main())
