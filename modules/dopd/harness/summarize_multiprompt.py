#!/usr/bin/env python3
"""Dérive le résumé J15 vérifiable depuis ``results.json`` (J16, CPU seulement).

Le fichier ``results.json`` de J15 reste **immuable** : il a été produit par le
harnais au commit ``fe499ff`` et son empreinte est citée. Deux champs y ont
cependant été ajoutés après coup par un script non versionné
(``_artefact/j15_stamp_status.py``) : ``multiprompt.gate_j15_status`` et
``multiprompt.criteria``. Ce script remplace cette dérivation ad hoc par du code
versionné et **n'écrit jamais dans ``results.json``** : il produit un artefact
dérivé distinct, reproductible par ``--check``.

Il publie aussi les corrections factuelles de J16 :

* ventilation des prompts par **signe** et par significativité (le code J15 ne
  distinguait pas « IC exclut zéro » de « effet positif ») ;
* réduction relative multi-prompt réelle (8,58 %, et non 16,12 % de J14) ;
* TV corrigée contre la troncature **renormalisée** (= masse discardée) ;
* décision de porte recalculée par ``decide_multiprompt_gate``.

Aucun GPU, aucun torch, aucun modèle : uniquement NumPy et le JSON versionné.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np

from academiia.dopd.diagnostic import (
    decide_multiprompt_gate,
    prompt_sign_breakdown,
    sign_test,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = (
    "modules/dopd/eval/annexe_d_qwen3_0.6b_multiprompt/results.json"
)
DEFAULT_OUTPUT = (
    "modules/dopd/eval/annexe_d_qwen3_0.6b_multiprompt/derived_summary.json"
)
SCHEMA = "dopd.multiprompt_derived/1"
SCRIPT = "modules/dopd/harness/summarize_multiprompt.py"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean(np.asarray([row[key] for row in rows], dtype=np.float64)))


def _binomial_two_sided(positive: int, negative: int) -> float:
    """p exact du test des signes sur les prompts (binomiale, deux queues)."""
    total = positive + negative
    if total == 0:
        return float("nan")
    tail = sum(math.comb(total, index) for index in range(positive, total + 1))
    return float(min(1.0, 2.0 * tail / 2**total))


def _blocks_per_configuration(rows: list[dict]) -> dict[str, int]:
    distinct: dict[str, set[int]] = {}
    for row in rows:
        distinct.setdefault(str(row["config_index"]), set()).add(row["block_index"])
    return {name: len(values) for name, values in sorted(distinct.items())}


def derive(results_path: Path) -> dict:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    rows = data["rows"]
    multiprompt = data["multiprompt"]
    if not rows:
        raise ValueError("results.json ne contient aucune ligne de marginaux")

    n = len(rows)
    mean_causal = _mean(rows, "causal_kl")
    mean_future = _mean(rows, "future_kl")
    mean_delta = _mean(rows, "delta_kl")
    published_delta = float(multiprompt["aggregate_delta_kl"])
    aggregate_error = abs(mean_delta - published_delta)
    if aggregate_error > 1e-12:
        raise ValueError(
            "l'agrégat recalculé diffère du champ publié : "
            f"{mean_delta!r} vs {published_delta!r}"
        )

    per_prompt = multiprompt["per_prompt"]
    prompt_ci95 = {
        prompt: tuple(values)
        for prompt, values in multiprompt["prompt_ci95"].items()
    }
    breakdown = prompt_sign_breakdown(per_prompt, prompt_ci95)
    deltas = np.asarray([row["delta_kl"] for row in rows], dtype=np.float64)
    recomputed_sign = sign_test(deltas)
    if recomputed_sign["positive"] != multiprompt["sign_test"]["positive"]:
        raise ValueError("le test des signes recalculé contredit le résultat publié")
    if recomputed_sign["negative"] != multiprompt["sign_test"]["negative"]:
        raise ValueError("le test des signes recalculé contredit le résultat publié")
    effective_signs = recomputed_sign["n"]
    sign_z = (
        (abs(recomputed_sign["positive"] - effective_signs / 2.0) - 0.5)
        / math.sqrt(effective_signs / 4.0)
        if effective_signs
        else float("nan")
    )

    visible = [row for row in rows if row["visible_future"]]
    degraded = [row for row in visible if row["delta_kl"] < 0.0]
    zero_exact = sum(1 for row in rows if row["delta_kl"] == 0.0)
    near_zero = sum(1 for row in rows if 0.0 < abs(row["delta_kl"]) < 1e-9)

    top_k = multiprompt["top_k"]
    mean_discarded = float(top_k["mean_discarded_mass"])
    legacy_tv = float(top_k["mean_tv_distance"])
    if abs(legacy_tv - mean_discarded) <= 1e-12:
        tv_legacy_status = "already_renormalized"
    elif abs(legacy_tv - mean_discarded / 2.0) <= 1e-12:
        tv_legacy_status = "legacy_half_tv"
    else:
        tv_legacy_status = "unexpected_relation"
    tv_corrected = mean_discarded

    blocks = _blocks_per_configuration(rows)
    decision = decide_multiprompt_gate(
        mean_delta_kl=mean_delta,
        aggregate_ci95=tuple(multiprompt["aggregate_ci95_delta_kl"]),
        median_delta_kl=float(multiprompt["median_delta_kl"]),
        n_prompts=breakdown["n_prompts"],
        n_prompts_significant_positive=breakdown["n_significant_positive"],
        n_prompts_significant_negative=breakdown["n_significant_negative"],
        sign_test_p_value=float(multiprompt["sign_test"]["p_value"]),
        blocks_per_configuration=blocks,
        total_marginals=n,
    )
    published_status = multiprompt.get("gate_j15_status")
    prompt_sign_p = _binomial_two_sided(
        breakdown["n_positive"], breakdown["n_negative"]
    )

    return {
        "schema": SCHEMA,
        "kind": "dopd_annexe_d_multiprompt_derived",
        "source": {
            "path": _relative(results_path),
            "sha256": _sha256_file(results_path),
            "n_rows": n,
        },
        "producer": {
            "script": SCRIPT,
            "script_sha256": _sha256_file(_REPO_ROOT / SCRIPT),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "headline": {
            "n_marginals": n,
            "mean_causal_kl": mean_causal,
            "mean_future_kl": mean_future,
            "mean_delta_kl": mean_delta,
            "aggregate_ci95_delta_kl": list(multiprompt["aggregate_ci95_delta_kl"]),
            "relative_reduction_pct": 100.0 * mean_delta / mean_causal,
            "median_delta_kl": float(multiprompt["median_delta_kl"]),
            "relative_reduction_j14_pct_reference": 16.12,
        },
        "prompts": {
            **breakdown,
            "n_prompts_ci_excludes_zero": (
                breakdown["n_significant_positive"]
                + breakdown["n_significant_negative"]
            ),
            "proportion_prompts_ci_excludes_zero": (
                (breakdown["n_significant_positive"] + breakdown["n_significant_negative"])
                / breakdown["n_prompts"]
            ),
            "proportion_prompts_significant_positive": (
                breakdown["n_significant_positive"] / breakdown["n_prompts"]
            ),
            "sign_test_exact_two_sided_p": prompt_sign_p,
            "per_prompt": per_prompt,
            "prompt_ci95": {
                prompt: list(values) for prompt, values in prompt_ci95.items()
            },
        },
        "marginals": {
            "n_visible_future": len(visible),
            "n_degraded": len(degraded),
            "degraded_fraction": len(degraded) / len(visible) if visible else float("nan"),
            "mean_delta_kl_degraded": (
                float(np.mean([row["delta_kl"] for row in degraded]))
                if degraded
                else float("nan")
            ),
            "min_delta_kl_degraded": (
                float(min(row["delta_kl"] for row in degraded))
                if degraded
                else float("nan")
            ),
            "n_zero_exact": zero_exact,
            "n_near_zero_numerical": near_zero,
            "n_zero_within_1e-9": zero_exact + near_zero,
            "fraction_zero_within_1e-9": (zero_exact + near_zero) / n,
            "sign_test_all_marginals": {
                **recomputed_sign,
                "z_normal_approximation": sign_z,
                "p_value_note": (
                    "0.0 = saturation numérique ; p < 1e-300"
                    if recomputed_sign["p_value"] == 0.0
                    else "valeur non saturée"
                ),
            },
        },
        "top_k": {
            "k": int(top_k["k"]),
            "n_marginals_measured": int(top_k["n_marginals_measured"]),
            "mean_discarded_mass": mean_discarded,
            "max_discarded_mass": float(top_k["max_discarded_mass"]),
            "mean_tv_distance_renormalized": tv_corrected,
            "mean_tv_distance_as_published_j15": legacy_tv,
            "tv_relation": tv_legacy_status,
            "identity": "tv_renormalized == discarded_mass",
        },
        "gate": {
            "derived_status": decision["status"],
            "published_status_in_results_json": published_status,
            "consistent_with_published": published_status == decision["status"],
            "criteria": decision["criteria"],
            "rationale": decision["rationale"],
        },
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive le resume J15 verifiable (signes, TV corrigee, decision de "
            "porte) depuis results.json, sans GPU ni torch."
        )
    )
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="ne rien ecrire ; echouer si le fichier derive n'est pas identique",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    results_path = Path(args.results)
    if not results_path.is_absolute():
        results_path = _REPO_ROOT / results_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _REPO_ROOT / output_path
    summary = derive(results_path)
    payload = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if not output_path.is_file():
            print(f"ECHEC : {output_path} absent")
            return 1
        existing = output_path.read_text(encoding="utf-8")
        if existing != payload:
            print(f"ECHEC : {output_path} differe de la derivation courante")
            return 1
        print(f"OK : {output_path} identique a la derivation courante")
        return 0
    output_path.write_text(payload, encoding="utf-8")
    gate = summary["gate"]
    prompts = summary["prompts"]
    top_k = summary["top_k"]
    print(f"resume ecrit : {output_path}")
    print(
        f"prompts : {prompts['n_positive']} positifs / {prompts['n_negative']} negatifs ; "
        f"{prompts['n_significant_positive']} significativement positifs, "
        f"{prompts['n_significant_negative']} significativement negatif(s)"
    )
    print(
        f"reduction relative : {summary['headline']['relative_reduction_pct']:.4f} % ; "
        f"TV renormalisee moyenne : {100.0 * top_k['mean_tv_distance_renormalized']:.4f} % "
        f"({top_k['tv_relation']})"
    )
    print(
        f"decision derivee : {gate['derived_status']} "
        f"(publiee : {gate['published_status_in_results_json']}, "
        f"coherente : {gate['consistent_with_published']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
