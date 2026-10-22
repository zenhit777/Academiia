#!/usr/bin/env python3
"""eval_compare.py — Évaluation par génération d'un modèle servi par vLLM.

Interroge un serveur vLLM (API OpenAI-compatible) avec un jeu de prompts
JSONL et produit un rapport JSON : latence, débit, réponses non vides,
répétitions dégénérées, format respecté, réponses contenant la valeur
attendue.

Toutes les métriques sont calculées à partir de la réponse BRUTE du
serveur : aucun score de qualité n'est inventé. La perplexité n'est pas
utilisable (modèle de diffusion) — seul le comportement de génération est
mesuré.

Un seul modèle à la fois (les checkpoints font ~49 Go : ils ne tiennent pas
simultanément en VRAM ni en RAM).

Usage :
    python modules/a2d/eval/eval_compare.py \
        --base http://localhost:8000/v1/chat/completions \
        --model jainius \
        --prompts modules/a2d/legacy/eval/prompts.jsonl \
        --output eval/results_jainius.json
"""

import argparse
import datetime
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

from academiia.a2d.textcheck import check_format, normalize
from academiia.core.text import degenerate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", default="http://localhost:8000/v1/chat/completions",
                   help="URL de l'endpoint chat/completions")
    p.add_argument("--model", required=True, help="nom du modele servi")
    p.add_argument("--prompts", default="modules/a2d/legacy/eval/prompts.jsonl",
                   help="jeu de prompts (JSONL)")
    p.add_argument("--output", required=True, help="fichier JSON de sortie")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--limit", type=int, default=0,
                   help="ne traiter que les N premiers prompts (0 = tous)")
    p.add_argument("--pause", type=float, default=0.0,
                   help="pause en secondes entre deux requetes")
    return p.parse_args()


def load_prompts(path: str, limit: int) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            out.append(json.loads(line))
    return out[:limit] if limit else out


def query(args, prompt: str) -> dict:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        args.base, data=body, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            raw = resp.read().decode("utf-8")
            status = resp.status
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}",
                "latency_s": time.monotonic() - t0, "text": ""}
    except (urllib.error.URLError, TimeoutError) as e:
        return {"ok": False, "error": f"transport: {e}", "latency_s": time.monotonic() - t0,
                "text": ""}
    latency = time.monotonic() - t0
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"ok": False, "error": "reponse non JSON", "latency_s": latency, "text": ""}
    choice = (data.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    usage = data.get("usage") or {}
    return {
        "ok": True,
        "status": status,
        "text": text,
        "latency_s": latency,
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def main() -> int:
    args = parse_args()
    prompts = load_prompts(args.prompts, args.limit)
    records = []
    for i, p in enumerate(prompts, 1):
        res = query(args, p["prompt"])
        text = res.get("text", "")
        rec = {
            "id": p.get("id", f"p{i:03d}"),
            "category": p.get("category", "other"),
            "lang": p.get("lang", "?"),
            "prompt": p["prompt"],
            "ok": res.get("ok", False),
            "latency_s": round(res.get("latency_s", 0.0), 4),
            "completion_tokens": res.get("completion_tokens"),
            "finish_reason": res.get("finish_reason"),
            "text": text,
        }
        if not rec["ok"]:
            rec["error"] = res.get("error")
            rec["empty"] = True
            rec["degenerate"] = None
        else:
            rec["empty"] = not text.strip()
            reason = degenerate(text)
            rec["degenerate"] = reason
            if p.get("expected"):
                exp = normalize(p["expected"])
                rec["expected"] = p["expected"]
                rec["expected_match"] = exp in normalize(text)
            if p.get("format"):
                rec["format"] = p["format"]
                rec["format_ok"] = check_format(text, p["format"])
        records.append(rec)
        flag = "."
        if not rec["ok"]:
            flag = "E"
        elif rec.get("degenerate"):
            flag = "D"
        elif rec.get("empty"):
            flag = "0"
        elif rec.get("expected_match") is False:
            flag = "x"
        print(f"[{i}/{len(prompts)}] {rec['id']:<12} {flag} "
              f"{rec['latency_s']:6.2f}s", flush=True)
        if args.pause:
            time.sleep(args.pause)

    ok = [r for r in records if r["ok"]]
    latencies = [r["latency_s"] for r in ok]
    total_tokens = sum(r["completion_tokens"] or 0 for r in ok)
    total_time = sum(latencies)
    expected_rows = [r for r in ok if r.get("expected")]
    format_rows = [r for r in ok if r.get("format")]

    def rate(rows, key):
        if not rows:
            return None
        return round(100.0 * sum(1 for r in rows if r.get(key)) / len(rows), 2)

    aggregate = {
        "n_prompts": len(records),
        "n_ok": len(ok),
        "n_error": len(records) - len(ok),
        "latency_mean_ms": round(1000 * statistics.fmean(latencies), 1) if latencies else None,
        "latency_median_ms": round(1000 * statistics.median(latencies), 1) if latencies else None,
        "latency_p95_ms": round(1000 * percentile(latencies, 0.95), 1) if latencies else None,
        "ms_per_token": round(1000 * total_time / total_tokens, 2) if total_tokens else None,
        "tokens_per_s": round(total_tokens / total_time, 2) if total_time else None,
        "completion_tokens_total": total_tokens,
        "pct_non_empty": rate(ok, "empty") if ok else None,
        "pct_degenerate": rate(ok, "degenerate") if ok else None,
        "pct_expected_match": rate(expected_rows, "expected_match"),
        "n_expected": len(expected_rows),
        "pct_format_ok": rate(format_rows, "format_ok"),
        "n_format": len(format_rows),
    }
    # pct_non_empty est le taux de reponses NON VIDES (inversion du flag "empty")
    if aggregate["pct_non_empty"] is not None:
        aggregate["pct_non_empty"] = round(100.0 - aggregate["pct_non_empty"], 2)

    per_category = {}
    for cat in sorted({r["category"] for r in records}):
        rows = [r for r in ok if r["category"] == cat]
        exp_rows = [r for r in rows if r.get("expected")]
        fmt_rows = [r for r in rows if r.get("format")]
        per_category[cat] = {
            "n": len(rows),
            "latency_mean_ms": round(1000 * statistics.fmean([r["latency_s"] for r in rows]), 1)
            if rows else None,
            "pct_degenerate": rate(rows, "degenerate"),
            "pct_expected_match": rate(exp_rows, "expected_match"),
            "n_expected": len(exp_rows),
            "pct_format_ok": rate(fmt_rows, "format_ok"),
            "n_format": len(fmt_rows),
        }

    report = {
        "model": args.model,
        "base": args.base,
        "prompts_file": args.prompts,
        "date": datetime.datetime.now().astimezone().isoformat(),
        "config": {"max_tokens": args.max_tokens, "temperature": args.temperature},
        "aggregate": aggregate,
        "per_category": per_category,
        "records": records,
    }
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== Resume ===")
    for key, value in aggregate.items():
        print(f"  {key:<26} {value}")
    print("\n=== Par categorie ===")
    for cat, stats in per_category.items():
        print(f"  {cat:<12} {stats}")
    print(f"\nEcrit : {args.output}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
