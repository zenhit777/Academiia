#!/usr/bin/env python3
"""smoke_chat.py — Smoke test de génération via l'API OpenAI-compatible de vLLM.

Client HTTP stdlib uniquement (urllib) : interroge POST /v1/chat/completions
d'un serveur vLLM déjà lancé (ne charge AUCUN modèle lui-même).

Contrôles objectifs seulement : HTTP 200, texte non vide, pas de répétition
dégénérée évidente. Aucun score de qualité n'est inventé.

Sorties : réponse brute (JSON), temps de réponse, finish_reason, longueur du
texte. Sauvegarde : eval/smoke_<modèle>_<horodatage>.json (petits fichiers
de preuve, volontairement NON ignorés par git).

En cas d'erreur 5xx : affiche la commande de logs et rappelle
CUDA_LAUNCH_BLOCKING=1 (les erreurs CUDA sont asynchrones — le point de
crash affiché est souvent trompeur).
"""

import argparse
import datetime
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default="http://localhost:8000",
                   help="URL de base du serveur vLLM (défaut : http://localhost:8000)")
    p.add_argument("--model", default="jainius",
                   help="nom du modèle servi (défaut : jainius)")
    p.add_argument("--timeout", type=int, default=900,
                   help="timeout HTTP en secondes (défaut : 900, cf. serveur vLLM)")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--message",
                   default="Combien font 7 × 8 ? Réponds en une phrase.",
                   help="message utilisateur envoyé au modèle")
    p.add_argument("--output-dir", default="eval",
                   help="dossier de sauvegarde du JSON (défaut : eval/)")
    return p.parse_args()


def degenerate(text: str) -> str | None:
    """Retourne une description de répétition dégénérée évidente, sinon None."""
    if not text:
        return "texte vide"
    words = text.split()
    # Un même mot répété quasi partout.
    if len(words) >= 8:
        uniq_ratio = len(set(words)) / len(words)
        if uniq_ratio <= 0.2:
            return f"vocabulaire dégénéré ({len(set(words))} mots uniques / {len(words)})"
    # Un n-gramme (4-16 chars) répété de très nombreuses fois.
    for n in (4, 8, 16):
        if len(text) >= n * 10:
            chunk = text[:n]
            if text.count(chunk) >= 10:
                return f"répétition dégénérée : {chunk!r} × {text.count(chunk)}"
    return None


def save(result: dict, output_dir: str) -> pathlib.Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"smoke_{result['model']}_{ts}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.message}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        args.url.rstrip("/") + "/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"}, method="POST")

    print(f"POST {args.url}/v1/chat/completions  model={args.model!r} max_tokens={args.max_tokens}")
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            status = resp.status
            raw = resp.read().decode()
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:2000]
        print(f"ÉCHEC HTTP {e.code} en {time.monotonic() - t0:.1f} s : {detail}", file=sys.stderr)
        if e.code >= 500:
            print(
                "\nLe serveur a renvoyé une erreur 5xx. Si le conteneur a crashé :\n"
                "  docker logs --tail 200 jainius-vllm\n"
                "Relancer avec CUDA_LAUNCH_BLOCKING=1 dans .env pour localiser le vrai\n"
                "point du crash GPU (les erreurs CUDA sont asynchrones).",
                file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"Serveur injoignable ({e}) — le serveur est-il lancé ? `docker compose ps`", file=sys.stderr)
        return 1

    elapsed = time.monotonic() - t0
    print(f"HTTP {status} en {elapsed:.1f} s")
    print("--- Réponse brute ---")
    print(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("Réponse non-JSON — le serveur n'a pas fourni une completion valide.", file=sys.stderr)
        return 1

    choice = (data.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    finish_reason = choice.get("finish_reason")
    print("--- Analyse ---")
    print(f"finish_reason : {finish_reason!r}")
    print(f"longueur du texte : {len(text)} caractères")
    print(f"texte : {text!r}")

    problem = degenerate(text)
    if problem:
        print(f"ÉCHEC : {problem}", file=sys.stderr)
        verdict = "FAIL"
    else:
        print("OK : réponse non vide, pas de répétition dégénérée évidente.")
        verdict = "PASS"

    result = {
        "model": args.model, "url": args.url, "payload": payload,
        "elapsed_s": round(elapsed, 2), "finish_reason": finish_reason,
        "text": text, "verdict": verdict, "raw": data, "saved_at": datetime.datetime.now().isoformat(),
    }
    path = save(result, args.output_dir)
    print(f"Sauvegardé : {path}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
