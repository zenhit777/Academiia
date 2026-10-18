#!/usr/bin/env python3
"""Validation pure-Python de docker-compose.yml (pour CI sans pip system-wide).

Exécute les mêmes 9 assertions que test_compose.py mais avec pyyaml natif.
Utilisé quand pip n'est pas disponible ou bloqué par PEP 668.

Backslash de continuation : dans un script bash contenu dans une ligne YAML multi-ligne,
les backslashes à la fin d'une ligne sont valides (continuation) et ne doivent PAS
être rejetés. Seuls les backslashes isolés (non suivis d'espace/tab/fin de chaîne)
sont problématiques.

Note importante : VLLM_USE_V2_MODEL_RUNNER_OVERRIDE contient le sous-chaîne
"VLLM_USE_V2_MODEL_RUNNER", ce qui fausse un test simple par substring.
On cherche plutôt "VLLM_USE_V2_MODEL_RUNNER=0" (forçage runner V1) pour détecter
un problème, pas juste la présence du nom de variable.
"""

import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Erreur : pyyaml non installé. Installez-le avec:", file=sys.stderr)
    print("  pip install --break-system-packages pyyaml", file=sys.stderr)
    sys.exit(1)


def has_valid_backslash_continuation(script_line: str) -> bool:
    """Vérifie que tous les backslashes dans le script sont des continuations valides."""
    for i, ch in enumerate(script_line):
        if ch == "\\":
            # Backslash suivi d'espace ou tab → continuation valide
            if i + 1 < len(script_line) and script_line[i + 1] in (" ", "\t"):
                continue
            # Fin de chaîne → pas besoin de continuation, c'est OK
            if i == len(script_line) - 1:
                continue
            # Dans des guillemets (simple ou double), \ est littéral → valide
            if script_line[i - 1] in ('"', "'") or script_line[i + 1] in ('"', "'"):
                continue
            # Backslash isolé → invalide (probable bug)
            return False
    return True


# Chemin absolu vers la racine du projet Academiia
PROJECT_ROOT = Path("/mnt/d/Intelligence_Artificielle/Projects/Academiia")


def validate_compose(yaml_path: Path) -> dict[str, bool | list[str]]:
    """Valide la structure de docker-compose.yml. Retourne un dictionnaire avec les résultats."""

    with open(yaml_path, "r", encoding="utf-8") as f:
        compose = yaml.safe_load(f)

    # DEBUG
    print("DEBUG: type(compose) =", type(compose).__name__)
    if isinstance(compose, dict):
        print("DEBUG: keys =", list(compose.keys()))

    results: dict[str, bool | list[str]] = {}

    # 1. Entrypoint doit être ["bash", "-c"]
    entrypoint = compose["services"]["vllm"]["entrypoint"]
    results["entrypoint_bash_c"] = entrypoint == ["bash", "-c"]

    # 2. command doit avoir exactement un élément (script bash composite)
    command = compose["services"]["vllm"]["command"]
    results["command_single_element"] = len(command) == 1

    # 3. Le script du command ne contient que des backslashes de continuation valides
    script_line = command[0] if command else ""
    results["valid_backslash_continuation"] = has_valid_backslash_continuation(script_line)

    # 4. VLLM_USE_V2_MODEL_RUNNER n'est PAS dans environment par défaut.
    env_vars = compose["services"]["vllm"]["environment"]
    runner_v1_present = any(str(e) == "VLLM_USE_V2_MODEL_RUNNER=0" for e in env_vars)
    runner_override_active = any(
        "VLLM_USE_V2_MODEL_RUNNER_OVERRIDE=0" in str(e) or "export VLLM_USE_V2_MODEL_RUNNER" in str(e)
        for e in env_vars
    )
    results["no_runner_v2_default"] = not (runner_v1_present or runner_override_active)

    # 5. Defaults WSL2/CUDA/eager présents
    defaults_ok = all(
        var in str(env_vars)
        for var in ["VLLM_WSL2_ENABLE_PIN_MEMORY", "CUDA_LAUNCH_BLOCKING", "VLLM_ENFORCE_EAGER"]
    )
    results["wsl2_defaults"] = defaults_ok

    # 6. VLLM_MAX_MODEL_LEN=4096 présent dans le command (via variable env)
    cmd_str = str(command) if command else ""
    has_max_len_4096 = "--max-model-len" in cmd_str and "4096" in cmd_str
    results["max_model_len_4096"] = has_max_len_4096

    # 7. --enforce-eager présent dans le branch `VLLM_ENFORCE_EAGER=1`
    has_enforce_eager = "--enforce-eager" in cmd_str and "eager" in cmd_str
    results["enforce_eager_present"] = has_enforce_eager

    # 8. --cpu-offload-gb présent (défaut 30 Go)
    has_cpu_offload_gb = "--cpu-offload-gb" in cmd_str
    results["cpu_offload_gb_present"] = has_cpu_offload_gb

    # 9. VLLM_EXTRA_ARGS accepté dans le command (via ${VLLM_EXTRA_ARGS:-})
    has_extra_args = "${VLLM_EXTRA_ARGS:-}" in cmd_str or "VLLM_EXTRA_ARGS" in cmd_str
    results["extra_args_accepted"] = has_extra_args

    return results


def main() -> int:
    yaml_path = PROJECT_ROOT / "vLLM" / "docker-compose.yml"

    if not yaml_path.exists():
        print(f"Fichier introuvable : {yaml_path}", file=sys.stderr)
        sys.exit(1)

    results = validate_compose(yaml_path)

    all_ok = True
    for key, passed in sorted(results.items()):
        status = "✓" if passed else "✗"
        print(f"  {status} {key}: {passed!r}")
        if not passed:
            all_ok = False

    print()
    if all_ok:
        print("✅ Tous les tests de non-régression passent !")
        return 0
    else:
        print("❌ Certains tests ont échoué.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
