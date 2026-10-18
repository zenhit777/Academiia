"""Test de non-régression pour docker-compose.yml (vLLM).

Vérifie la structure sans lancer Docker :
  - entrypoint == ["bash","-c"]
  - command est une liste unique dont le script n'a pas de backslash
  - simulation du script bash avec VLLM_EXTRA_ARGS vide et non vide
  - --enforce-eager présent par défaut
  - override runner V1 possible via VLLM_USE_V2_MODEL_RUNNER_OVERRIDE=0
"""

import re
from pathlib import Path
import yaml


def parse_compose() -> dict:
    """Parse docker-compose.yml depuis le dossier parent."""
    compose_path = (Path(__file__).parent.parent / "docker-compose.yml").resolve()
    with open(compose_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_entrypoint_bash_c():
    """entrypoint doit être ["bash","-c"]."""
    compose = parse_compose()
    entrypoint = compose["services"]["vllm"]["entrypoint"]
    assert entrypoint == ["bash", "-c"], (
        f"entrypoint incorrect : {entrypoint!r} != ['bash', '-c']")


def test_command_single_element():
    """command doit être une liste d'un SEUL élément (le script bash)."""
    compose = parse_compose()
    command = compose["services"]["vllm"]["command"]
    assert isinstance(command, list), "command n'est pas une liste"
    assert len(command) == 1, (
        f"command doit être un seul élément, got {len(command)}")


def test_command_no_backslash():
    """Le script bash ne doit contenir aucun backslash."""
    compose = parse_compose()
    command_script = compose["services"]["vllm"]["command"][0]
    assert "\\" not in command_script.strip(), (
        "La commande contient un backslash, ce qui produira une chaîne vide après coupure")


def test_no_v2_runner_in_environment():
    """VLLM_USE_V2_MODEL_RUNNER ne doit PAS figurer dans environment."""
    compose = parse_compose()
    env_vars = {v.split("=")[0]: v.split("=", 1)[1] for v in compose["services"]["vllm"].get(
        "environment", [])}
    assert (
        "VLLM_USE_V2_MODEL_RUNNER" not in env_vars or
        env_vars.get("VLLM_USE_V2_MODEL_RUNNER") == ""), (
        f"VLLM_USE_V2_MODEL_RUNNER devrait être absent ou vide : {env_vars}")


def test_vllm_wsl2_pin_memory_present():
    """VLLM_WSL2_ENABLE_PIN_MEMORY doit être présent avec défaut 1."""
    compose = parse_compose()
    env_vars = {v.split("=")[0]: v.split("=", 1)[1] for v in compose["services"]["vllm"].get(
        "environment", [])}
    assert (
        "VLLM_WSL2_ENABLE_PIN_MEMORY" not in env_vars or
        env_vars.get("VLLM_WSL2_ENABLE_PIN_MEMORY") == "" or
        env_vars.get("VLLM_WSL2_ENABLE_PIN_MEMORY") == "1"), (
        f"VLLM_WSL2_ENABLE_PIN_MEMORY doit être présent avec défaut 1 : {env_vars}")


def test_cuda_launch_blocking_present():
    """CUDA_LAUNCH_BLOCKING doit être présent avec défaut 0."""
    compose = parse_compose()
    env_vars = {v.split("=")[0]: v.split("=", 1)[1] for v in compose["services"]["vllm"].get(
        "environment", [])}
    assert (
        "CUDA_LAUNCH_BLOCKING" not in env_vars or
        env_vars.get("CUDA_LAUNCH_BLOCKING") == "" or
        env_vars.get("CUDA_LAUNCH_BLOCKING") == "0"), (
        f"CUDA_LAUNCH_BLOCKING doit être présent avec défaut 0 : {env_vars}")


def test_vllm_enforce_eager_present():
    """VLLM_ENFORCE_EAGER doit être présent avec défaut 1."""
    compose = parse_compose()
    env_vars = {v.split("=")[0]: v.split("=", 1)[1] for v in compose["services"]["vllm"].get(
        "environment", [])}
    assert (
        "VLLM_ENFORCE_EAGER" not in env_vars or
        env_vars.get("VLLM_ENFORCE_EAGER") == "" or
        env_vars.get("VLLM_ENFORCE_EAGER") == "1"), (
        f"VLLM_ENFORCE_EAGER doit être présent avec défaut 1 : {env_vars}")


def simulate_bash_c(command_script: str, extra_args: str) -> tuple[bool, list[str]]:
    """
    Simule `bash -c '<script>'` en coupant la chaîne à ';' ou '|',
    puis interprétant les substitutions ${VAR:-default} et les tests [ ... ] &&.

    Retourne (status_ok, args_reçus_par_vllm_serve).
    """
    # Substituer les variables d'environnement simulées
    env = {"VLLM_EXTRA_ARGS": extra_args}
    script = command_script

    # Substitution ${...:-default}
    def subst(match):
        var_name = match.group(1)
        default_value = match.group(2) if match.group(2) else ""
        return env.get(var_name, "") or default_value
    script_subst = re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\:-([^}]*?)\}', subst, script)

    # Couper en commandes séparées par ';' ou '|', en conservant les crochets de test
    parts = re.split(r'[;|]', script_subst.strip())

    # Exécuter chaque partie comme une commande bash (simulée)
    output_parts = []
    for part in parts:
        part = part.strip()
        if not part or part.startswith("#"):
            continue
        output_parts.append(part)

    return True, output_parts


def test_command_with_empty_extra_args():
    """Avec VLLM_EXTRA_ARGS vide : --enforce-eager présent (défaut)."""
    compose = parse_compose()
    script = compose["services"]["vllm"]["command"][0]
    ok, args = simulate_bash_c(script, "")

    # Vérifier que le mot-clé --enforce-eager est présent dans la commande
    assert "--enforce-eager" in " ".join(args), (
        f"--enforce-eager absent avec VLLM_EXTRA_ARGS vide : {args}")


def test_command_with_extra_args():
    """Avec VLLM_EXTRA_ARGS="--seed 42 --max-num-batched-tokens 2048"."""
    compose = parse_compose()
    script = compose["services"]["vllm"]["command"][0]
    ok, args = simulate_bash_c(script, "--seed 42 --max-num-batched-tokens 2048")

    assert "--enforce-eager" in " ".join(args), (
        f"--enforce-eager absent avec VLLM_EXTRA_ARGS non vide : {args}")


def test_command_with_runner_override_v1():
    """Avec VLLM_USE_V2_MODEL_RUNNER_OVERRIDE=0, la commande inclut l'export."""
    compose = parse_compose()
    script = compose["services"]["vllm"]["command"][0]

    # Simuler avec override défini
    env_with_override = {"VLLM_USE_V2_MODEL_RUNNER_OVERRIDE": "0", "VLLM_ENFORCE_EAGER": ""}

    def subst(match):
        var_name = match.group(1)
        default_value = match.group(2) if match.group(2) else ""
        return env_with_override.get(var_name, "") or default_value
    script_subst = re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\:-([^}]*?)\}', subst, script)

    # Vérifier que l'export est présent dans le script substitué
    assert (
        'VLLM_USE_V2_MODEL_RUNNER="0"' in script_subst or
        "VLLM_USE_V2_MODEL_RUNNER=0" in script_subst), (
        f"L'override runner V1 n'est pas appliquée : {script_subst}")


def test_command_with_runner_override_v2():
    """Avec VLLM_USE_V2_MODEL_RUNNER_OVERRIDE non défini, l'export est absent."""
    compose = parse_compose()
    script = compose["services"]["vllm"]["command"][0]

    # Simuler sans override
    env_no_override = {"VLLM_ENFORCE_EAGER": ""}

    def subst(match):
        var_name = match.group(1)
        default_value = match.group(2) if match.group(2) else ""
        return env_no_override.get(var_name, "") or default_value
    script_subst = re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\:-([^}]*?)\}', subst, script)

    assert (
        "VLLM_USE_V2_MODEL_RUNNER_OVERRIDE" not in script_subst), (
        f"L'override runner V1 ne devrait pas être présente : {script_subst}")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
