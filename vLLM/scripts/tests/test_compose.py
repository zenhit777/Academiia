"""Test de non-régression pour vLLM/docker-compose.yml (pytest).

Parse le compose (PyYAML) et simule REELLEMENT le script bash avec
`bash -c` (une fonction vllm() imprimant ses arguments remplace
`exec vllm`). Verifie :
  - entrypoint == ["bash","-c"] ; command = liste d'UN element ;
  - aucun backslash dans le script (bug 686c5a8 : le repli YAML
    transforme les sauts de ligne en espaces, bash recevrait un
    backslash litteral) ;
  - aucun argument vide/reduit a un espace, avec VLLM_EXTRA_ARGS vide
    ET non vide ;
  - VLLM_USE_V2_MODEL_RUNNER absent de environment: (runner V2 =
    defaut de vLLM ; le V1 ne se force que via l'override .env) ;
  - --enforce-eager present par defaut (VLLM_ENFORCE_EAGER=1), absent
    si VLLM_ENFORCE_EAGER=0 ;
  - l'override VLLM_USE_V2_MODEL_RUNNER_OVERRIDE=0 exporte bien
    VLLM_USE_V2_MODEL_RUNNER=0 dans le conteneur, et rien sinon.
"""

import os
import subprocess
from pathlib import Path

import yaml
COMPOSE_PATH = Path(__file__).resolve().parents[2] / "docker-compose.yml"

# Harnais : remplace `exec vllm` par une fonction qui imprime RUNNER puis
# un argument par ligne. Le script doit appeler vllm via `exec vllm ...`.
HARNESS = (
    "vllm() { echo \"RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-<unset>}\"; "
    'for a in "$@"; do echo "ARG:$a"; done; } ; '
)


def load_compose() -> dict:
    with open(COMPOSE_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_script(env_overrides: dict[str, str]) -> tuple[str, list[str], list[str]]:
    """Exécute le script du compose via bash -c. Retourne (runner, args, stderr)."""
    script = load_compose()["services"]["vllm"]["command"][0]
    assert "exec vllm" in script, "le script doit lancer vllm via `exec vllm`"
    env = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_")}
    env.update(env_overrides)
    proc = subprocess.run(
        ["bash", "-c", HARNESS + script.replace("exec vllm", "vllm")],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, f"bash a échoué (rc={proc.returncode}) : {proc.stderr}"
    lines = proc.stdout.splitlines()
    runner = lines[0].removeprefix("RUNNER=")
    args = [l[4:] for l in lines[1:] if l.startswith("ARG:")]
    return runner, args, proc.stderr.splitlines()


def get_env_list() -> list:
    return load_compose()["services"]["vllm"]["environment"]


# --- Structure YAML ---

def test_entrypoint_bash_c():
    assert load_compose()["services"]["vllm"]["entrypoint"] == ["bash", "-c"]


def test_command_single_element():
    command = load_compose()["services"]["vllm"]["command"]
    assert isinstance(command, list) and len(command) == 1


def test_no_backslash_in_script():
    script = load_compose()["services"]["vllm"]["command"][0]
    assert "\\" not in script, (
        "backslash dans le bloc >- : le repli YAML le rend litteral "
        "(bug 686c5a8, 'unrecognized arguments')"
    )


def test_no_v2_runner_in_environment():
    # VLLM_USE_V2_MODEL_RUNNER_OVERRIDE contient le nom de la variable a
    # surveiller : on verifie l'egalite stricte des cles declarees.
    for entry in get_env_list():
        key = entry.split("=", 1)[0]
        assert key != "VLLM_USE_V2_MODEL_RUNNER", (
            f"VLLM_USE_V2_MODEL_RUNNER ne doit pas etre defini dans "
            f"environment (runner V2 = defaut de vLLM) : {entry}"
        )


# --- Simulation bash : aucun argument vide, eager conditionnel ---

def test_no_empty_args_default():
    _, args, _ = run_script({})
    assert args, "vllm n'a reçu aucun argument"
    bad = [a for a in args if not a.strip()]
    assert not bad, f"arguments vides/espaces avec VLLM_EXTRA_ARGS vide : {bad}"
    assert "--enforce-eager" in args, f"--enforce-eager absent par défaut : {args}"


def test_no_empty_args_with_extra_args():
    _, args, _ = run_script({"VLLM_EXTRA_ARGS": "--seed 1 --max-num-batched-tokens 2048"})
    bad = [a for a in args if not a.strip()]
    assert not bad, f"arguments vides/espaces avec VLLM_EXTRA_ARGS non vide : {bad}"
    for flag in ("--seed", "1", "--max-num-batched-tokens", "2048"):
        assert flag in args, f"{flag} manquant : {args}"
    assert "--enforce-eager" in args


def test_enforce_eager_omitted_when_disabled():
    _, args, _ = run_script({"VLLM_ENFORCE_EAGER": "0"})
    assert "--enforce-eager" not in args, (
        f"--enforce-eager doit être absent si VLLM_ENFORCE_EAGER=0 : {args}"
    )


def test_runner_override_exported_only_when_set():
    runner, args, _ = run_script({"VLLM_USE_V2_MODEL_RUNNER_OVERRIDE": "0"})
    assert runner == "0", f"l'override doit exporter VLLM_USE_V2_MODEL_RUNNER=0, lu {runner}"
    runner, _, _ = run_script({})
    assert runner == "<unset>", (
        f"sans override, VLLM_USE_V2_MODEL_RUNNER ne doit pas être défini, lu {runner}"
    )
