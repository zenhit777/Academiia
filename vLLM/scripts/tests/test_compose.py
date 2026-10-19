"""Test de non-régression pour vLLM/docker-compose.yml (pytest).

Reproduit le pipeline RÉEL : Compose interpole d'abord `$VAR`/`${VAR}`
(y compris `${V:-défaut}`) dans TOUT le fichier — seul `$$` produit un `$`
littéral — puis le shell du conteneur exécute le script résultant via
`bash -c` (simulé ici avec une fonction vllm() imprimant ses arguments).

Vérifie :
  - entrypoint == ["bash","-c"] ; command = liste d'UN element ;
  - aucun backslash dans le script (bug 686c5a8) ;
  - interpolation Compose conforme ($$ -> $, ${V}, $V, ${V:-d}, ${V-d}) ;
  - (1) défauts -> --enforce-eager présent ; (2) VLLM_ENFORCE_EAGER=0 ->
    absent ; (3) VLLM_EXTRA_ARGS multi-flags -> arguments distincts ;
  - (4) aucun argument vide/réduit à un espace ;
  - (5) aucun ${...}/$VAR résiduel inattendu dans le script interpolé ;
  - VLLM_USE_V2_MODEL_RUNNER absent de environment: ;
  - override runner exporté seulement s'il est défini ;
  - si `docker compose` est disponible : la commande RÉSOLUE contient
    bien $${EAGER} (échappé pour le shell), sinon test skipped.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

COMPOSE_PATH = Path(__file__).resolve().parents[2] / "docker-compose.yml"

HARNESS = (
    "vllm() { echo \"RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-<unset>}\"; "
    'for a in "$@"; do echo "ARG:$a"; done; } ; '
)


def load_compose() -> dict:
    with open(COMPOSE_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- Interpolation conforme à Compose (spécification : $$ -> $, ${V}, $V,
# --- ${V:-d} (défaut si non défini OU vide), ${V-d} (défaut si non défini)).

_VAR_BRACED = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::?-([^}]*))?\}")
_VAR_BARE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


def compose_interpolate(script: str, env: dict[str, str]) -> str:
    """Interpole comme Compose : $$ d'abord (protect), puis ${V...}/$V.

    Compose remplace $$ par un marqueur unique, résout les variables,
    puis restaure les $ littéraux.
    """
    DOLLAR = "\x00DOLLAR\x00"
    out = script.replace("$$", DOLLAR)

    def sub_braced(m: re.Match) -> str:
        name, default = m.group(1), m.group(2)
        if name in env:
            return env[name]
        if default is not None and (env.get(name, "") == "" or ":-" in m.group(0)):
            return default
        if default is not None:  # ${V-d} : défaut seulement si non défini
            return env.get(name, default)
        return ""

    out = _VAR_BRACED.sub(sub_braced, out)
    out = _VAR_BARE.sub(lambda m: env.get(m.group(1), ""), out)
    return out.replace(DOLLAR, "$")


def run_script(env_compose: dict[str, str], env_shell: dict[str, str] | None = None) -> tuple[str, list[str]]:
    """Interpole (Compose) puis exécute via bash -c (harnais vllm)."""
    script = load_compose()["services"]["vllm"]["command"][0]
    interpolated = compose_interpolate(script, env_compose)
    assert "exec vllm" in interpolated
    env = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_")}
    env.update(env_shell or {})
    proc = subprocess.run(
        ["bash", "-c", HARNESS + interpolated.replace("exec vllm", "vllm")],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, f"bash a échoué : {proc.stderr}"
    lines = proc.stdout.splitlines()
    runner = lines[0].removeprefix("RUNNER=")
    args = [l[4:] for l in lines[1:] if l.startswith("ARG:")]
    return runner, args


def env_shell_for(env_compose: dict[str, str]) -> dict[str, str]:
    """Variables que Compose passerait aussi au conteneur via environment:."""
    shell = {}
    if "VLLM_USE_V2_MODEL_RUNNER_OVERRIDE" in env_compose:
        shell["VLLM_USE_V2_MODEL_RUNNER_OVERRIDE"] = env_compose["VLLM_USE_V2_MODEL_RUNNER_OVERRIDE"]
    return shell


# --- Structure YAML ---

def test_entrypoint_bash_c():
    assert load_compose()["services"]["vllm"]["entrypoint"] == ["bash", "-c"]


def test_command_single_element():
    command = load_compose()["services"]["vllm"]["command"]
    assert isinstance(command, list) and len(command) == 1


def test_no_backslash_in_script():
    script = load_compose()["services"]["vllm"]["command"][0]
    assert "\\" not in script, "backslash dans le bloc >- (bug 686c5a8)"


def test_no_v2_runner_in_environment():
    for entry in load_compose()["services"]["vllm"]["environment"]:
        key = entry.split("=", 1)[0]
        assert key != "VLLM_USE_V2_MODEL_RUNNER"


# --- (5) Pas de $ résiduel inattendu après interpolation Compose ---

def test_no_uninterpolated_dollar_residue():
    script = load_compose()["services"]["vllm"]["command"][0]
    out = compose_interpolate(script, {"VLLM_EXTRA_ARGS": ""})
    residue = re.findall(r"\$\{[^}]*\}|\$[A-Za-z_][A-Za-z0-9_]*", out)
    # Seuls restent les $ littéraux voulus (shell du conteneur).
    allowed = {"${VLLM_USE_V2_MODEL_RUNNER_OVERRIDE:-}", "$VLLM_USE_V2_MODEL_RUNNER_OVERRIDE",
               "${EAGER}"}
    unexpected = [r for r in residue if r not in allowed]
    assert not unexpected, f"variables non interpolées par Compose ni échappées en $$ : {unexpected}"


# --- (1)(2)(3)(4) Simulation bash après interpolation Compose ---

def test_enforce_eager_present_by_default():
    _, args = run_script({})
    assert "--enforce-eager" in args, f"--enforce-eager absent par défaut : {args}"
    bad = [a for a in args if not a.strip()]
    assert not bad, f"arguments vides : {bad}"


def test_enforce_eager_absent_when_disabled():
    _, args = run_script({"VLLM_ENFORCE_EAGER": "0"})
    assert "--enforce-eager" not in args, f"doit être absent si VLLM_ENFORCE_EAGER=0 : {args}"


def test_extra_args_split_into_distinct_args():
    _, args = run_script({"VLLM_EXTRA_ARGS": "--seed 1 --max-num-batched-tokens 2048"})
    for expected in ("--seed", "1", "--max-num-batched-tokens", "2048"):
        assert expected in args, f"{expected} absent/distinct : {args}"
    bad = [a for a in args if not a.strip()]
    assert not bad, f"arguments vides avec VLLM_EXTRA_ARGS : {bad}"


def test_runner_override_exported_only_when_set():
    runner, _ = run_script({"VLLM_USE_V2_MODEL_RUNNER_OVERRIDE": "0"},
                           env_shell={"VLLM_USE_V2_MODEL_RUNNER_OVERRIDE": "0"})
    assert runner == "0"
    runner, _ = run_script({})
    assert runner == "<unset>", f"override fuiterait dans le défaut : {runner}"


# --- docker compose config (si disponible) ---

def test_compose_config_resolved_command():
    if shutil.which("docker") is None:
        pytest.skip("docker compose non disponible sur cette machine")
    proc = subprocess.run(
        ["docker", "compose", "config"],
        cwd=COMPOSE_PATH.parent, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"docker compose config a échoué : {proc.stderr}"
    # $$EAGER doit SURVIVRE à l'interpolation Compose (littéral pour le shell) ;
    # --enforce-eager est positionné par le shell, pas visible en statique.
    assert "$${EAGER}" in proc.stdout or "${EAGER}" in proc.stdout, (
        "l'interpolation Compose a avalé ${EAGER} -> --enforce-eager ne serait jamais passé"
    )
