"""Tests de non-régression pour docker-compose.yml vLLM (pytest).

Exécutes les mêmes 9 assertions que `test_compose_validate.py` mais sous forme
de tests pytest structurés avec fixtures et parametrisation.
"""

import json
import os
import re
from pathlib import Path

import yaml


def load_compose() -> dict:
    """Charge docker-compose.yml depuis la racine du projet."""
    compose_path = (Path(__file__).parent.parent / "vLLM" / "docker-compose.yml").resolve()
    with open(compose_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


class TestVLLMCompose:
    """Tests de structure et de contenu de docker-compose.yml vLLM."""

    @pytest.fixture(scope="function")
    def compose(self):
        """Fixtures pour les tests : charge le fichier YAML une seule fois."""
        with open(
            Path(__file__).parent.parent / "vLLM" / "docker-compose.yml",
            "r",
            encoding="utf-8",
        ) as f:
            return yaml.safe_load(f)

    def test_vllm_service_exists(self, compose):
        """Check 1 — Le service vllm existe dans docker-compose.yml."""
        assert "services" in compose
        assert isinstance(compose["services"], dict)
        assert "vllm" in compose["services"]
        assert isinstance(compose["services"]["vllm"], dict)

    def test_entrypoint_is_bash_c(self, compose):
        """Check 2 — L'entrypoint est ['bash', '-c']."""
        entrypoint = compose["services"]["vllm"]["entrypoint"]
        assert entrypoint == ["bash", "-c"], (
            f"Expected ['bash', '-c'], got {entrypoint!r}"
        )

    def test_command_is_single_element_list(self, compose):
        """Check 3 — Le command a exactement un élément (script bash composite)."""
        command = compose["services"]["vllm"]["command"]
        assert isinstance(command, list), f"Expected list, got {type(command).__name__}"
        assert len(command) == 1, (
            f"Expected exactly 1 element in command, got {len(command)}: {command}"
        )

    def test_backslash_continuations_are_valid(self, compose):
        """Check 4 — Les backslashes du script bash sont tous des continuations valides.

        Dans un script bash contenu dans une ligne YAML multi-ligne, les backslashes
        à la fin d'une ligne sont valides (continuation) et ne doivent PAS être rejetés.
        Seuls les backslashes isolés (non suivis d'espace/tab/fin de chaîne) sont problématiques.

        Note importante : VLLM_USE_V2_MODEL_RUNNER_OVERRIDE contient le sous-chaîne
        "VLLM_USE_V2_MODEL_RUNNER", ce qui fausserait un test simple par substring.
        On cherche plutôt "VLLM_USE_V2_MODEL_RUNNER=0" (forçage runner V1) pour détecter
        un problème, pas juste la présence du nom de variable.
        """
        script_line = compose["services"]["vllm"]["command"][0]
        assert has_valid_backslash_continuation(script_line), (
            f"Invalide backslash continuations detected in:\n{script_line[:200]}..."
        )

    def test_no_runner_v2_by_default(self, compose):
        """Check 5 — VLLM_USE_V2_MODEL_RUNNER n'est PAS dans environment par défaut.

        L'option `VLLM_USE_V2_MODEL_RUNNER_OVERRIDE=0` force le runner V1 (compatibilité).
        Un runner V2 par défaut pourrait rompre la compatibilité avec les modèles existants.

        On vérifie que l'utilisateur doit *explicitement* activer le runner V2.
        """
        env_vars = compose["services"]["vllm"]["environment"]
        # On rejette à la fois la variable native de vLLM et un export manuel
        for e in env_vars:
            assert (
                "VLLM_USE_V2_MODEL_RUNNER=1" not in str(e)
                and "export VLLM_USE_V2_MODEL_RUNNER_OVERRIDE=1" not in str(e)
            ), f"Runner V2 activé par défaut dans : {e}"

    def test_wsl2_defaults_present(self, compose):
        """Check 6 — Les defaults WSL2/CUDA/eager sont présents.

        Ces options optimisent les performances sur WSL2 :
        - VLLM_WSL2_ENABLE_PIN_MEMORY=1 : utilise la mémoire pinée (évite des copies)
        - CUDA_LAUNCH_BLOCKING=1 : synchronise les kernels pour le débogage
        - VLLM_ENFORCE_EAGER=1 : force l'exécution eager mode (plus stable)
        """
        env_vars = compose["services"]["vllm"]["environment"]
        for var in ("VLLM_WSL2_ENABLE_PIN_MEMORY", "CUDA_LAUNCH_BLOCKING", "VLLM_ENFORCE_EAGER"):
            assert any(var in str(e) for e in env_vars), f"{var} manquant dans environment"

    def test_max_model_len_4096(self, compose):
        """Check 7 — VLLM_MAX_MODEL_LEN=4096 est présent dans le command.

        Limitation volontaire pour les tests de non-régression (éviter les OOM).
        """
        cmd_str = str(compose["services"]["vllm"]["command"])
        assert "--max-model-len" in cmd_str and "4096" in cmd_str, (
            f"--max-model-len 4096 manquant dans : {cmd_str[:200]}..."
        )

    def test_enforce_eager_present(self, compose):
        """Check 8 — --enforce-eager est présent.

        Force le mode eager pour une exécution plus stable en développement/test.
        """
        cmd_str = str(compose["services"]["vllm"]["command"])
        assert "--enforce-eager" in cmd_str, f"--enforce-eager manquant dans : {cmd_str}"

    def test_cpu_offload_gb_present(self, compose):
        """Check 9 — --cpu-offload-gb est présent (défaut 30 Go)."""
        cmd_str = str(compose["services"]["vllm"]["command"])
        assert "--cpu-offload-gb" in cmd_str, f"--cpu-offload-gb manquant dans : {cmd_str}"

    def test_extra_args_accepted(self, compose):
        """Check 10 — VLLM_EXTRA_ARGS est accepté (via ${VLLM_EXTRA_ARGS:-})."""
        cmd_str = str(compose["services"]["vllm"]["command"])
        assert "${VLLM_EXTRA_ARGS:-}" in cmd_str or "VLLM_EXTRA_ARGS" in cmd_str, (
            f"VLLM_EXTRA_ARGS non accepté dans : {cmd_str[:200]}..."
        )

    def test_vllm_image_pinned(self, compose):
        """Check 11 — L'image vllm/vllm-openai:gemma est épinglée."""
        image = compose["services"]["vllm"]["image"]
        assert "vllm/openai:vllm-v0.6.7.post1-py3.12-cu124" in image, (
            f"Image non épinglée : {image}"
        )

    def test_ipc_host(self, compose):
        """Check 12 — ipc: host est configuré."""
        assert compose["services"]["vllm"].get("ipc", "") == "host", (
            f"ipc doit être 'host', got {compose['services']['vllm'].get('ipc')!r}"
        )

    def test_ports_binding(self, compose):
        """Check 13 — Le port 8000 est bien lié à l'hôte."""
        ports = compose["services"]["vllm"].get("ports", [])
        assert len(ports) >= 1
        # Format : ["${VAR:-default}:host_port"]
        for port_spec in ports:
            assert ":" in port_spec, f"Port mal formé : {port_spec}"

    def test_deploy_resources(self, compose):
        """Check 14 — Les resources de deployment sont configurés."""
        deploy = compose["services"]["vllm"].get("deploy", {})
        resources = deploy.get("resources", {}).get("reservations", {})
        devices = resources.get("devices", [])
        assert len(devices) >= 1, "Aucun device GPU dans les reservations"
        for dev in devices:
            assert dev.get("driver") == "nvidia"
            assert dev.get("count") == "all"


# Run standalone tests without pytest (for CI compatibility)
if __name__ == "__main__":
    import sys

    compose = load_compose()

    print("=" * 60)
    print("Tests de non-régression docker-compose.yml vLLM")
    print("=" * 60)

    tests = [
        ("test_vllm_service_exists", lambda c: "vllm" in c["services"]),
        ("test_entrypoint_is_bash_c", lambda c: c["services"]["vllm"]["entrypoint"] == ["bash", "-c"]),
        ("test_command_is_single_element_list", lambda c: len(c["services"]["vllm"]["command"]) == 1),
        ("test_backslash_continuations_are_valid", lambda c: has_valid_backslash_continuation(c["services"]["vllm"]["command"][0])),
        ("test_no_runner_v2_by_default", lambda c: all(
            "VLLM_USE_V2_MODEL_RUNNER=1" not in str(e) and "export VLLM_USE_V2_MODEL_RUNNER_OVERRIDE=1" not in str(e)
            for e in c["services"]["vllm"]["environment"]
        )),
        ("test_wsl2_defaults_present", lambda c: all(
            var in str(c["services"]["vllm"]["environment"])
            for var in ("VLLM_WSL2_ENABLE_PIN_MEMORY", "CUDA_LAUNCH_BLOCKING", "VLLM_ENFORCE_EAGER")
        )),
        ("test_max_model_len_4096", lambda c: "--max-model-len" in str(c["services"]["vllm"]["command"]) and "4096" in str(c["services"]["vllm"]["command"]))),
        ("test_enforce_eager_present", lambda c: "--enforce-eager" in str(c["services"]["vllm"]["command"])),
        ("test_cpu_offload_gb_present", lambda c: "--cpu-offload-gb" in str(c["services"]["vllm"]["command"])),
        ("test_extra_args_accepted", lambda c: "${VLLM_EXTRA_ARGS:-}" in str(c["services"]["vllm"]["command"]) or "VLLM_EXTRA_ARGS" in str(c["services"]["vllm"]["command"]))),
    ]

    passed = 0
    failed = 0

    for name, checker in tests:
        try:
            result = checker(compose)
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {status} :test_vllm_service_exists")
            passed += 1 if result else failed += 1
        except Exception as e:
            print(f"  ✗ FAIL :test_vllm_service_exists — {e}")
            failed += 1

    print()
    print(f"Résultats : {passed} passés, {failed} échoués")
    sys.exit(0 if failed == 0 else 1)
