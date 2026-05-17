from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "gpt_realtime" / "run_clanker.py"
spec = importlib.util.spec_from_file_location("run_clanker_test", SCRIPT_PATH)
run_clanker = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = run_clanker
spec.loader.exec_module(run_clanker)


class RealtimeHandoffTest(unittest.TestCase):
    def test_canonical_target_accepts_marshmellow_spelling(self):
        self.assertEqual(run_clanker.canonical_target("marshmellow"), "marshmallow")
        self.assertEqual(run_clanker.canonical_target("Marshmallow"), "marshmallow")

    def test_load_policy_configs_reads_food_policy_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "food_policies.json"
            path.write_text(
                json.dumps(
                    {
                        "targets": {
                            "strawberry": {
                                "display_name": "Strawberry",
                                "policy_repo_id": "org/strawberry",
                                "task": "Pick Strawberry",
                            },
                            "oreo": {
                                "display_name": "Oreo",
                                "policy_repo_id": "org/oreo",
                                "task": "Pick Oreo",
                            },
                            "marshmallow": {
                                "display_name": "Marshmallow",
                                "policy_repo_id": "org/marshmallow",
                                "task": "Pick Marshmallow",
                            },
                        }
                    }
                )
            )

            configs = run_clanker.load_policy_configs(path)

        self.assertEqual(configs["oreo"].policy_repo_id, "org/oreo")
        self.assertEqual(configs["marshmallow"].display_name, "Marshmallow")

    def test_policy_runner_dry_run_returns_structured_success(self):
        config = run_clanker.PolicyConfig(
            target="strawberry",
            path=Path("food_policies.json"),
            display_name="Strawberry",
            policy_repo_id="org/strawberry",
            task="Pick Strawberry",
        )
        runner = run_clanker.PolicyRunner(
            {"strawberry": config},
            dry_run=True,
            dry_run_seconds=0,
        )

        result = runner.run("strawberry")

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["success"])
        self.assertTrue(result["spoken_by_control_loop"])
        self.assertEqual(result["policy_repo_id"], "org/strawberry")


if __name__ == "__main__":
    unittest.main()
