"""Reject unusable experiment plans before downloading data or training."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.run_experiments import main


class ExperimentPlanTests(unittest.TestCase):
    def test_uncalibratable_legacy_config_fails_before_any_subprocess(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "legacy.yaml"
            config.write_text(
                json.dumps({"source": "har", "loss_type": "triplet", "epochs": 20})
            )
            with (
                mock.patch(
                    "sys.argv",
                    ["run_experiments", "--configs", str(config), "--prepare"],
                ),
                mock.patch("subprocess.run") as run,
                contextlib.redirect_stderr(io.StringIO()) as error,
            ):
                with self.assertRaises(SystemExit) as raised:
                    main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("val_subjects >= 2", error.getvalue())
            run.assert_not_called()

    def test_existing_checkpoint_in_later_config_fails_before_first_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            configs = [root / "first.yaml", root / "second.yaml"]
            for path in configs:
                path.write_text(
                    json.dumps({"source": "har", "model": "cnn_lstm", "epochs": 20})
                )
            output = root / "runs"
            (output / "second").mkdir(parents=True)
            (output / "second/best.pt").touch()
            with (
                mock.patch(
                    "sys.argv",
                    [
                        "run_experiments",
                        "--configs",
                        *map(str, configs),
                        "--out-root",
                        str(output),
                    ],
                ),
                mock.patch("subprocess.run") as run,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    main()
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
