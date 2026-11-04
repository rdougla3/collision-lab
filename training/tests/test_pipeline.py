import contextlib
import io
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch

from training.balance import balance_targets, read_plan
from training.data import RunData, DecisionDataset, collate_windows
from training.network_factory import NetworkSpec, make_network
from training.train import TrainConfig, train, evaluate

ROOT = Path(__file__).resolve().parents[2]


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.dataset = cls.root / "run"
        subprocess.run(["node", str(ROOT / "dist/cli.js"), "--out", str(cls.dataset),
                        "--episodes", "30", "--max-actions", "60",
                        "--random-layouts", "--decisions-only"], check=True, capture_output=True)
        cls.dataset_run = RunData(cls.dataset)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_balancing_preserves_all_collision_targets_and_original_history(self):
        plan = balance_targets(self.dataset_run, .5, 12)
        self.assertEqual(plan, balance_targets(self.dataset_run, .5, 12))
        selected = [tuple(key) for key in plan["selected"]]
        collisions = {key for key in self.dataset_run.keys["train"] if self.dataset_run.episodes[key[0]][key[1]]["lookaheadCollision"]}
        self.assertTrue(collisions.issubset(set(selected)))
        future_only = {key for key in collisions if not self.dataset_run.episodes[key[0]][key[1]]["collision"]}
        self.assertTrue(future_only)
        self.assertTrue(future_only.issubset(set(selected)))
        self.assertLessEqual(abs(plan["achievedCollisionFraction"] - .5), .05)
        dataset = DecisionDataset(self.dataset_run, "train", 8, selected)
        for i, (episode_id, end) in enumerate(selected):
            inputs, target = dataset[i]
            original = self.dataset_run.episodes[episode_id][max(0, end - 7):end + 1]
            self.assertTrue(torch.allclose(inputs, torch.tensor([self.dataset_run.features(row) for row in original])))
            self.assertEqual(target.item(), float(original[-1]["lookaheadCollision"]))
        self.assertEqual(len(DecisionDataset(self.dataset_run, "test", 8)), len(self.dataset_run.keys["test"]))
        with self.assertRaises(ValueError):
            DecisionDataset(self.dataset_run, "train", 8, self.dataset_run.keys["test"])
        path = self.root / "plan.json"
        path.write_text(json.dumps(plan))
        self.assertEqual(read_plan(self.dataset_run, path), selected)
        plan["datasetFingerprint"] = "wrong"
        path.write_text(json.dumps(plan))
        with self.assertRaises(ValueError):
            read_plan(self.dataset_run, path)

    def test_early_histories_are_retained_and_right_padding_does_not_change_predictions(self):
        dataset = DecisionDataset(self.dataset_run, "train", 8)
        samples = [dataset[i] for i in range(min(8, len(dataset))) ]
        inputs, lengths, _ = collate_windows(samples)
        self.assertEqual(lengths[0].item(), 1)
        for kind in ("feedforward", "rnn", "gru"):
            model = make_network(kind, hidden_sizes=[5], hidden_size=5, num_layers=2)
            batch = model(inputs, lengths)
            individual = torch.cat([model(x.unsqueeze(0)) for x, _ in samples])
            self.assertTrue(torch.allclose(batch, individual, atol=1e-6), kind)

    def test_train_best_checkpoint_and_browser_parity_for_all_network_types(self):
        for kind in ("feedforward", "rnn", "gru"):
            torch.manual_seed(7)
            network = make_network(NetworkSpec(kind=kind, hidden_sizes=[5, 3], hidden_size=5, num_layers=2))
            output = self.root / kind
            with contextlib.redirect_stdout(io.StringIO()):
                trained, report = train(self.dataset_run, network, output,
                    TrainConfig(epochs=3, batch_size=16, sequence_length=4, patience=3, seed=7))
            losses = [epoch["validation"]["loss"] for epoch in report["history"]]
            self.assertEqual(report["bestValidationLoss"], min(losses))
            saved = torch.load(output / "weights.pt", weights_only=True)
            self.assertTrue(all(torch.equal(value, saved[name]) for name, value in trained.state_dict().items()))
            windows = DecisionDataset(self.dataset_run, "test", 1 if kind == "feedforward" else 4)
            samples = [windows[i][0] for i in range(min(12, len(windows)))]
            expected = [trained(x.unsqueeze(0)).item() for x in samples]
            path = output / "windows.json"
            path.write_text(json.dumps([x.tolist() for x in samples]))
            script = f"""
                import {{ readFileSync }} from 'node:fs';
                import {{ BrowserNetwork }} from {json.dumps((ROOT / 'dist/network.js').as_uri())};
                const model = new BrowserNetwork(JSON.parse(readFileSync(process.argv[1], 'utf8')));
                const windows = JSON.parse(readFileSync(process.argv[2], 'utf8'));
                console.log(JSON.stringify(windows.map(window => model.logit(window))));
            """
            result = subprocess.run(["node", "--input-type=module", "-e", script, str(output / "model.json"), str(path)],
                                    check=True, text=True, capture_output=True)
            actual = json.loads(result.stdout)
            for a, b in zip(actual, expected):
                self.assertAlmostEqual(a, b, places=5, msg=kind)
            self.assertEqual(report["test"]["samples"], len(self.dataset_run.keys["test"]))
            self.assertEqual(report["test"]["positiveTargets"], sum(
                self.dataset_run.episodes[e][i]["lookaheadCollision"] for e, i in self.dataset_run.keys["test"]))
            artifact = json.loads((output / "model.json").read_text())
            self.assertEqual(artifact["prediction"], self.dataset_run.prediction)
            self.assertEqual(artifact["schemaVersion"], 2)

    def test_loader_rejects_datasets_with_an_old_decision_interval(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = json.loads((self.dataset / "manifest.json").read_text())
            manifest["simulation"]["actionSteps"] = 25
            (Path(folder) / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "5 steps per action"):
                RunData(folder)

    def test_loader_rejects_missing_or_wrong_prediction_semantics(self):
        with tempfile.TemporaryDirectory() as folder:
            for prediction in (None, {"target": "collision", "steps": 5, "continuation": "hold-world-force"}):
                manifest = json.loads((self.dataset / "manifest.json").read_text())
                manifest["prediction"] = prediction
                (Path(folder) / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "lookahead labels"):
                    RunData(folder)

    def test_factory_rejects_invalid_specs_and_config(self):
        for spec in (NetworkSpec(kind="other"), NetworkSpec(hidden_size=0), NetworkSpec(hidden_sizes=[])):
            with self.assertRaises(ValueError):
                make_network(spec)
        with self.assertRaises(ValueError):
            TrainConfig(epochs=0).validate()
        with self.assertRaises(ValueError):
            balance_targets(self.dataset_run, 0)

    def test_loader_rejects_incomplete_negative_horizons_and_wrong_labels(self):
        destination = self.root / "invalid-run"
        destination.mkdir()
        (destination / "manifest.json").write_bytes((self.dataset / "manifest.json").read_bytes())
        original = (self.dataset / "decisions.jsonl").read_text().splitlines()
        for field, value in (("elapsedSteps", 0), ("terminated", True), ("lookaheadSteps", 5), ("lookaheadElapsedSteps", 0), ("lookaheadCollision", None)):
            rows = [json.loads(line) for line in original]
            negative = next(row for row in rows if not row["collision"])
            negative[field] = value
            (destination / "decisions.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            with self.assertRaises(ValueError):
                RunData(destination)


if __name__ == "__main__":
    unittest.main()
