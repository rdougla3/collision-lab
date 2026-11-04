"""Validate runs and build windows from intact episodes before sampling targets."""
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

PREDICTION_SPEC = {"target": "lookaheadCollision", "steps": 50, "continuation": "hold-world-force"}


class RunData:
    def __init__(self, path):
        self.path = Path(path)
        manifest_bytes = (self.path / "manifest.json").read_bytes()
        self.manifest = json.loads(manifest_bytes)
        if self.manifest.get("status") != "complete" or self.manifest.get("simulator") != "circle-inertia-v1":
            raise ValueError("Expected a completed circle-inertia-v1 run")
        self.config = self.manifest["simulation"]
        if self.config.get("actionSteps") != 5 or self.config.get("dt") != 0.02:
            raise ValueError("Training requires 5 steps per action at 0.02 seconds per step; collect a fresh dataset")
        if self.manifest.get("schemaVersion") != 2 or self.manifest.get("prediction") != PREDICTION_SPEC:
            raise ValueError("Training requires one-second lookahead labels; collect a fresh dataset")
        self.prediction = dict(PREDICTION_SPEC)
        raw = (self.path / "decisions.jsonl").read_bytes()
        self.fingerprint = hashlib.sha256(manifest_bytes + b"\0" + raw).hexdigest()
        self.episodes = {}
        self.keys = {split: [] for split in ("train", "validation", "test")}
        layouts = {}
        for line in raw.splitlines():
            row = json.loads(line)
            episode_id, index, split = row["episodeId"], row["decisionIndex"], row["split"]
            if type(episode_id) is not int or episode_id < 0 or type(index) is not int or split not in self.keys:
                raise ValueError("Invalid episode ID, decision index, or split")
            episode = self.episodes.setdefault(episode_id, [])
            if index != len(episode) or (episode and (episode[-1]["terminated"] or episode[-1]["truncated"])):
                raise ValueError("Decisions must be consecutive and stop at episode boundaries")
            if episode and (split != episode[0]["split"] or row["observation"] != episode[-1]["nextObservation"]):
                raise ValueError("Episode split or temporal continuity changed")
            sensors = row["observation"]["sensors"]
            action = row["control"]["action"]
            if len(sensors) != 5 or any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= self.config["sensorRange"] + 1e-6 for v in sensors):
                raise ValueError("Invalid sensor readings")
            if type(action) is not int or not -5 <= action <= 5 or type(row["collision"]) is not bool:
                raise ValueError("Invalid action or collision target")
            if (row.get("schemaVersion") != 2 or type(row.get("lookaheadCollision")) is not bool
                    or type(row.get("lookaheadSteps")) is not int or row["lookaheadSteps"] != self.prediction["steps"]):
                raise ValueError("Missing or inconsistent one-second lookahead target")
            prediction_elapsed = row.get("lookaheadElapsedSteps")
            if (type(prediction_elapsed) is not int or not 1 <= prediction_elapsed <= row["lookaheadSteps"]
                    or (not row["lookaheadCollision"] and prediction_elapsed != row["lookaheadSteps"])
                    or (row["lookaheadCollision"] and prediction_elapsed <= self.config["actionSteps"]) != row["collision"]
                    or (row["collision"] and prediction_elapsed != row["elapsedSteps"])):
                raise ValueError("Lookahead outcome disagrees with its horizon or the actual action prefix")
            if row["plannedSteps"] != self.config["actionSteps"]:
                raise ValueError("Mixed action horizons are not supported in one model")
            if (type(row["terminated"]) is not bool or type(row["truncated"]) is not bool
                    or row["terminated"] != row["collision"] or (row["terminated"] and row["truncated"])):
                raise ValueError("Episode termination flags do not match the collision label")
            elapsed = row["elapsedSteps"]
            if (type(elapsed) is not int or not 1 <= elapsed <= row["plannedSteps"]
                    or (not row["collision"] and elapsed != row["plannedSteps"])
                    or row["nextObservation"]["step"] - row["observation"]["step"] != elapsed):
                raise ValueError("Decision label has an invalid or incomplete horizon")
            if self.manifest["options"].get("randomLayouts"):
                layout = row.get("layoutId")
                if not layout or layouts.setdefault(layout, split) != split:
                    raise ValueError("A randomized layout occurs in multiple splits")
                if episode and layout != episode[0]["layoutId"]:
                    raise ValueError("Layout changed within an episode")
            episode.append(row)
            self.keys[split].append((episode_id, index))
        if not self.episodes or any(not (rows[-1]["terminated"] or rows[-1]["truncated"]) for rows in self.episodes.values()):
            raise ValueError("Dataset is empty or contains unfinished episodes")

    def features(self, row):
        # Physical normalization needs no fitted statistics from held-out data.
        return [v / self.config["sensorRange"] for v in row["observation"]["sensors"]] + [row["control"]["action"] / 5]


class DecisionDataset(Dataset):
    def __init__(self, run, split, sequence_length=1, selected=None):
        if type(sequence_length) is not int or sequence_length < 1:
            raise ValueError("sequence_length must be a positive integer")
        self.run, self.sequence_length = run, sequence_length
        self.keys = list(run.keys[split] if selected is None else selected)
        allowed = set(run.keys[split])
        if len(set(self.keys)) != len(self.keys) or any(key not in allowed for key in self.keys):
            raise ValueError("Selected targets must be unique members of their split")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        episode_id, end = self.keys[index]
        episode = self.run.episodes[episode_id]
        rows = episode[max(0, end - self.sequence_length + 1):end + 1]
        return torch.tensor([self.run.features(row) for row in rows], dtype=torch.float32), torch.tensor(float(episode[end]["lookaheadCollision"]), dtype=torch.float32)


def collate_windows(batch):
    inputs, labels = zip(*batch)
    lengths = torch.tensor([len(value) for value in inputs], dtype=torch.long)
    return pad_sequence(inputs, batch_first=True), lengths, torch.stack(labels)


def make_loaders(run, sequence_length=1, batch_size=64, selected=None, seed=42):
    generator = torch.Generator().manual_seed(seed)
    return {
        split: DataLoader(DecisionDataset(run, split, sequence_length, selected if split == "train" else None),
                          batch_size=batch_size, shuffle=split == "train", collate_fn=collate_windows,
                          generator=generator if split == "train" else None)
        for split in ("train", "validation", "test")
    }
