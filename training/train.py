"""Train a supplied network, restore its best validation checkpoint, export it."""
import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random

import torch
from torch import nn

from .balance import balance_targets, read_plan
from .data import RunData, make_loaders
from .network_factory import NetworkSpec, make_network


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    sequence_length: int = 8
    patience: int = 8
    seed: int = 42
    threshold: float = 0.25
    collision_fraction: float | None = None
    optimizer: str = "adam"

    def validate(self):
        for name in ("epochs", "batch_size", "sequence_length", "patience"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.epochs > 1000 or self.batch_size > 4096 or self.sequence_length > 256:
            raise ValueError("Maximum epochs/batch_size/sequence_length: 1000/4096/256")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0 or not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Invalid learning rate or weight decay")
        if not 0 < self.threshold < 1 or self.optimizer not in ("adam", "sgd"):
            raise ValueError("Threshold must be in (0, 1); optimizer must be adam or sgd")
        if type(self.seed) is not int or not 0 <= self.seed <= 0xffffffff:
            raise ValueError("Seed must be an unsigned 32-bit integer")


@torch.no_grad()
def evaluate(model, loader, threshold):
    model.eval()
    total, loss_sum, fp, fn, positives = 0, 0.0, 0, 0, 0
    for inputs, lengths, labels in loader:
        logits = model(inputs, lengths).squeeze(-1)
        loss_sum += nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="sum").item()
        predicted = torch.sigmoid(logits) >= threshold
        actual = labels == 1
        total += len(labels)
        positives += actual.sum().item()
        fp += (predicted & ~actual).sum().item()
        fn += (~predicted & actual).sum().item()
    negatives = total - positives
    return {
        "loss": loss_sum / total if total else None, "samples": total,
        "falsePositives": fp, "missedCollisions": fn, "positiveTargets": positives,
        "falsePositiveRate": fp / negatives if negatives else None,
        "missedCollisionRate": fn / positives if positives else None,
    }


def train(dataset, network, output, config=None, balance_plan=None):
    """Accept a dataset path/RunData and an initialized factory network.

    Returns the restored best model and its report. CLI seeds model initialization
    before calling this; programmatic callers control initialization themselves.
    """
    config = config or TrainConfig()
    config.validate()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    run = dataset if isinstance(dataset, RunData) else RunData(dataset)
    if network.spec.input_size != 6 or network.spec.output_size != 1:
        raise ValueError("This dataset requires six input features and one collision output")
    if not run.keys["train"] or not run.keys["validation"] or not run.keys["test"]:
        raise ValueError("Training requires nonempty train, validation, and test splits; collect at least 10 episodes")
    if balance_plan and config.collision_fraction is not None:
        raise ValueError("Choose a balance plan or collision_fraction, not both")
    balance = None
    selected = None
    if balance_plan:
        selected = read_plan(run, balance_plan)
    elif config.collision_fraction is not None:
        balance = balance_targets(run, config.collision_fraction, config.seed)
        selected = [tuple(key) for key in balance["selected"]]
    sequence_length = 1 if network.spec.kind == "feedforward" else config.sequence_length
    loaders = make_loaders(run, sequence_length, config.batch_size, selected, config.seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps({"status": "training"}))
    if balance:
        (output / "balance.json").write_text(json.dumps(balance))
    network.cpu()
    optimizer_type = torch.optim.Adam if config.optimizer == "adam" else torch.optim.SGD
    optimizer = optimizer_type(network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history = []
    initial = evaluate(network, loaders["validation"], config.threshold)
    if not math.isfinite(initial["loss"]):
        raise ValueError("Initial validation loss is not finite")
    best_loss, best_epoch = initial["loss"], 0
    best_weights = copy.deepcopy(network.state_dict())
    history.append({"epoch": 0, "trainLoss": None, "validation": initial})
    print(json.dumps({"event": "epoch", **history[-1]}), flush=True)
    for epoch in range(1, config.epochs + 1):
        network.train()
        total, total_loss = 0, 0.0
        for inputs, lengths, labels in loaders["train"]:
            optimizer.zero_grad()
            logits = network(inputs, lengths).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Training diverged: non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
            total += len(labels)
        validation = evaluate(network, loaders["validation"], config.threshold)
        if not math.isfinite(validation["loss"]):
            raise ValueError("Validation loss is not finite")
        history.append({"epoch": epoch, "trainLoss": total_loss / total, "validation": validation})
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            best_weights = copy.deepcopy(network.state_dict())
        print(json.dumps({"event": "epoch", **history[-1], "bestEpoch": best_epoch}), flush=True)
        if epoch - best_epoch >= config.patience:
            break
    network.load_state_dict(best_weights)
    network.eval()
    # The test split is touched only once, after selecting the best validation epoch.
    test_metrics = evaluate(network, loaders["test"], config.threshold)
    report = {
        "status": "complete", "prediction": run.prediction, "bestEpoch": best_epoch, "bestValidationLoss": best_loss,
        "test": test_metrics, "history": history, "hyperparameters": asdict(config),
        "network": asdict(network.spec), "datasetFingerprint": run.fingerprint,
        "splitTargets": {split: len(loader.dataset) for split, loader in loaders.items()},
        "balanced": selected is not None,
    }
    artifact = {
        "schemaVersion": 2, "simulator": "circle-inertia-v1", "prediction": run.prediction, "network": asdict(network.spec),
        "sequenceLength": sequence_length, "threshold": config.threshold,
        "normalization": {"sensorRange": run.config["sensorRange"], "actionScale": 5},
        "simulation": run.config, "weights": network.export_weights(),
        "training": {"bestEpoch": best_epoch, "validationLoss": best_loss, "test": test_metrics,
                     "datasetFingerprint": run.fingerprint, "randomLayouts": run.manifest["options"].get("randomLayouts", False)},
    }
    torch.save(best_weights, output / "weights.pt")
    (output / "model.json").write_text(json.dumps(artifact, allow_nan=False))
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"event": "complete", "bestEpoch": best_epoch, "test": test_metrics, "model": str(output / "model.json")}), flush=True)
    return network, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--network", choices=["feedforward", "rnn", "gru"], default="feedforward")
    parser.add_argument("--hidden", default="32,16", help="Feedforward widths; the first width is used for RNN/GRU")
    parser.add_argument("--layers", type=int, default=1, help="Recurrent layers")
    parser.add_argument("--network-config", help="JSON NetworkSpec; overrides --network/--hidden/--layers")
    parser.add_argument("--config", help="Optional JSON TrainConfig overrides")
    parser.add_argument("--balance-plan", help="Optional balance.json produced by training.balance")
    args = parser.parse_args()
    try:
        config = TrainConfig(**json.loads(Path(args.config).read_text())) if args.config else TrainConfig()
        config.validate()
        hidden = [int(value) for value in args.hidden.split(",")]
        spec = NetworkSpec(**json.loads(Path(args.network_config).read_text())) if args.network_config else NetworkSpec(
            kind=args.network, hidden_sizes=hidden, hidden_size=hidden[0], num_layers=args.layers)
        torch.manual_seed(config.seed)
        train(args.dataset, make_network(spec), args.out, config, args.balance_plan)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
        parser.exit(1, f"Training failed: {error}\n")


if __name__ == "__main__":
    main()
