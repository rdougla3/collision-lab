"""Undersample training targets without deleting the history used by an RNN."""
import argparse
import json
import math
import random
from pathlib import Path

from .data import RunData


def balance_targets(run, collision_fraction=0.5, seed=42):
    if not math.isfinite(collision_fraction) or not 0 < collision_fraction <= 1:
        raise ValueError("collision_fraction must be in (0, 1]")
    positive, negative = [], []
    for key in run.keys["train"]:
        (positive if run.episodes[key[0]][key[1]]["lookaheadCollision"] else negative).append(key)
    if not positive:
        raise ValueError("Training split contains no lookahead collision targets; collect more episodes")
    natural = len(positive) / (len(positive) + len(negative))
    if collision_fraction + 1e-12 < natural:
        raise ValueError(f"Target {collision_fraction:.3f} is below current fraction {natural:.3f}; this tool only removes safe targets")
    keep = min(len(negative), round(len(positive) * (1 - collision_fraction) / collision_fraction))
    selected = sorted(positive + random.Random(seed).sample(negative, keep))
    return {
        "schemaVersion": 2, "prediction": run.prediction, "datasetFingerprint": run.fingerprint, "seed": seed,
        "requestedCollisionFraction": collision_fraction,
        "achievedCollisionFraction": len(positive) / len(selected),
        "originalTrainTargets": len(run.keys["train"]), "selectedTrainTargets": len(selected),
        "collisionTargets": len(positive), "safeTargets": keep,
        "selected": selected,
    }


def read_plan(run, path):
    plan = json.loads(Path(path).read_text())
    if (plan.get("schemaVersion") != 2 or plan.get("prediction") != run.prediction
            or plan.get("datasetFingerprint") != run.fingerprint):
        raise ValueError("Balance plan belongs to a different dataset")
    return [tuple(key) for key in plan["selected"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--collision-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        plan = balance_targets(RunData(args.dataset), args.collision_fraction, args.seed)
        with open(args.out, "x") as stream:
            json.dump(plan, stream, indent=2)
        print(json.dumps({key: value for key, value in plan.items() if key != "selected"}))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Balancing failed: {error}\n")


if __name__ == "__main__":
    main()
