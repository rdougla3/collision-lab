# Collision Lab: collect, train, and test

A TypeScript robot simulator with Python/PyTorch training and browser inference.
Edit an arena, collect random walks, compare feedforward, RNN,
and GRU collision predictors, then use a trained model to steer toward goals.
The browser and headless collector share the same deterministic physics core.

## Run the workbench

Requires Node 18+, npm, and Python 3.10+ with PyTorch for training.

```bash
git clone https://github.com/rdougla3/collision-lab.git
cd collision-lab
npm ci
python3 -m venv .venv
.venv/bin/python -m pip install -r training/requirements.txt
npm run dev
```

Open **http://127.0.0.1:5173**. Startup restores the bundled shared example into
`data/` and `models/`, checking its SHA-256 hashes. The compressed bundle is about
52 MB; allow about 350 MB for the expanded example. Existing matching files are
reused, and different files are never silently overwritten.

Restart the server after source edits: `npm run dev` builds once and has no hot
reload. The server uses `.venv/bin/python` (or `.venv/Scripts/python.exe` on Windows),
then a parent `.venv/bin/python`, then `python3`. Override this with
`COLLISION_PYTHON=/path/to/python npm run dev` if needed. Loading the shared model
and collecting data do not require Python; balancing and training do.

The desktop workbench fits the arena and the active workflow side by side. The
viewport frame follows the arena proportions. Playback and speed sit at the top
left; layout tools sit at the top right. The bottom strip groups run status,
readings, distance sensors, and an always-visible color legend. On phones the
control groups and panels stack. Job logs, extra observations, model files/evaluation, and
advanced training settings expand when needed.

The **Train** tab provides three stages, selected above the controls:

1. Edit the arena and collect a dataset. Choose an episode count
   and maximum actions per episode. A fresh seed is chosen automatically. Collection runs headlessly in Node using a
   snapshot of the edited layout, independently of canvas playback.
2. Select a dataset and optionally create a balance plan. The collision fraction
   applies only to training targets; the original observations remain intact.
3. Choose feedforward, RNN, or GRU, set dimensions and hyperparameters, and train.
   Progress shows validation loss, false positives, and missed collisions. The
   best validation checkpoint is exported and loaded when training finishes.

In **Test**, load a saved model or open an exported `model.json`, then Run or Step.
Use **Select** to drag the goal. The display shows all 11 candidate collision
scores, the selected action, and goal/collision counts. Restart automatically chooses a new exploration seed. The layout is edited directly and persists across restarts. Decisions are fixed at five physics steps (0.1 seconds); models with a different
interval or prediction target must be retrained. Scores predict collision over one
second of holding a candidate force; execution still lasts only 0.1 seconds. Other
physics parameters come from the loaded model.

Run/pause and speed controls inspect the random-walk preview. **Collect dataset**
runs collection on the server, then saves its files in this browser. Playback speed never
changes physics. Only one collection, balancing, or training job runs at a time;
the Cancel button stops it. Incomplete datasets/models are excluded from lists.

Browser collection saves decision records and episode summaries, without the
optional per-physics-step trace. RNN and GRU training still use complete decision
histories. Collection, balancing, training, and preview restarts each choose a fresh
seed automatically; saved artifacts retain their seeds for reproducibility. The
CLI retains explicit seed and fine-grained export options.

Training uses a Python process on the server. Inference runs in JavaScript, and
saved browser models still work when the API is unavailable. This server remains
local tooling bound to loopback; publication is a separate deployment task.

## Shared example and browser-owned experiments

`examples/shared-example.json` pins the user's final collection, balance plan,
and trained model by their existing directory names and dataset fingerprint:

- 10,000 episodes, 326,780 decisions, and a 50/50 balance plan retaining 126,966
  training targets.
- Feedforward network with widths 32,16; selected checkpoint at epoch 10.
- Collection `run-1793743566303-d64fe53b`, balance
  `balance-1793743799351-de69e0e9.json`, model `model-1793743811356-bbaf0539`.

These three artifacts form **one read-only shared example**. The example model is
loaded initially, and its original dataset layout is available through **Use
dataset layout**. Other historical directories are preserved on disk but are not
listed or exposed by the server. Editing the current arena never rewrites the
example's data. The exact source files are included in `examples/bundle/` as gzip archives;
`npm run example:restore` restores and verifies them without regenerating data or
retraining the model. Generated `data/`, `models/`, and private job directories
remain Git-ignored.

New browser experiments use this lifecycle:

1. Each browser receives its own random workspace token, stored in localStorage.
   Every job/input/result request is scoped to that token on the server. A different
   browser cannot list, download, upload to, cancel, or acknowledge those jobs.
2. Collection/training runs in a temporary `.runtime/job-…` directory. For local
   datasets, the browser uploads `manifest.json` and `decisions.jsonl` only when
   needed for balancing/training; selected local balance plans are uploaded too.
   Requests using the shared example read its pinned files directly.
3. Result files are transferred as gzip blobs and committed together with their
   metadata in IndexedDB. This includes dataset JSONL/CSV, balance plans, and model
   JSON/weights/reports. Large datasets are not stored as localStorage strings.
4. Only after the local transaction succeeds does the browser acknowledge the
   result. The server then removes the job's input/output files. Unacknowledged
   results can be recovered after a refresh/server restart for up to 24 hours.
   Old temporary jobs expire automatically; running processes are not removed.

If saving fails (for example, browser quota is full), the result remains on the
server for retry rather than being acknowledged. **Refresh**
retries retrieving available results. Local delete controls remove browser-owned
artifacts; the shared example cannot be deleted. Models opened from JSON files
are saved locally, and loaded models can be downloaded as JSON backups.

This is browser/device storage without accounts or synchronization. Clearing site
data or browser eviction can remove local experiments and the workspace token.
Local metadata, layout, and cached models work offline; new collection, balancing,
and training require the server. Changing the origin/port creates a separate
browser library. One server compute job runs at a time; other browsers see a busy
status but never the running experiment's details.

`COLLISION_RUNTIME_DIR` can override temporary storage (use different directories
for separate server processes). `COLLISION_EXAMPLE_CONFIG` can select another
explicit example descriptor. Both default to this project's paths.

## Edit the arena

The editor is always available in Train and Test. The default layout is a deliberate
arrangement of four offset, orthogonal partitions, distinct from the assignment's
map. The exploration seed affects the robot's walk, not obstacle placement.

- **+ Wall / + Post:** choose the tool, then tap a position in the arena.
- **Select:** drag an obstacle. Selected walls have endpoint handles for
  length/orientation changes; posts have a radius handle.
- **Remove selected obstacle** (trash icon): removes an interior obstacle; Delete/Backspace works when
  the canvas is focused. Boundary walls are fixed.
- **Reset layout** (under **⋯**): restores the default arrangement. **Use layout** loads
  the selected collection's recorded geometry.
- In Test, use **Select** to drag the single goal. It
  has no resize handles, cannot be added/deleted, and is absent in Train.

Obstacle changes pause and restart the robot, clearing its trajectory/recurrent
history. The goal's position is preserved when valid; placing a new obstacle over
it is rejected. Geometry is saved in localStorage, survives refreshes/mode changes,
and is sent unchanged with each new collection request. An in-progress collection
keeps its starting snapshot even if the preview is edited. Up to 40 interior
obstacles are supported. Pointer controls support mouse and touch.

## Collect data from the CLI

CLI commands intentionally write to the given filesystem output directory and
are separate from browser-owned experiments. `--world layout.json` accepts a
validated edited arena with fixed boundary walls.

```bash
# Small run to inspect the format:
npm run collect -- --out data/example --episodes 20 --seed 42

# Larger run with the full sequence of physics steps:
npm run collect -- --out data/run-001 --episodes 1000 --seed 42

# Smaller files, retaining consecutive action-level observations for RNNs:
npm run collect -- --out data/run-002 --episodes 1000 --seed 43 --decisions-only

# Fixed 0.1-second decisions, with a longer episode limit:
npm run collect -- --out data/rnn-001 --episodes 1000 --max-actions 200

npm run collect -- --help
```

The output directory **must not already exist**. Files are streamed with bounded
buffers. The final manifest reports sample counts, action coverage, class balance,
and elapsed wall-clock time. Use only runs with `"status": "complete"`; an
interrupted or failed run retains a collecting manifest and may have partial files.

`--episodes` counts episodes, **not samples**. An episode ends at its first
collision or after `--max-actions` decisions (default 50). Each decision lasts up
to five physics steps (0.1 seconds), ending early on collision. Thus 1,000 default episodes have
an upper bound of 250,000 step records. Full step JSONL is intentionally
rich and can be large; use `--decisions-only` when action-level sequences suffice.
The action interval is fixed in collection and testing; there is no UI, CLI, or
API setting for it. Increase the maximum actions per episode to collect longer
histories (the default 50 actions allows up to five simulated seconds).

`--persistence 0.65` retains the previous action with probability 0.65; otherwise
it draws uniformly from all 11 actions. This gives correlated wandering with
exploration of the extreme turns. It is a new seeded policy, not the original
Perlin-noise implementation. No collision-avoidance model filters exploration.

## Physics and conventions

- A circular robot (radius 18) moves among static capsule-shaped walls.
- Coordinates use **x right, y up**, with angles in radians counterclockwise.
  The canvas performs the y flip once for all displayed objects.
- Physics uses fixed `dt = 0.02` seconds. Rendering speed never enters physics.
- Acceleration and linear drag update velocity; speed and heading turn rate
  are capped. Velocity persists across action changes, making motion history useful.
- New runs use acceleration 140 units/s² and a heading turn-rate cap of 3 rad/s,
  doubled from the initial defaults. The speed cap remains 100 units/s. Decisions
  occur every 5 steps (0.1 seconds), down from 100 steps (2 seconds) initially.
- Actions `-5..5` specify force directions at 18-degree offsets from the heading
  **when the decision begins**. The resulting world-space force direction is held
  fixed for the action, even while the robot rotates.
- Five center-origin rays at `[66, 33, 0, -33, -66]` degrees measure distance to
  wall surfaces, capped at 180. They return raw distances, not normalized features.
- A swept circle checks the full displacement against thick walls and rounded
  endpoints, preventing tunneling through thin walls. On first contact, position
  stops at the contact point and the episode terminates. There is no collision
  response/bounce solver; final velocity is the incoming velocity for diagnostics.
- A reset starts at a seeded, collision-free position and random heading with
  zero velocity. Reset is explicit and never occurs inside a recorded transition.

Collect fresh data and retrain to use the one-second prediction target. Older
files remain on disk, but training and model loading reject incompatible timing
or target semantics. Manifest, decision, balance-plan, and model schemas are now
version 2; step and episode records keep their unchanged version-1 formats. The
physics equations still use `circle-inertia-v1`. A recurrent history of eight
decisions covers at most 0.8 simulated seconds, independently of the one-second
prediction horizon.

## Execute for 0.1 seconds, predict for 1 second

Before each real action, the collector forks the current simulator state,
including momentum, heading, tick, physics, and wall geometry. In that isolated
copy it applies the chosen **fixed world-space force direction** for up to 50
steps (one second), stopping at first collision. It does not recompute force
from the robot's changing heading. Only the chosen action is labeled; collection
does not generate eleven alternate actions or fictional recurrent history.

The real robot executes five steps (0.1 seconds) and then chooses again. The
hypothetical outcome never moves the real robot, consumes its policy RNG, or
terminates its episode. Even the last action of a truncated real episode receives
a full lookahead rollout. The extra physics work makes collection slower than
recording only the actual five steps; hypothetical transitions are not exported.

A label answers "Would holding this force for one second collide?", not "Will the
actual controller collide after making several further decisions?" This gives
earlier warning but can conservatively reject forces that would be safe with
subsequent corrections. It does not guarantee an escape route.

This is **new physics**, versioned as `circle-inertia-v1`, with a new barrier map.
It is not a Pymunk port. The old checkpoints and scaler must not be assumed valid;
train on this environment's data. Configuration and version are stored in every run's manifest. Fixed-map geometry
is stored there too; randomized geometry is stored in each episode record. Reproducibility is tested within this runtime;
different JavaScript engines may have tiny floating-point differences.

## Optional CLI layout experiments

`--random-layouts` generates four seeded internal barriers per episode, varying
position, orientation, and length within separated regions. Boundary walls stay
fixed. Obstacles remain stationary during an episode; this is variation between
scenarios, not moving-obstacle simulation. This remains an optional CLI research utility. The browser uses its edited
layout exclusively and has no random-layout checkbox.

Each randomized episode records `layoutId`, `layoutSeed`, and its complete `world`;
step/decision records carry `layoutId`. Within one collected run, layouts are
unique to episodes and cannot cross train/validation/test boundaries. A fixed-map
run evaluates new trajectories on the same geometry. Randomized evaluation checks
new configurations within this generator's family, not arbitrary unseen worlds.

## Files and label meaning

| File | Unit of one record | Purpose |
| --- | --- | --- |
| `manifest.json` | One run | Version, seed, physics, policy, label definitions, completion and counts |
| `episodes.jsonl` | One episode | Episode seed, split, initial/final observations, counts and end reason |
| `steps.jsonl` | One physics step | Fine-grained temporal data; omitted with `--decisions-only` |
| `decisions.jsonl` | One executed action | Action-level sequence and collision prediction targets |
| `decisions.csv` | One executed action | Flat feedforward training view, with IDs and split metadata |

JSONL means one complete JSON object per line. Every step and decision includes:

- `episodeId` and `split` (`train`, `validation`, or `test`). IDs are local to a
  run: include the run identity when combining datasets.
- `decisionIndex`, beginning at zero in each episode.
- `observation`: **before** the step/action, with `step`, `time`, five `sensors`,
  and ground-truth `state` (position, velocity, heading, angular velocity).
- `control`: the chosen action and the world-space force direction actually held.
- `nextObservation`: **after** the step/action, before any reset.
- `collision`, `terminated`, and `truncated`.

**Step labels:** `collision` means contact during this particular physics step.
It is not the training target. `stepInAction` records
the phase within the held action.

**Actual decision outcome:** `collision` means contact during the five executed
steps. `plannedSteps` is 5; `elapsedSteps` is shorter if collision ended the action
early. `elapsedTime = elapsedSteps * dt`. `terminated` means actual collision;
`truncated` means the real episode reached its action limit without contact.

**Training target:** `lookaheadCollision` means contact in the independent
50-step rollout. `lookaheadSteps` is 50; `lookaheadElapsedSteps` is the first
collision step, or 50 if safe throughout. A positive lookahead target can have
`collision = false` and the real episode can continue. Actual collisions imply a
positive lookahead target because the first five steps share the same physics.

The manifest and model carry `prediction = {target: "lookaheadCollision", steps:
50, continuation: "hold-world-force"}`. The loader checks this contract and the
consistency of actual/predicted outcomes. `counts.collisions` counts actual episode
collisions; `counts.lookaheadCollisions` counts positive training targets.

CSV columns are:

```text
episode_id,split,decision_index,start_step,start_time,
sensor_0,sensor_1,sensor_2,sensor_3,sensor_4,action,collision,
elapsed_steps,terminated,truncated,
lookahead_collision,lookahead_steps,lookahead_elapsed_steps
```

The CSV has a header and metadata, unlike the original assignment's seven-column
CSV. Select the five sensor columns plus `action` for six-feature models and
`lookahead_collision` as the target. Neither future outcome belongs in the input.

## Balance and load data

The trainer reads a completed run's `decisions.jsonl`. It validates consecutive
indices, observation continuity, episode boundaries, action horizons, and split
isolation. It loads decision records into memory; fine-grained `steps.jsonl` is
optional research data and is not consumed by this trainer.

```bash
# Create a reproducible selection of training targets:
../.venv/bin/python -m training.balance \
  --dataset data/run-002 --out data/run-002/balance-half.json \
  --collision-fraction 0.5 --seed 42
```

Balancing retains all positive `lookaheadCollision` targets and randomly
undersamples negative lookahead targets, regardless of the actual five-step outcome.
It writes a selection plan rather than deleting CSV/JSONL rows. An omitted target
can still appear in another target's history. The plan records a dataset
fingerprint, seed, requested fraction, and achieved fraction (rounded to whole
samples). Requesting a fraction below the existing collision rate fails because
this tool only removes safe targets. Validation/test targets are never balanced.
50/50 is an experiment setting, not a guaranteed optimum; undersampling also
changes the meaning of the raw output scores, so use validation to select the
operating threshold and assess false positives/missed collisions.

`training.data.RunData`, `DecisionDataset`, and `make_loaders` provide the data
API. Each window contains up to `sequence_length` chronological `(sensors, action)`
pairs ending at the decision being predicted. The label is that decision's
`lookaheadCollision` outcome. Early, short histories are retained, right padded in a batch,
and accompanied by lengths; packed recurrent processing excludes padding.

```python
from training.data import RunData, make_loaders

run = RunData("data/run-002")
loaders = make_loaders(run, sequence_length=8, batch_size=64, seed=42)
inputs, lengths, labels = next(iter(loaders["train"]))
# inputs: float32 [batch, time, 6]; labels: float32 [batch]
# lengths: number of valid time steps in each right-padded window
```

Whole episodes are assigned 80% train, 10% validation, 10% test, rounding evaluation
counts down. At least ten episodes are needed to populate all three splits.
Training shuffles complete windows. Neither windows nor recurrent memory cross
an episode boundary. Features are five distances divided by the physical sensor
range plus action divided by 5; this normalization needs no fitted statistics.
Current-action `nextObservation`, elapsed time, lookahead fields, collision label, and ground-truth
velocity are excluded from inputs to avoid future-information leakage and keep
all architectures on the same six sensor/action features.

## Construct and train a network

```python
import torch
from training.network_factory import make_network
from training.train import TrainConfig, train

torch.manual_seed(42)  # Programmatic callers control initialization.
model = make_network("gru", input_size=6, output_size=1,
                     hidden_size=32, num_layers=1)
model, report = train(
    "data/run-002", model, "models/gru-example",
    config=TrainConfig(epochs=30, sequence_length=8),
    balance_plan="data/run-002/balance-half.json",
)
```

`make_network("feedforward", hidden_sizes=[32, 16])` builds ReLU hidden layers.
`make_network("rnn", hidden_size=32, num_layers=1)` builds a tanh RNN; `"gru"`
builds a GRU. All use a final linear layer and return logits. The factory accepts
a `NetworkSpec` too, with general input/output dimensions; this collision trainer
and browser require six inputs and one output. Up to four hidden layers of 512
units are supported, though large recurrent windows/models can slow browser playback.

The training API accepts a dataset path or `RunData`, a factory network, a new
output directory, and optional `TrainConfig`. Defaults are Adam, learning rate
0.001, 30 epochs, batch 64, history 8, early-stopping patience 8, weight decay 0,
seed 42, and collision cutoff 0.25. SGD is also available. Feedforward uses only
the current decision, regardless of configured history. For automatic balancing,
set `collision_fraction` instead of passing a balance plan.

```bash
../.venv/bin/python -m training.train \
  --dataset data/run-002 --out models/gru-001 \
  --network gru --hidden 32 --layers 1 \
  --config examples/train-config.json \
  --balance-plan data/run-002/balance-half.json

# An unbalanced feedforward baseline, using default training settings:
../.venv/bin/python -m training.train \
  --dataset data/run-002 --out models/feedforward-001 \
  --network feedforward --hidden 32,16
```

The CLI also accepts `--network-config` with a JSON `NetworkSpec`, overriding
architecture/dimension flags. See `examples/train-config.json` for hyperparameters;
omitted fields use defaults. Outputs must not already exist.

Training runs on CPU with `BCEWithLogitsLoss`, matching `[batch]` logit/label shapes,
mini-batch optimizer updates, and gradient clipping. Validation is evaluated after
each epoch. A deep copy of the lowest-validation-loss weights is restored before
final test evaluation; even the initial epoch-0 checkpoint is retained if no
training epoch improves it. Early stopping counts epochs without improvement.
The test split is evaluated once per training run, after model selection.

| Output | Contents |
| --- | --- |
| `model.json` | Browser weights, architecture, normalization, history length, threshold, physics, prediction semantics, and evaluation summary |
| `weights.pt` | PyTorch state dict for the selected checkpoint |
| `report.json` | Epoch history, hyperparameters, selected epoch, dataset fingerprint, split counts, and held-out metrics |
| `balance.json` | Selection plan when balancing was requested through `TrainConfig` |

False positive rate is rejected safe examples / all safe examples. Missed
collision rate is accepted collision examples / all collision examples. Rates
use the one-second lookahead labels, not actual five-step outcomes, and are
`null`/N/A when their denominator is zero. `positiveTargets` counts true lookahead
labels in that evaluation split. These are diagnostics at the saved
threshold, not additional loss terms. Compare architectures on the same run and
use validation for tuning; repeatedly choosing models from test results would
make that test set part of model selection.

## Browser inference and goal seeking

The exported JSON contains ordinary numeric weights; no Python or model runtime
is required in the browser. `BrowserNetwork` implements the same feedforward,
tanh RNN, and PyTorch GRU equations. Tests compare its logits against PyTorch for
all three network types, including multilayer recurrent networks.

Recurrent training and inference both use **finite history windows**, starting
hidden state at zero for each window. For each proposed action, the controller
combines actual prior decisions with current sensors and that candidate action.
Evaluating eleven candidates does not mutate history. Only the action actually
executed is committed, and history is cleared on episode reset.

At each action boundary, the controller:

1. Predicts one-second held-force collision scores for actions -5 through +5
   and accepts scores below the saved cutoff.
2. Among accepted actions, chooses the force direction closest to the goal,
   breaking ties with lower collision score.
3. If none pass, chooses the lowest-scoring action, breaking ties toward the goal.
4. Holds the selected force for five steps (0.1 seconds), then observes and repeats.
   Test mode uses the trained network; it does not run hypothetical physics rollouts.

Reaching within 35 units chooses another goal. Collision resets the episode;
10,000 steps without reaching a goal also trigger a timeout/reset. This is a
local steering policy, not a global path planner. It can oscillate, get stuck,
or collide. The shared example provides a trained starting point, not a guarantee
of navigation on every edited layout.
The simulator exposes those outcomes so the architectures can be compared.

## Structure and verification

```text
src/geometry.ts        Ray/capsule intersections and swept collision geometry
src/simulation.ts      Fixed-step environment, sensors, fixed/random layouts
src/prediction.ts      Isolated held-force rollout labels and prediction contract
scripts/restore-example.mjs  Restore and verify the bundled shared example
examples/bundle/      Compressed original collection, balance plan, and model
src/seeds.ts           Automatic seeds for browser experiments
src/random.ts          Seeded randomness and persistent random walk
src/collector.ts       Streaming episode/decision generator
src/types.ts           Versioned simulation and dataset contracts
src/cli.ts             JSONL/CSV export and completion manifest
src/network.ts         Browser inference and model validation
src/goal-seeking.ts     Candidate scoring, goal selection, recurrent history
src/browser.ts         Canvas, playback, goal interaction
src/editor.ts          Pointer tools for obstacle editing and goal movement
src/layout.ts          Persisted layout and shared geometry validation
src/storage.ts         IndexedDB artifact/file storage and workspace identity
src/artifacts.ts       Shared browser/server artifact and job contracts
src/workflow.ts        Train/Test controls, jobs, and model loading
src/server.ts          Scoped temporary jobs, gzip file transfer, pinned example
training/data.py       Validated datasets and sequence loaders
training/balance.py    Reproducible training-target selection
training/network_factory.py  Parameterized feedforward/RNN/GRU construction
training/train.py      Training, evaluation, checkpoint selection, export
```

```bash
npm test
.venv/bin/python -B -m unittest discover -s training/tests -v
```

Tests cover geometry, inertia, reproducibility, continuity, isolated lookahead
labels, preserved real trajectories, incompatible model rejection, label horizons, split
isolation, output protection, randomized layout recording, candidate histories,
goal resets, sequence-preserving balance plans, padding, best checkpoints, and
Python/JavaScript prediction parity. A separate Chromium smoke check exercised
collection → balancing → GRU training → goal seeking, JSON export/import with the
API unavailable, desktop/mobile editing, goal restrictions, two-browser isolation,
local persistence, result cleanup, and uploaded-dataset training. API tests also
check cross-workspace rejection and recovery after a server restart. Bundle tests
verify restoration in an empty directory and protection of existing files. These small test runs verify the
pipeline; they do not establish navigation performance.
