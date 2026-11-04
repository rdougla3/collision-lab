import { readFileSync, closeSync, mkdirSync, openSync, renameSync, writeFileSync, writeSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";
import { collect, DEFAULT_COLLECTION, validateOptions, type CollectionOptions } from "./collector.js";
import { DEFAULT_CONFIG, Simulation, SIMULATOR_VERSION } from "./simulation.js";
import type { DecisionRecord } from "./types.js";
import { validateLayout } from "./layout.js";
import { PREDICTION } from "./prediction.js";

class BufferedFile {
  private readonly fd: number;
  private buffer = "";
  constructor(path: string) { this.fd = openSync(path, "wx"); }
  write(text: string): void {
    this.buffer += text;
    if (this.buffer.length >= 65536) this.flush();
  }
  private flush(): void {
    const bytes = Buffer.from(this.buffer);
    let offset = 0;
    while (offset < bytes.length) offset += writeSync(this.fd, bytes, offset);
    this.buffer = "";
  }
  close(): void { try { this.flush(); } finally { closeSync(this.fd); } }
}

function csvRow(record: DecisionRecord): string {
  return [record.episodeId, record.split, record.decisionIndex, record.observation.step,
    record.observation.time, ...record.observation.sensors, record.control.action,
    Number(record.collision), record.elapsedSteps, Number(record.terminated), Number(record.truncated),
    Number(record.lookaheadCollision), record.lookaheadSteps, record.lookaheadElapsedSteps].join(",") + "\n";
}

function main(): void {
  const { values } = parseArgs({ options: {
    world: { type: "string" }, out: { type: "string" }, episodes: { type: "string", default: String(DEFAULT_COLLECTION.episodes) },
    "max-actions": { type: "string", default: String(DEFAULT_COLLECTION.maxActions) },
    seed: { type: "string", default: "42" },
    persistence: { type: "string", default: "0.65" },
    "decisions-only": { type: "boolean", default: false }, help: { type: "boolean", short: "h" },
    "random-layouts": { type: "boolean", default: false },
  } });
  if (values.help) {
    console.log(`Collect random-walk training data (no window or real-time delays).

npm run collect -- --out data/run-001 [options]

--out PATH           New output directory; existing directories are refused
--episodes N         Number of independent episodes (default: 1000)
--max-actions N      End an episode after N decisions if it has not collided (50)
--seed N             Reproducible unsigned 32-bit seed (42)
--persistence P      Probability of retaining the previous action (0.65)
--decisions-only     Omit per-physics-step JSONL; decision sequences remain intact
--world PATH         Use an edited layout JSON file (fixed throughout the run)
--random-layouts     Generate a unique seeded obstacle layout for every episode

Episodes are split 80/10/10 into train/validation/test (rounded toward train).
Each action lasts ${DEFAULT_CONFIG.actionSteps} physics steps (${DEFAULT_CONFIG.actionSteps * DEFAULT_CONFIG.dt} seconds); this interval is fixed.
Training labels predict collision if the same world-space force continues for ${PREDICTION.steps} steps (one second).
Actual five-step collisions are recorded separately. Hypothetical rollouts never change the real trajectory.
Raw sequences are never shuffled or undersampled.`);
    return;
  }
  if (!values.out) throw new Error("--out is required. Use --help for options.");
  const options: CollectionOptions = {
    seed: Number(values.seed), episodes: Number(values.episodes),
    maxActions: Number(values["max-actions"]), persistence: Number(values.persistence),
    randomLayouts: values["random-layouts"],
    world: values.world ? validateLayout(JSON.parse(readFileSync(values.world, "utf8"))) : undefined,
  };
  validateOptions(options);
  const simulation = new Simulation(options.config, options.world);
  const output = resolve(values.out);
  mkdirSync(dirname(output), { recursive: true });
  mkdirSync(output); // Never overwrite an existing run, even an empty one.

  const manifest = {
    schemaVersion: 2, simulator: SIMULATOR_VERSION, status: "collecting", prediction: PREDICTION,
    createdAt: new Date().toISOString(), options, simulation: simulation.config, world: simulation.world,
    coordinates: "x right, y up, angles in radians counterclockwise; distances in simulation units",
    policy: "persistent random walk with uniform action exploration",
    splits: options.randomLayouts ? "unique layouts per episode; disjoint layouts across splits" : "whole episodes on one fixed layout; not an unseen-layout evaluation",
    labels: {
      steps: "collision on this physics step",
      decisions: "any collision during this executed action (up to actionSteps * dt seconds)",
      training: "lookaheadCollision: collision within 50 steps if the candidate world-space force is held; evaluated in an independent simulation copy",
      termination: "collision; no automatic reset inside a transition",
      truncation: "max-actions reached after a complete action",
    },
    files: {
      steps: values["decisions-only"] ? null : "steps.jsonl",
      decisions: "decisions.jsonl", episodes: "episodes.jsonl", csv: "decisions.csv",
    },
  };
  writeFileSync(resolve(output, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n", { flag: "wx" });
  const opened: BufferedFile[] = [];
  const open = (name: string): BufferedFile => {
    const file = new BufferedFile(resolve(output, name)); opened.push(file); return file;
  };
  const counts = { steps: 0, decisions: 0, episodes: 0, collisions: 0, lookaheadCollisions: 0, truncated: 0 };
  const actions: Record<string, number> = Object.fromEntries(Array.from({ length: 11 }, (_, i) => [String(i - 5), 0]));
  const splitCounts = { train: 0, validation: 0, test: 0 };
  const start = performance.now();
  try {
    const steps = values["decisions-only"] ? null : open("steps.jsonl");
    const decisions = open("decisions.jsonl");
    const episodes = open("episodes.jsonl");
    const csv = open("decisions.csv");
    csv.write("episode_id,split,decision_index,start_step,start_time,sensor_0,sensor_1,sensor_2,sensor_3,sensor_4,action,collision,elapsed_steps,terminated,truncated,lookahead_collision,lookahead_steps,lookahead_elapsed_steps\n");
    for (const event of collect(options)) {
      if (event.type === "step") {
        counts.steps++;
        steps?.write(JSON.stringify(event.record) + "\n");
      } else if (event.type === "decision") {
        counts.decisions++;
        if (event.record.lookaheadCollision) counts.lookaheadCollisions++;
        actions[String(event.record.control.action)]!++;
        decisions.write(JSON.stringify(event.record) + "\n");
        csv.write(csvRow(event.record));
      } else {
        counts.episodes++;
        splitCounts[event.record.split]++;
        if (event.record.endReason === "collision") counts.collisions++;
        else counts.truncated++;
        episodes.write(JSON.stringify(event.record) + "\n");
        if (counts.episodes % 100 === 0 || counts.episodes === options.episodes) {
          console.log(`${counts.episodes}/${options.episodes} episodes; ${counts.decisions} actions; ${counts.collisions} collisions`);
        }
      }
    }
  } finally {
    for (const file of opened) file.close();
  }
  const elapsedSeconds = (performance.now() - start) / 1000;
  const completed = { ...manifest, status: "complete", counts, splitCounts, actionCounts: actions, elapsedSeconds };
  writeFileSync(resolve(output, "manifest.tmp.json"), JSON.stringify(completed, null, 2) + "\n", { flag: "wx" });
  renameSync(resolve(output, "manifest.tmp.json"), resolve(output, "manifest.json"));
  console.log(`Saved to ${output}\n${counts.steps} real steps in ${elapsedSeconds.toFixed(2)}s. Lookahead collision targets: ${(100 * counts.lookaheadCollisions / counts.decisions).toFixed(1)}%; actual collisions: ${counts.collisions}.`);
  if (options.episodes < 10) console.log("Fewer than 10 episodes: no validation/test episodes were allocated.");
}

try { main(); } catch (error) {
  console.error(`Collection failed: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
}
