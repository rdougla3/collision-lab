import { Random, RandomWalk } from "./random.js";
import { lookahead } from "./prediction.js";
import { createRandomWorld, Simulation, validateDecisionTiming } from "./simulation.js";
import type { CollectionEvent, EpisodeEnd, SimulationConfig, Split, World } from "./types.js";

export interface CollectionOptions {
  seed: number;
  episodes: number;
  maxActions: number;
  persistence: number;
  config?: Partial<SimulationConfig>;
  world?: World;
  randomLayouts?: boolean;
}

export const DEFAULT_COLLECTION: CollectionOptions = {
  seed: 42, episodes: 1000, maxActions: 50, persistence: 0.65,
};

export function validateOptions(options: CollectionOptions): void {
  new Random(options.seed);
  new RandomWalk(new Random(0), options.persistence);
  for (const key of ["episodes", "maxActions"] as const) {
    if (!Number.isSafeInteger(options[key]) || options[key] <= 0) throw new Error(`${key} must be a positive integer.`);
  }
  if (options.world && options.randomLayouts) throw new Error("Choose either a custom world or random layouts.");
}

/** Shuffle episode IDs, not rows; round toward train for very small runs. */
export function episodeSplits(episodes: number, seed: number): Split[] {
  const evaluationCount = Math.floor(episodes * 0.1);
  const splits: Split[] = Array.from({ length: episodes }, (_, i) =>
    i < evaluationCount ? "test" : i < 2 * evaluationCount ? "validation" : "train");
  const random = new Random((seed ^ 0xa511e9b3) >>> 0);
  for (let i = splits.length - 1; i > 0; i--) {
    const j = random.integer(0, i);
    [splits[i], splits[j]] = [splits[j]!, splits[i]!];
  }
  return splits;
}

/** Streaming generator: memory usage is independent of total dataset size. */
export function* collect(options: CollectionOptions): Generator<CollectionEvent> {
  validateOptions(options);
  let simulation = new Simulation(options.config, options.world);
  validateDecisionTiming(simulation.config);
  const splits = episodeSplits(options.episodes, options.seed);
  for (let episodeId = 0; episodeId < options.episodes; episodeId++) {
    const seed = (options.seed + Math.imul(episodeId, 0x9e3779b9)) >>> 0;
    const layoutSeed = (seed ^ 0x7f4a7c15) >>> 0;
    if (options.randomLayouts) simulation = new Simulation(options.config, createRandomWorld(layoutSeed));
    const layoutId = simulation.world.id;
    const random = new Random(seed);
    const initialObservation = simulation.reset(random);
    const walk = new RandomWalk(random, options.persistence);
    const split = splits[episodeId]!;
    let steps = 0;
    let decisions = 0;
    let endReason: EpisodeEnd = "action_limit";

    for (let decisionIndex = 0; decisionIndex < options.maxActions; decisionIndex++) {
      const observation = simulation.observe();
      const control = simulation.controlFor(walk.next());
      const prediction = lookahead(simulation, control);
      let collision = false;
      let elapsedSteps = 0;
      for (let stepInAction = 0; stepInAction < simulation.config.actionSteps; stepInAction++) {
        const result = simulation.step(control);
        steps++;
        elapsedSteps++;
        collision = result.collision;
        const truncated = !collision && decisionIndex === options.maxActions - 1
          && stepInAction === simulation.config.actionSteps - 1;
        yield { type: "step", record: {
          schemaVersion: 1, episodeId, layoutId, split, decisionIndex, stepInAction,
          ...result, terminated: collision, truncated,
        } };
        if (collision) break;
      }
      decisions++;
      yield { type: "decision", record: {
        schemaVersion: 2, episodeId, layoutId, split, decisionIndex, observation,
        nextObservation: simulation.observe(), control,
        plannedSteps: simulation.config.actionSteps, elapsedSteps,
        elapsedTime: elapsedSteps * simulation.config.dt,
        ...prediction, collision, terminated: collision,
        truncated: !collision && decisionIndex === options.maxActions - 1,
      } };
      if (collision) { endReason = "collision"; break; }
    }
    yield { type: "episode", record: {
      schemaVersion: 1, episodeId, seed, layoutId,
      ...(options.randomLayouts ? { layoutSeed, world: simulation.world } : {}),
      split, steps, decisions, endReason,
      initialObservation, finalObservation: simulation.observe(),
    } };
  }
}
