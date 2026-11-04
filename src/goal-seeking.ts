import { BrowserNetwork } from "./network.js";
import { wrapAngle } from "./geometry.js";
import { ACTIONS, Random } from "./random.js";
import { createWorld, Simulation } from "./simulation.js";
import type { Control, StepRecord, Vec2, World } from "./types.js";

export class GoalSeeking {
  readonly simulation: Simulation;
  goal: Vec2;
  scores: { action: number; risk: number }[] = [];
  reason = "";
  goalsReached = 0;
  collisions = 0;
  timeouts = 0;
  episode = 0;
  private readonly random: Random;
  private history: number[][] = [];
  private control!: Control;
  private pendingFeatures: number[] = [];
  private remaining = 0;
  private decisions = 0;
  private needsReset = false;
  private stepsSinceGoal = 0;

  constructor(readonly model: BrowserNetwork, seed = 42, world: World = createWorld()) {
    this.random = new Random(seed);
    this.simulation = new Simulation(model.artifact.simulation, world);
    this.simulation.reset(this.random);
    this.goal = this.chooseGoal();
  }

  private chooseGoal(): Vec2 {
    const position = this.simulation.observe().state.position;
    for (let attempt = 0; attempt < 10000; attempt++) {
      const point = { x: this.random.next() * this.simulation.world.width, y: this.random.next() * this.simulation.world.height };
      if (this.simulation.isFree(point, 25) && Math.hypot(point.x - position.x, point.y - position.y) > 150) return point;
    }
    throw new Error("Could not place a goal with sufficient clearance");
  }

  setGoal(point: Vec2): void {
    if (!this.simulation.isFree(point, 10)) throw new Error("Place the goal in open space away from walls");
    this.goal = { ...point };
    this.stepsSinceGoal = 0;
  }

  private decide(): void {
    const observation = this.simulation.observe();
    const desired = Math.atan2(this.goal.y - observation.state.position.y, this.goal.x - observation.state.position.x);
    this.scores = ACTIONS.map(action => ({ action, risk: this.model.predict(this.history, observation.sensors, action) }));
    const available = this.scores.filter(candidate => candidate.risk < this.model.artifact.threshold);
    const headingError = (action: number): number => Math.abs(wrapAngle(observation.state.heading + action * this.simulation.config.actionAngle - desired));
    const ranked = [...(available.length ? available : this.scores)].sort((a, b) =>
      available.length ? headingError(a.action) - headingError(b.action) || a.risk - b.risk
        : a.risk - b.risk || headingError(a.action) - headingError(b.action));
    const chosen = ranked[0]!;
    this.reason = available.length ? "Allowed direction closest to goal" : "All actions rejected: trying lowest predicted risk";
    this.control = this.simulation.controlFor(chosen.action);
    this.pendingFeatures = this.model.features(observation.sensors, chosen.action);
    this.remaining = this.simulation.config.actionSteps;
    this.decisions++;
  }

  step(): StepRecord {
    if (this.needsReset) {
      this.simulation.reset(this.random);
      this.history = [];
      this.remaining = 0;
      this.decisions = 0;
      this.stepsSinceGoal = 0;
      this.episode++;
      this.goal = this.chooseGoal();
      this.needsReset = false;
    }
    if (this.remaining === 0) this.decide();
    const stepInAction = this.simulation.config.actionSteps - this.remaining;
    const result = this.simulation.step(this.control);
    this.remaining--;
    this.stepsSinceGoal++;
    const position = result.nextObservation.state.position;
    if (!result.collision && Math.hypot(position.x - this.goal.x, position.y - this.goal.y) < 35) {
      this.goalsReached++;
      this.goal = this.chooseGoal();
      this.stepsSinceGoal = 0;
    }
    const truncated = !result.collision && this.stepsSinceGoal >= 10000;
    if (result.collision) this.collisions++;
    if (truncated) this.timeouts++;
    this.needsReset = result.collision || truncated;
    if (this.remaining === 0 && !this.needsReset) {
      // Commit only the action actually executed. Candidate scoring never mutates history.
      this.history.push(this.pendingFeatures);
      this.history = this.history.slice(-Math.max(1, this.model.artifact.sequenceLength - 1));
    }
    return {
      schemaVersion: 1, episodeId: this.episode, layoutId: this.simulation.world.id,
      split: "test", decisionIndex: this.decisions - 1, stepInAction,
      ...result, terminated: result.collision, truncated,
    };
  }
}
