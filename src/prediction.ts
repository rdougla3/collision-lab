import type { Simulation } from "./simulation.js";
import type { Control, PredictionSpec } from "./types.js";

export const PREDICTION: Readonly<PredictionSpec> = Object.freeze({
  target: "lookaheadCollision", steps: 50, continuation: "hold-world-force",
});

export function validatePrediction(value: unknown): void {
  const spec = value as Partial<PredictionSpec> | null | undefined;
  if (spec?.target !== PREDICTION.target || spec?.steps !== PREDICTION.steps
    || spec?.continuation !== PREDICTION.continuation) {
    throw new Error("Expected one-second lookahead labels (50 steps, held world-space force). Collect fresh data and retrain older models.");
  }
}

/** Label one candidate without advancing the real simulation or adding fictional history. */
export function lookahead(simulation: Simulation, control: Control): {
  lookaheadCollision: boolean; lookaheadSteps: number; lookaheadElapsedSteps: number;
} {
  const shadow = simulation.fork();
  for (let step = 1; step <= PREDICTION.steps; step++) {
    if (shadow.step(control).collision) {
      return { lookaheadCollision: true, lookaheadSteps: PREDICTION.steps, lookaheadElapsedSteps: step };
    }
  }
  return { lookaheadCollision: false, lookaheadSteps: PREDICTION.steps, lookaheadElapsedSteps: PREDICTION.steps };
}
