import { Simulation, SIMULATOR_VERSION, validateDecisionTiming } from "./simulation.js";
import type { SimulationConfig } from "./types.js";
import type { PredictionSpec } from "./types.js";
import { validatePrediction } from "./prediction.js";

export interface ModelArtifact {
  schemaVersion: 2;
  prediction: PredictionSpec;
  simulator: string;
  network: { kind: "feedforward" | "rnn" | "gru"; input_size: number; output_size: number;
    hidden_sizes: number[]; hidden_size: number; num_layers: number };
  sequenceLength: number;
  threshold: number;
  normalization: { sensorRange: number; actionScale: number };
  simulation: SimulationConfig;
  weights: Record<string, number[] | number[][]>;
  training?: Record<string, unknown>;
}

const sigmoid = (x: number): number => x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));

/** Small, auditable CPU inference implementation; parameters are plain JSON. */
export class BrowserNetwork {
  readonly artifact: ModelArtifact;
  constructor(value: unknown) {
    if (!value || typeof value !== "object") throw new Error("Invalid model JSON");
    const model = value as ModelArtifact;
    const spec = model.network;
    validatePrediction(model.prediction);
    if (model.schemaVersion !== 2 || model.simulator !== SIMULATOR_VERSION || !spec
      || !["feedforward", "rnn", "gru"].includes(spec.kind) || spec.input_size !== 6 || spec.output_size !== 1) {
      throw new Error("Expected a Collision Lab model with six inputs and one output");
    }
    const dimension = (n: number): boolean => Number.isInteger(n) && n > 0 && n <= 512;
    if (!Array.isArray(spec.hidden_sizes) || spec.hidden_sizes.length < 1 || spec.hidden_sizes.length > 4
      || !spec.hidden_sizes.every(dimension) || !dimension(spec.hidden_size)
      || !Number.isInteger(spec.num_layers) || spec.num_layers < 1 || spec.num_layers > 4
      || !Number.isInteger(model.sequenceLength) || model.sequenceLength < 1 || model.sequenceLength > 256
      || !Number.isFinite(model.threshold) || model.threshold <= 0 || model.threshold >= 1) {
      throw new Error("Invalid network dimensions, sequence length, or threshold");
    }
    if (!model.simulation || !model.normalization || !model.weights
      || !Number.isFinite(model.normalization.sensorRange) || model.normalization.sensorRange <= 0
      || model.normalization.sensorRange !== model.simulation.sensorRange || model.normalization.actionScale !== 5
      || (spec.kind === "feedforward" && model.sequenceLength !== 1)) throw new Error("Invalid model normalization or simulation metadata");
    for (const key of ["dt", "actionSteps", "robotRadius", "acceleration", "drag", "maxSpeed", "maxTurnRate", "sensorRange", "actionAngle"] as const) {
      if (!Number.isFinite(model.simulation[key])) throw new Error(`Missing simulation parameter ${key}`);
    }
    if (!Array.isArray(model.simulation.sensorAngles) || model.simulation.sensorAngles.length !== 5
      || !model.simulation.sensorAngles.every(Number.isFinite)) throw new Error("Expected five finite sensor angles");
    new Simulation(model.simulation); // Validate the same configuration that inference will run.
    validateDecisionTiming(model.simulation);
    this.artifact = structuredClone(model);
    const vector = (key: string, length: number): void => {
      const v = this.artifact.weights[key];
      if (!Array.isArray(v) || v.length !== length || !v.every(x => typeof x === "number" && Number.isFinite(x))) throw new Error(`Invalid weight vector ${key}`);
    };
    const matrix = (key: string, rows: number, columns: number): void => {
      const m = this.artifact.weights[key];
      if (!Array.isArray(m) || m.length !== rows || !m.every(row => Array.isArray(row)
        && row.length === columns && row.every(x => typeof x === "number" && Number.isFinite(x)))) throw new Error(`Invalid weight matrix ${key}`);
    };
    let width = 6;
    if (spec.kind === "feedforward") {
      for (const [layer, size] of spec.hidden_sizes.entries()) {
        matrix(`layers.${layer}.weight`, size, width); vector(`layers.${layer}.bias`, size); width = size;
      }
    } else {
      for (let layer = 0; layer < spec.num_layers; layer++) {
        const gates = spec.kind === "gru" ? 3 : 1;
        matrix(`recurrent.weight_ih_l${layer}`, gates * spec.hidden_size, width);
        matrix(`recurrent.weight_hh_l${layer}`, gates * spec.hidden_size, spec.hidden_size);
        vector(`recurrent.bias_ih_l${layer}`, gates * spec.hidden_size);
        vector(`recurrent.bias_hh_l${layer}`, gates * spec.hidden_size);
        width = spec.hidden_size;
      }
    }
    matrix("output.weight", 1, width); vector("output.bias", 1);
  }

  features(sensors: number[], action: number): number[] {
    if (sensors.length !== 5 || !sensors.every(Number.isFinite) || !Number.isInteger(action) || action < -5 || action > 5) {
      throw new Error("Prediction requires five sensor values and an action from -5 to 5");
    }
    return [...sensors.map(value => value / this.artifact.normalization.sensorRange), action / 5];
  }

  private linear(input: number[], weight: string, bias: string): number[] {
    const weights = this.artifact.weights[weight] as number[][];
    const biases = this.artifact.weights[bias] as number[];
    return weights.map((row, i) => row.reduce((sum, value, j) => sum + value * input[j]!, biases[i]!));
  }

  /** Normalized chronological inputs. Hidden state always starts at zero per window. */
  logit(features: number[][]): number {
    if (!features.length || features.some(row => row.length !== 6 || !row.every(Number.isFinite))) throw new Error("Invalid feature window");
    const sequence = features.slice(-this.artifact.sequenceLength);
    const spec = this.artifact.network;
    let result: number[] = [];
    if (spec.kind === "feedforward") {
      result = sequence.at(-1)!;
      for (let layer = 0; layer < spec.hidden_sizes.length; layer++) {
        result = this.linear(result, `layers.${layer}.weight`, `layers.${layer}.bias`).map(x => Math.max(0, x));
      }
    } else {
      const states = Array.from({ length: spec.num_layers }, () => Array(spec.hidden_size).fill(0) as number[]);
      for (const observation of sequence) {
        let input = observation;
        for (let layer = 0; layer < spec.num_layers; layer++) {
          const previous = states[layer]!;
          const x = this.linear(input, `recurrent.weight_ih_l${layer}`, `recurrent.bias_ih_l${layer}`);
          const h = this.linear(previous, `recurrent.weight_hh_l${layer}`, `recurrent.bias_hh_l${layer}`);
          const size = spec.hidden_size;
          states[layer] = previous.map((old, j) => {
            if (spec.kind === "rnn") return Math.tanh(x[j]! + h[j]!);
            // PyTorch gate order: reset, update, new. Reset applies after the
            // hidden-state linear transform, including its new-gate bias.
            const reset = sigmoid(x[j]! + h[j]!);
            const update = sigmoid(x[size + j]! + h[size + j]!);
            const candidate = Math.tanh(x[2 * size + j]! + reset * h[2 * size + j]!);
            return (1 - update) * candidate + update * old;
          });
          input = states[layer]!;
        }
        result = states.at(-1)!;
      }
    }
    return this.linear(result, "output.weight", "output.bias")[0]!;
  }

  predict(history: number[][], sensors: number[], action: number): number {
    return sigmoid(this.logit([...history, this.features(sensors, action)]));
  }
}
