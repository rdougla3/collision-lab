export interface Vec2 { x: number; y: number }
export interface Wall { id: string; start: Vec2; end: Vec2; radius: number }
export interface World { id: string; width: number; height: number; walls: Wall[] }

export interface SimulationConfig {
  dt: number;
  actionSteps: number;
  robotRadius: number;
  acceleration: number;
  drag: number;
  maxSpeed: number;
  maxTurnRate: number;
  sensorRange: number;
  sensorAngles: number[];
  actionAngle: number;
}

export interface RobotState {
  position: Vec2;
  velocity: Vec2;
  heading: number;
  angularVelocity: number;
}

export interface Observation {
  step: number;
  time: number;
  sensors: number[];
  /** Ground truth for debugging or optional richer models; not sensor inputs. */
  state: RobotState;
}

export interface Control {
  action: number;
  /** Direction is fixed in world coordinates for the whole action interval. */
  direction: Vec2;
}

export interface StepResult {
  observation: Observation;
  nextObservation: Observation;
  control: Control;
  collision: boolean;
  collisionWall: string | null;
}

export type Split = "train" | "validation" | "test";
export type EpisodeEnd = "collision" | "action_limit";

export interface StepRecord extends StepResult {
  schemaVersion: 1;
  episodeId: number;
  layoutId?: string;
  split: Split;
  decisionIndex: number;
  stepInAction: number;
  terminated: boolean;
  truncated: boolean;
}

export interface DecisionRecord {
  schemaVersion: 2;
  episodeId: number;
  layoutId?: string;
  split: Split;
  decisionIndex: number;
  observation: Observation;
  nextObservation: Observation;
  control: Control;
  plannedSteps: number;
  elapsedSteps: number;
  elapsedTime: number;
  /** True iff any step during this executed action collided. */
  collision: boolean;
  /** Counterfactual outcome if this world-space force continues for the prediction horizon. */
  lookaheadCollision: boolean;
  lookaheadSteps: number;
  lookaheadElapsedSteps: number;
  terminated: boolean;
  truncated: boolean;
}

export interface PredictionSpec {
  target: "lookaheadCollision";
  steps: number;
  continuation: "hold-world-force";
}

export interface EpisodeRecord {
  schemaVersion: 1;
  episodeId: number;
  layoutId?: string;
  layoutSeed?: number;
  world?: World;
  seed: number;
  split: Split;
  steps: number;
  decisions: number;
  endReason: EpisodeEnd;
  initialObservation: Observation;
  finalObservation: Observation;
}

export type CollectionEvent =
  | { type: "step"; record: StepRecord }
  | { type: "decision"; record: DecisionRecord }
  | { type: "episode"; record: EpisodeRecord };
