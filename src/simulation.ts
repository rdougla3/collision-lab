import { distanceToSegment, rayWall, wrapAngle } from "./geometry.js";
import { Random } from "./random.js";
import type { Control, Observation, RobotState, SimulationConfig, StepResult, Vec2, World } from "./types.js";

export const SIMULATOR_VERSION = "circle-inertia-v1";

export const DEFAULT_CONFIG: Readonly<SimulationConfig> = {
  dt: 0.02,
  actionSteps: 5,
  robotRadius: 18,
  acceleration: 140,
  drag: 0.8,
  maxSpeed: 100,
  maxTurnRate: 3,
  sensorRange: 180,
  sensorAngles: [66, 33, 0, -33, -66].map(degrees => degrees * Math.PI / 180),
  actionAngle: Math.PI / 10,
};

/** Collection and deployed models share one fixed decision cadence. */
export function validateDecisionTiming(config: SimulationConfig): void {
  if (config.actionSteps !== DEFAULT_CONFIG.actionSteps || config.dt !== DEFAULT_CONFIG.dt) {
    throw new Error("This simulator requires 5 steps per action at 0.02 seconds per step. Collect fresh data and retrain older models.");
  }
}

/** Static capsule walls. Coordinates: x right, y up; angles counterclockwise. */
export function createWorld(): World {
  const segments: [string, number, number, number, number][] = [
    ["south", 0, 0, 1080, 0], ["east", 1080, 0, 1080, 900],
    ["north", 1080, 900, 0, 900], ["west", 0, 900, 0, 0],
    ["west-barrier", 300, 220, 300, 650],
    ["upper-barrier", 550, 690, 870, 690],
    ["lower-barrier", 570, 230, 870, 230],
    ["east-barrier", 780, 370, 780, 550],
  ];
  return {
    id: "gallery-v1", width: 1080, height: 900,
    walls: segments.map(([id, x1, y1, x2, y2]) => ({
      id, start: { x: x1, y: y1 }, end: { x: x2, y: y2 }, radius: 8,
    })),
  };
}

/** Layout randomness is independent of the robot/policy RNG. One seed = one map. */
export function createRandomWorld(seed: number): World {
  const random = new Random(seed);
  const world = createWorld();
  world.id = `random-barriers-v1-${seed}`;
  world.walls = world.walls.slice(0, 4);
  // Separated placement cells leave gaps between barriers and avoid enclosed rooms.
  const cells = [[250, 250], [780, 250], [250, 650], [780, 650]];
  for (const [index, cell] of cells.entries()) {
    const x = cell[0]! + (random.next() - .5) * 100;
    const y = cell[1]! + (random.next() - .5) * 90;
    const angle = random.next() * Math.PI;
    const halfLength = 65 + random.next() * 55;
    const dx = Math.cos(angle) * halfLength;
    const dy = Math.sin(angle) * halfLength;
    world.walls.push({ id: `obstacle-${index}`, start: { x: x - dx, y: y - dy }, end: { x: x + dx, y: y + dy }, radius: 8 });
  }
  return world;
}

export class Simulation {
  readonly config: SimulationConfig;
  readonly world: World;
  private robot!: RobotState;
  private tick = 0;
  private terminal = false;

  constructor(config: Partial<SimulationConfig> = {}, world = createWorld()) {
    this.config = { ...DEFAULT_CONFIG, ...config, sensorAngles: [...(config.sensorAngles ?? DEFAULT_CONFIG.sensorAngles)] };
    for (const key of ["dt", "robotRadius", "acceleration", "maxSpeed", "maxTurnRate", "sensorRange", "actionAngle"] as const) {
      if (!Number.isFinite(this.config[key]) || this.config[key] <= 0) throw new Error(`${key} must be positive and finite.`);
    }
    if (!Number.isFinite(this.config.drag) || this.config.drag < 0) throw new Error("drag must be finite and nonnegative.");
    if (!Number.isSafeInteger(this.config.actionSteps) || this.config.actionSteps <= 0) throw new Error("actionSteps must be a positive integer.");
    if (this.config.sensorAngles.length !== 5 || !this.config.sensorAngles.every(Number.isFinite)) throw new Error("Exactly five finite sensor angles are required.");
    if (!Number.isFinite(world.width) || !Number.isFinite(world.height) || world.width <= 0 || world.height <= 0) throw new Error("World dimensions must be positive and finite.");
    for (const wall of world.walls) {
      if (![wall.start.x, wall.start.y, wall.end.x, wall.end.y, wall.radius].every(Number.isFinite) || wall.radius < 0) {
        throw new Error("Wall coordinates and radii must be finite, with nonnegative radii.");
      }
    }
    this.world = structuredClone(world);
    this.reset(new Random(0));
  }

  isFree(position: Vec2, clearance = 0): boolean {
    const radius = this.config.robotRadius + clearance;
    return position.x > radius && position.x < this.world.width - radius
      && position.y > radius && position.y < this.world.height - radius
      && this.world.walls.every(wall => distanceToSegment(position, wall.start, wall.end) > radius + wall.radius);
  }

  /** Reset is explicit: collision observations are never replaced by a respawn. */
  reset(random: Random, state?: RobotState): Observation {
    if (state) {
      if (![state.position.x, state.position.y, state.velocity.x, state.velocity.y, state.heading, state.angularVelocity].every(Number.isFinite)
        || !this.isFree(state.position) || Math.hypot(state.velocity.x, state.velocity.y) > this.config.maxSpeed) {
        throw new Error("Reset state must be finite, collision-free, and within maxSpeed.");
      }
      this.robot = structuredClone(state);
    } else {
      let position: Vec2 | undefined;
      for (let attempt = 0; attempt < 10000; attempt++) {
        const candidate = { x: random.next() * this.world.width, y: random.next() * this.world.height };
        if (this.isFree(candidate, 12)) { position = candidate; break; }
      }
      if (!position) throw new Error("Could not find a free spawn location.");
      this.robot = { position, velocity: { x: 0, y: 0 }, heading: random.next() * Math.PI * 2 - Math.PI, angularVelocity: 0 };
    }
    this.tick = 0;
    this.terminal = false;
    return this.observe();
  }

  observe(): Observation {
    const state = structuredClone(this.robot);
    return {
      step: this.tick, time: this.tick * this.config.dt, state,
      sensors: this.config.sensorAngles.map(offset => {
        const angle = state.heading + offset;
        const direction = { x: Math.cos(angle), y: Math.sin(angle) };
        return this.world.walls.reduce((distance, wall) => Math.min(distance, rayWall(state.position, direction, wall)), this.config.sensorRange);
      }),
    };
  }

  /** Independent physics state for hypothetical rollouts; no real episode/RNG changes. */
  fork(): Simulation {
    const copy = new Simulation(this.config, this.world);
    copy.robot = structuredClone(this.robot);
    copy.tick = this.tick;
    copy.terminal = this.terminal;
    return copy;
  }

  controlFor(action: number): Control {
    if (!Number.isInteger(action) || action < -5 || action > 5) throw new Error("Action must be an integer from -5 to 5.");
    const angle = this.robot.heading + action * this.config.actionAngle;
    return { action, direction: { x: Math.cos(angle), y: Math.sin(angle) } };
  }

  step(control: Control): StepResult {
    if (this.terminal) throw new Error("Episode ended in a collision; call reset before stepping again.");
    if (!Number.isInteger(control.action) || control.action < -5 || control.action > 5
      || !Number.isFinite(control.direction.x) || !Number.isFinite(control.direction.y)
      || Math.abs(Math.hypot(control.direction.x, control.direction.y) - 1) > 1e-6) {
      throw new Error("Control must contain a valid action and unit direction.");
    }
    const observation = this.observe();
    const { dt, acceleration, drag, maxSpeed, maxTurnRate, robotRadius } = this.config;
    const velocity = {
      x: this.robot.velocity.x + (acceleration * control.direction.x - drag * this.robot.velocity.x) * dt,
      y: this.robot.velocity.y + (acceleration * control.direction.y - drag * this.robot.velocity.y) * dt,
    };
    const speed = Math.hypot(velocity.x, velocity.y);
    if (speed > maxSpeed) { velocity.x *= maxSpeed / speed; velocity.y *= maxSpeed / speed; }
    let headingChange = 0;
    if (speed > 0.01) {
      headingChange = Math.max(-maxTurnRate * dt, Math.min(maxTurnRate * dt,
        wrapAngle(Math.atan2(velocity.y, velocity.x) - this.robot.heading)));
    }

    const distance = Math.hypot(velocity.x, velocity.y) * dt;
    let travel = distance;
    let collisionWall: string | null = null;
    if (distance > 0) {
      const direction = { x: velocity.x * dt / distance, y: velocity.y * dt / distance };
      // Swept circle collision: catches a wall even if a single step crosses it.
      for (const wall of this.world.walls) {
        const hit = rayWall(this.robot.position, direction, wall, robotRadius);
        if (hit <= travel) { travel = hit; collisionWall = wall.id; }
      }
      this.robot.position.x += direction.x * travel;
      this.robot.position.y += direction.y * travel;
    }
    this.robot.velocity = velocity;
    this.robot.heading = wrapAngle(this.robot.heading + headingChange);
    this.robot.angularVelocity = headingChange / dt;
    this.tick++;
    this.terminal = collisionWall !== null;
    return {
      observation, nextObservation: this.observe(), control: structuredClone(control),
      collision: this.terminal, collisionWall,
    };
  }
}
