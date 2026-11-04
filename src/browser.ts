import { collect, DEFAULT_COLLECTION } from "./collector.js";
import { DEFAULT_CONFIG } from "./simulation.js";
import { BrowserNetwork } from "./network.js";
import { GoalSeeking } from "./goal-seeking.js";
import type { CollectionEvent, EpisodeRecord, SimulationConfig, StepRecord } from "./types.js";
import "./workflow.js";
import { getLayout } from "./layout.js";
import { ArenaEditor } from "./editor.js";
import { freshSeed } from "./seeds.js";

function element<T extends HTMLElement>(id: string): T {
  const result = document.getElementById(id);
  if (!result) throw new Error(`Missing element ${id}`);
  return result as T;
}

const canvas = element<HTMLCanvasElement>("world");
const context = canvas.getContext("2d");
if (!context) throw new Error("Canvas 2D is unavailable.");
const ctx: CanvasRenderingContext2D = context;
let world = getLayout();
let editor: ArenaEditor;
let config: SimulationConfig = structuredClone(DEFAULT_CONFIG);
let mode: "train" | "test" = "train";
let model: BrowserNetwork | undefined;
let goalSeeking: GoalSeeking | undefined;
let lastScoreKey = "";
const play = element<HTMLButtonElement>("play");
const stepButton = element<HTMLButtonElement>("step");
const speedInput = element<HTMLInputElement>("speed");
const sensorValues = element<HTMLDivElement>("sensors");
for (let index = 0; index < 5; index++) sensorValues.append(document.createElement("span"));

let iterator: Generator<CollectionEvent>;
let current: StepRecord | undefined;
let episodeSteps: StepRecord[] = [];
let lastTrail: StepRecord[] = [];
let completed = 0;
let collisions = 0;
let running = false;
let finished = false;
let accumulator = 0;

function setRunning(value: boolean): void {
  const missingModel = mode === "test" && !goalSeeking;
  running = value && !finished && !missingModel;
  play.textContent = running ? "Pause" : "Run";
  play.disabled = finished || missingModel;
  stepButton.disabled = running || finished || missingModel;
  element("run-status").textContent = missingModel ? "Load a model to run" : finished ? "Run complete" : running ? "Running" : "Paused";
  accumulator = 0;
}

function finishEpisode(episode: EpisodeRecord): void {
  lastTrail = episodeSteps;
  episodeSteps = [];
  completed++;
  if (episode.endReason === "collision") collisions++;
}

/** Consume bookkeeping events until the next physics transition is reached. */
function advance(): void {
  if (mode === "test") {
    if (!goalSeeking) return;
    const result = goalSeeking.step();
    if (!current || current.episodeId !== result.episodeId) episodeSteps = [];
    current = result;
    episodeSteps.push(result);
    if (episodeSteps.length > 5000) episodeSteps.shift();
    collisions = goalSeeking.collisions;
    completed = goalSeeking.collisions + goalSeeking.timeouts;
    return;
  }
  while (!finished) {
    const event = iterator.next();
    if (event.done) {
      finished = true;
      setRunning(false);
      return;
    }
    if (event.value.type === "step") {
      current = event.value.record;
      episodeSteps.push(current);
      // Consume the terminal summary before the next episode begins.
      if (current.terminated || current.truncated) {
        const decision = iterator.next();
        const episode = iterator.next();
        if (decision.done || decision.value.type !== "decision" || episode.done || episode.value.type !== "episode") {
          throw new Error("Collector ended an episode without its summary records.");
        }
        finishEpisode(episode.value.record);
      }
      return;
    }
    if (event.value.type === "episode") finishEpisode(event.value.record);
  }
}

function restart(preserveGoal = false): void {
  const previousGoal = preserveGoal ? goalSeeking?.goal : undefined;
  const seed = freshSeed();
  element("error").textContent = "";
  const maxActions = Number(element<HTMLInputElement>("collect-actions").value);
  if (mode === "train" && (!Number.isInteger(maxActions) || maxActions < 1 || maxActions > 1000)) {
    element("error").textContent = "Maximum actions must be an integer between 1 and 1000."; return;
  }
  world = getLayout();
  if (mode === "test") {
    goalSeeking = model ? new GoalSeeking(model, seed, world) : undefined;
    config = model?.artifact.simulation ?? structuredClone(DEFAULT_CONFIG);
    if (previousGoal && goalSeeking?.simulation.isFree(previousGoal, 10)) goalSeeking.setGoal(previousGoal);
  } else {
    goalSeeking = undefined;
    config = structuredClone(DEFAULT_CONFIG);
    iterator = collect({ ...DEFAULT_COLLECTION, seed, episodes: 1000, maxActions, config, world });
  }
  current = undefined;
  episodeSteps = [];
  lastTrail = [];
  completed = 0;
  collisions = 0;
  finished = false;
  lastScoreKey = "";
  element("action-scores").replaceChildren();
  element("decision-reason").textContent = "Scores appear when a model chooses an action.";
  for (const value of sensorValues.children) value.textContent = "—";
  element("goals").textContent = "0";
  element("timeouts").textContent = "0";
  element("episode").textContent = "—";
  element("time").textContent = "—";
  element("action").textContent = "—";
  element("completed").textContent = "0";
  element("collisions").textContent = "0";
  editor?.setGoalEnabled(mode === "test" && !!goalSeeking);
  setRunning(false);
  advance();
  draw();
}

function arrow(x: number, y: number, dx: number, dy: number, color: string): void {
  const angle = Math.atan2(dy, dx);
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + dx, y + dy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x + dx, y + dy);
  ctx.lineTo(x + dx - 9 * Math.cos(angle - .5), y + dy - 9 * Math.sin(angle - .5));
  ctx.lineTo(x + dx - 9 * Math.cos(angle + .5), y + dy - 9 * Math.sin(angle + .5));
  ctx.closePath(); ctx.fill();
}

function draw(): void {
  const scale = Math.min(canvas.width / world.width, canvas.height / world.height);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#17211b";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  // Apply the y flip once to every world object, sensor, and vector.
  ctx.setTransform(scale, 0, 0, -scale, 0, world.height * scale);
  ctx.strokeStyle = "#243328"; ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x <= world.width; x += 60) { ctx.moveTo(x, 0); ctx.lineTo(x, world.height); }
  for (let y = 0; y <= world.height; y += 60) { ctx.moveTo(0, y); ctx.lineTo(world.width, y); }
  ctx.stroke();
  ctx.lineCap = "round";
  for (const wall of world.walls) {
    ctx.strokeStyle = "#87988b"; ctx.lineWidth = wall.radius * 2;
    ctx.beginPath();
    if (wall.start.x === wall.end.x && wall.start.y === wall.end.y) {
      ctx.fillStyle = "#87988b"; ctx.arc(wall.start.x, wall.start.y, wall.radius, 0, Math.PI * 2); ctx.fill();
    } else { ctx.moveTo(wall.start.x, wall.start.y); ctx.lineTo(wall.end.x, wall.end.y); ctx.stroke(); }
  }
  if (goalSeeking) {
    const goal = goalSeeking.goal;
    ctx.strokeStyle = "#ffc878"; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(goal.x, goal.y, 24, 0, 2 * Math.PI); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(goal.x - 12, goal.y); ctx.lineTo(goal.x + 12, goal.y);
    ctx.moveTo(goal.x, goal.y - 12); ctx.lineTo(goal.x, goal.y + 12); ctx.stroke();
  }
  if (!current) { editor?.draw(ctx); return; }
  ctx.strokeStyle = "#547b5e"; ctx.lineWidth = 2;
  ctx.beginPath();
  const trail = mode === "train" && (current.terminated || current.truncated) ? lastTrail : episodeSteps;
  for (const [index, step] of trail.entries()) {
    const { x, y } = step.observation.state.position;
    if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  const observation = current.nextObservation;
  const { position, heading, velocity } = observation.state;
  for (const [index, distance] of observation.sensors.entries()) {
    const angle = heading + config.sensorAngles[index]!;
    const x = position.x + Math.cos(angle) * distance;
    const y = position.y + Math.sin(angle) * distance;
    ctx.strokeStyle = "#bbec9a"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(position.x, position.y); ctx.lineTo(x, y); ctx.stroke();
    ctx.fillStyle = "#bbec9a"; ctx.beginPath(); ctx.arc(x, y, 3, 0, 2 * Math.PI); ctx.fill();
    sensorValues.children[index]!.textContent = distance.toFixed(0);
  }
  arrow(position.x, position.y, current.control.direction.x * 65, current.control.direction.y * 65, "#8bbcff");
  if (Math.hypot(velocity.x, velocity.y) > .1) arrow(position.x, position.y, velocity.x, velocity.y, "#ffb66d");
  ctx.fillStyle = current.collision ? "#ff8e78" : "#e8ede9";
  ctx.beginPath(); ctx.arc(position.x, position.y, config.robotRadius, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = "#17211b"; ctx.lineWidth = 4;
  ctx.beginPath(); ctx.moveTo(position.x, position.y);
  ctx.lineTo(position.x + Math.cos(heading) * 15, position.y + Math.sin(heading) * 15); ctx.stroke();
  element("episode").textContent = String(current.episodeId + 1);
  element("time").textContent = `${observation.time.toFixed(2)} s`;
  element("action").textContent = `${current.control.action > 0 ? "+" : ""}${current.control.action} / ${(current.control.action * config.actionAngle * 180 / Math.PI).toFixed(0)}°`;
  element("completed").textContent = String(completed);
  element("collisions").textContent = String(collisions);
  if (goalSeeking) {
    element("goals").textContent = String(goalSeeking.goalsReached);
    element("timeouts").textContent = String(goalSeeking.timeouts);
    element("decision-reason").textContent = goalSeeking.reason;
    const key = `${current.episodeId}/${current.decisionIndex}`;
    if (key !== lastScoreKey) {
      lastScoreKey = key;
      const rows = goalSeeking.scores.map(score => {
        const row = document.createElement("div"); row.className = "score-row";
        row.classList.toggle("selected", score.action === current!.control.action);
        const label = document.createElement("span"); label.textContent = String(score.action);
        const bar = document.createElement("div"); bar.className = "score-bar";
        const fill = document.createElement("span"); fill.style.width = `${100 * score.risk}%`;
        fill.classList.toggle("rejected", score.risk >= goalSeeking!.model.artifact.threshold);
        bar.append(fill);
        const risk = document.createElement("span"); risk.textContent = `${(100 * score.risk).toFixed(1)}%`;
        row.append(label, bar, risk); return row;
      });
      element("action-scores").replaceChildren(...rows);
    }
  }
  editor?.draw(ctx);
}

play.addEventListener("click", () => setRunning(!running));
stepButton.addEventListener("click", () => { advance(); draw(); });
element("restart").addEventListener("click", () => restart());
document.addEventListener("lab:layout", () => restart(true));
document.addEventListener("lab:mode", event => {
  mode = (event as CustomEvent<"train" | "test">).detail;
  restart();
});
document.addEventListener("lab:model", event => {
  model = (event as CustomEvent<BrowserNetwork>).detail;
  if (mode === "test") restart();
});
editor = new ArenaEditor(canvas, {
  pause: () => setRunning(false),
  preview: next => { world = next; draw(); },
  draw,
  goal: () => mode === "test" ? goalSeeking?.goal : undefined,
  moveGoal: point => { if (goalSeeking && mode === "test") { goalSeeking.setGoal(point); element("error").textContent = ""; draw(); } },
});
speedInput.addEventListener("input", () => { element("speed-value").textContent = `${speedInput.value}×`; });
document.addEventListener("visibilitychange", () => { if (document.hidden) setRunning(false); });
new ResizeObserver(() => {
  const width = canvas.getBoundingClientRect().width;
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(width * pixelRatio));
  canvas.height = Math.round(canvas.width * world.height / world.width);
  draw();
}).observe(canvas);

restart();
let previous = performance.now();
function frame(now: number): void {
  const elapsed = Math.min((now - previous) / 1000, .1);
  previous = now;
  if (running) {
    accumulator += elapsed * Number(speedInput.value);
    let steps = 0;
    while (accumulator >= config.dt && running && steps < 250) {
      advance(); accumulator -= config.dt; steps++;
    }
    draw();
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
