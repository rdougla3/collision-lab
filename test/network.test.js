import test from 'node:test';
import assert from 'node:assert/strict';
import { BrowserNetwork } from '../dist/network.js';
import { GoalSeeking } from '../dist/goal-seeking.js';
import { DEFAULT_CONFIG, createRandomWorld } from '../dist/simulation.js';
import { Random } from '../dist/random.js';
import { collect } from '../dist/collector.js';
import { PREDICTION } from '../dist/prediction.js';

function artifact() {
  return { schemaVersion: 2, prediction: { ...PREDICTION }, simulator: 'circle-inertia-v1',
    network: { kind: 'feedforward', input_size: 6, output_size: 1, hidden_sizes: [2], hidden_size: 2, num_layers: 1 },
    sequenceLength: 1, threshold: .25, normalization: { sensorRange: 180, actionScale: 5 },
    simulation: { ...DEFAULT_CONFIG },
    weights: { 'layers.0.weight': [[1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, -1]], 'layers.0.bias': [0, 0],
      'output.weight': [[1, -1]], 'output.bias': [0] } };
}

test('inference normalizes inputs, preserves candidate history and rejects malformed models', () => {
  const model = new BrowserNetwork(artifact());
  assert.deepEqual(model.features([180, 90, 0, 180, 0], -5), [1, .5, 0, 1, 0, -1]);
  const history = [[0, 0, 0, 0, 0, 0]];
  const before = structuredClone(history);
  const first = model.predict(history, [180, 90, 0, 180, 0], -5);
  model.predict(history, [180, 90, 0, 180, 0], 5);
  assert.equal(model.predict(history, [180, 90, 0, 180, 0], -5), first);
  assert.deepEqual(history, before);
  const invalid = artifact(); invalid.weights['output.weight'] = [[NaN, 0]];
  assert.throws(() => new BrowserNetwork(invalid), /weight/);
  const missingAngles = artifact(); delete missingAngles.simulation.sensorAngles;
  assert.throws(() => new BrowserNetwork(missingAngles), /sensor angles/);
});

test('random layouts are seeded and do not overlap across dataset splits', () => {
  assert.deepEqual(createRandomWorld(11), createRandomWorld(11));
  assert.notDeepEqual(createRandomWorld(11), createRandomWorld(12));
  const episodes = [...collect({ seed: 42, episodes: 20, maxActions: 1, persistence: .65,
    randomLayouts: true })].filter(e => e.type === 'episode').map(e => e.record);
  assert.equal(new Set(episodes.map(e => e.layoutId)).size, 20);
  assert.ok(episodes.every(e => e.world.id === e.layoutId));
  assert.deepEqual(episodes.map(e => createRandomWorld(e.layoutSeed)), episodes.map(e => e.world));
});

test('models with a different decision interval require retraining', () => {
  for (const actionSteps of [3, 25, 100]) {
    const saved = artifact(); saved.simulation.actionSteps = actionSteps;
    assert.throws(() => new BrowserNetwork(saved), /5 steps per action/);
  }
  const saved = artifact(); saved.simulation.dt = .04;
  assert.throws(() => new BrowserNetwork(saved), /5 steps per action/);
});

test('goal seeking reconsiders every five steps, counts goals and resets after collision', () => {
  const model = new BrowserNetwork(artifact());
  const histories = [];
  model.predict = (history, sensors, action) => { histories.push(structuredClone(history)); return action === 0 ? 0 : 1; };
  const runner = new GoalSeeking(model, 42, createRandomWorld(42));
  let position = runner.simulation.observe().state.position;
  runner.setGoal({ x: position.x + 1, y: position.y });
  runner.step();
  assert.equal(runner.goalsReached, 1);
  assert.equal(runner.simulation.config.actionSteps, 5);
  const phases = [0];
  for (let i = 0; i < 5; i++) phases.push(runner.step().stepInAction);
  assert.deepEqual(phases, [0, 1, 2, 3, 4, 0]);
  assert.equal(histories.length, 22); // Eleven candidates, two decisions.
  assert.ok(histories.slice(0, 11).every(h => h.length === 0));
  assert.ok(histories.slice(11).every(h => JSON.stringify(h) === JSON.stringify(histories[11])));
  // New runner near the east wall: action 0 will hit it in one step.
  const crash = new GoalSeeking(model, 3);
  crash.simulation.reset(new Random(0), { position: { x: 1053.9, y: 450 }, velocity: { x: 100, y: 0 }, heading: 0, angularVelocity: 0 });
  const result = crash.step();
  assert.equal(result.collision, true);
  assert.equal(crash.collisions, 1);
  const next = crash.step();
  assert.equal(next.episodeId, 1);
  assert.equal(next.observation.step, 0);
  assert.deepEqual(histories.at(-1), []);
});

test('older models cannot reinterpret short-horizon scores as lookahead predictions', () => {
  const missing = artifact(); delete missing.prediction;
  assert.throws(() => new BrowserNetwork(missing), /one-second lookahead/);
  for (const prediction of [{ ...PREDICTION, steps: 5 }, { ...PREDICTION, continuation: 'repeat-relative-action' }]) {
    assert.throws(() => new BrowserNetwork({ ...artifact(), prediction }), /one-second lookahead/);
  }
});
