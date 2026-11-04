import test from 'node:test';
import assert from 'node:assert/strict';
import { rayWall, distanceToSegment } from '../dist/geometry.js';
import { Random, RandomWalk } from '../dist/random.js';
import { Simulation } from '../dist/simulation.js';
import { collect, episodeSplits } from '../dist/collector.js';
import { lookahead } from '../dist/prediction.js';

function box(size = 200) {
  return { id: 'test-box', width: size, height: size, walls: [
    { id: 'west', start: { x: 0, y: 0 }, end: { x: 0, y: size }, radius: 1 },
    { id: 'east', start: { x: size, y: 0 }, end: { x: size, y: size }, radius: 1 },
    { id: 'south', start: { x: 0, y: 0 }, end: { x: size, y: 0 }, radius: 1 },
    { id: 'north', start: { x: 0, y: size }, end: { x: size, y: size }, radius: 1 },
  ] };
}
const state = (x = 100, y = 100) => ({
  position: { x, y }, velocity: { x: 0, y: 0 }, heading: 0, angularVelocity: 0,
});
const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-8, `${actual} != ${expected}`);

test('ray distances include wall thickness, round caps, parallel rays, and expansion', () => {
  const wall = { id: 'wall', start: { x: 10, y: -5 }, end: { x: 10, y: 5 }, radius: 2 };
  near(rayWall({ x: 0, y: 0 }, { x: 1, y: 0 }, wall), 8);
  near(rayWall({ x: 0, y: 0 }, { x: 1, y: 0 }, wall, 3), 5);
  near(rayWall({ x: 10, y: 10 }, { x: 0, y: -1 }, wall), 3);
  near(rayWall({ x: 0, y: 7 }, { x: 1, y: 0 }, wall), 10); // tangent to end cap
  assert.equal(rayWall({ x: 0, y: 10 }, { x: 1, y: 0 }, wall), Infinity);
  assert.equal(rayWall({ x: 0, y: 0 }, { x: -1, y: 0 }, wall), Infinity);
  near(distanceToSegment({ x: 13, y: 9 }, wall.start, wall.end), 5);
});

test('sensors measure before/after movement and are detached snapshots', () => {
  const sim = new Simulation({ sensorRange: 300 }, box());
  sim.reset(new Random(1), state());
  const observation = sim.observe();
  near(observation.sensors[2], 99);
  const result = sim.step(sim.controlFor(0));
  assert.deepEqual(result.observation, observation);
  assert.equal(result.nextObservation.step, 1);
  assert.equal(result.nextObservation.time, .02);
  assert.ok(result.nextObservation.sensors[2] < 99);
  result.nextObservation.state.position.x = -999;
  result.nextObservation.sensors[2] = -999;
  assert.ok(sim.observe().state.position.x > 100);
  assert.ok(sim.observe().sensors[2] > 0);
});

test('swept robot collision catches crossing a thin wall and requires explicit reset', () => {
  const world = box();
  world.walls.push({ id: 'thin', start: { x: 110, y: 30 }, end: { x: 110, y: 170 }, radius: 1 });
  const sim = new Simulation({ dt: 1, acceleration: 100, maxSpeed: 500, robotRadius: 5 }, world);
  sim.reset(new Random(1), state(80, 100));
  const result = sim.step(sim.controlFor(0));
  assert.equal(result.collision, true);
  assert.equal(result.collisionWall, 'thin');
  near(result.nextObservation.state.position.x, 104);
  assert.equal(result.observation.state.position.x, 80);
  assert.throws(() => sim.step(sim.controlFor(0)), /reset/);
  sim.reset(new Random(2));
  assert.equal(sim.observe().step, 0);
});

test('actions retain world force during their horizon and preserve prior momentum', () => {
  const sim = new Simulation({}, box(2000));
  const initial = state(1000, 1000);
  initial.velocity.x = 30;
  sim.reset(new Random(0), initial);
  const control = sim.controlFor(5);
  near(control.direction.x, 0);
  near(control.direction.y, 1);
  const first = sim.step(control);
  const second = sim.step(control);
  assert.ok(first.nextObservation.state.velocity.x > 0);
  assert.ok(first.nextObservation.state.velocity.y > 0);
  assert.deepEqual(first.control, second.control);
  assert.notEqual(second.nextObservation.state.heading, 0);
  assert.throws(() => sim.controlFor(6), /Action/);
  assert.throws(() => new Simulation({ dt: 0 }), /dt/);
});

test('seeded policy repeats deterministically and explores all actions', () => {
  const a = new RandomWalk(new Random(17));
  const b = new RandomWalk(new Random(17));
  const actions = Array.from({ length: 10000 }, () => a.next());
  assert.deepEqual(actions, Array.from({ length: 10000 }, () => b.next()));
  assert.equal(new Set(actions).size, 11);
  assert.ok(actions.filter((action, i) => i && action === actions[i - 1]).length > 6000);
});

test('stronger steering redirects moving robots faster while respecting the speed cap', () => {
  const initial = { ...state(1000, 1000), velocity: { x: 80, y: 0 } };
  const old = new Simulation({ acceleration: 70, maxTurnRate: 1.5 }, box(2000));
  const updated = new Simulation({}, box(2000));
  for (const sim of [old, updated]) {
    sim.reset(new Random(0), initial);
    const control = sim.controlFor(5);
    for (let i = 0; i < 25; i++) sim.step(control);
  }
  const before = old.observe().state;
  const after = updated.observe().state;
  assert.ok(Math.atan2(after.velocity.y, after.velocity.x) > Math.atan2(before.velocity.y, before.velocity.x));
  assert.ok(after.heading > before.heading);
  assert.ok(Math.hypot(after.velocity.x, after.velocity.y) <= updated.config.maxSpeed);
});

test('episodes retain temporal continuity, action labels, terminal outcomes and whole-episode splits', () => {
  const options = { seed: 42, episodes: 20, maxActions: 48, persistence: .65, world: box() };
  const events = [...collect(options)];
  assert.deepEqual(events, [...collect(options)]);
  const summaries = events.filter(e => e.type === 'episode').map(e => e.record);
  assert.equal(summaries.length, 20);
  assert.deepEqual(['train', 'validation', 'test'].map(split => summaries.filter(e => e.split === split).length), [16, 2, 2]);
  assert.ok(summaries.some(e => e.endReason === 'collision'));
  for (const episode of summaries) {
    const rows = events.filter(e => e.record.episodeId === episode.episodeId);
    assert.ok(rows.every(e => e.record.split === episode.split));
    const steps = rows.filter(e => e.type === 'step').map(e => e.record);
    const decisions = rows.filter(e => e.type === 'decision').map(e => e.record);
    assert.equal(steps.length, episode.steps);
    assert.equal(decisions.length, episode.decisions);
    assert.equal(steps[0].observation.step, 0);
    assert.deepEqual(steps[0].observation.state.velocity, { x: 0, y: 0 });
    assert.deepEqual(steps[0].observation, episode.initialObservation);
    assert.deepEqual(steps.at(-1).nextObservation, episode.finalObservation);
    for (let i = 1; i < steps.length; i++) {
      assert.deepEqual(steps[i - 1].nextObservation, steps[i].observation);
      assert.equal(steps[i - 1].terminated || steps[i - 1].truncated, false);
    }
    assert.equal(steps.at(-1).terminated || steps.at(-1).truncated, true);
    for (const decision of decisions) {
      const actionSteps = steps.filter(s => s.decisionIndex === decision.decisionIndex);
      assert.deepEqual(decision.observation, actionSteps[0].observation);
      assert.deepEqual(decision.nextObservation, actionSteps.at(-1).nextObservation);
      assert.equal(decision.collision, actionSteps.some(s => s.collision));
      assert.equal(decision.elapsedSteps, actionSteps.length);
      assert.ok(actionSteps.every(s => JSON.stringify(s.control) === JSON.stringify(decision.control)));
      if (!decision.collision) assert.equal(decision.elapsedSteps, 5);
      assert.ok(decision.observation.sensors.every(s => s >= 0 && s <= 180));
    }
  }
});

test('time-limit episodes are truncated, not labeled as collisions', () => {
  const options = { seed: 3, episodes: 2, maxActions: 1, persistence: 0, world: box(2000) };
  const events = [...collect(options)];
  const decisions = events.filter(e => e.type === 'decision').map(e => e.record);
  assert.equal(decisions.length, 2);
  assert.ok(decisions.every(d => d.truncated && !d.terminated && !d.collision));
  assert.deepEqual(episodeSplits(2, 42), ['train', 'train']);
});

test('collection rejects overrides of the fixed five-step decision interval', () => {
  assert.throws(() => [...collect({ seed: 42, episodes: 1, maxActions: 1, persistence: .65,
    config: { actionSteps: 25 } })], /5 steps per action/);
});

test('lookahead warns about a wall beyond the next action without moving the real robot', () => {
  const sim = new Simulation({}, box());
  sim.reset(new Random(0), { ...state(160, 100), velocity: { x: 80, y: 0 } });
  sim.step(sim.controlFor(0)); // Fork a noninitial state, including its tick and momentum.
  const observation = sim.observe();
  const control = sim.controlFor(0);
  const result = lookahead(sim, control);
  assert.equal(result.lookaheadCollision, true);
  assert.ok(result.lookaheadElapsedSteps > 5 && result.lookaheadElapsedSteps <= 50);
  assert.equal(result.lookaheadSteps, 50);
  assert.deepEqual(sim.observe(), observation);
  assert.deepEqual(lookahead(sim, control), result);
  for (let i = 0; i < 5; i++) assert.equal(sim.step(control).collision, false);
  assert.equal(sim.observe().step, observation.step + 5);
});

test('lookahead holds world force through heading changes and preserves isolated state', () => {
  const sim = new Simulation({}, box(2000));
  sim.reset(new Random(0), { ...state(1000, 1000), velocity: { x: 80, y: 0 } });
  const before = sim.observe();
  const fork = sim.fork();
  fork.world.walls[0].radius = 20;
  assert.equal(sim.world.walls[0].radius, 1);
  const control = sim.controlFor(5);
  const result = lookahead(sim, control);
  assert.deepEqual(result, { lookaheadCollision: false, lookaheadSteps: 50, lookaheadElapsedSteps: 50 });
  assert.deepEqual(sim.observe(), before);
});

test('counterfactual targets leave the seeded real walk unchanged, including episode limits', () => {
  const world = box();
  const options = { seed: 42, episodes: 30, maxActions: 1, persistence: .65, world };
  const events = [...collect(options)];
  const decisions = events.filter(e => e.type === 'decision').map(e => e.record);
  assert.ok(decisions.some(d => d.lookaheadCollision && !d.collision && d.truncated));
  for (const decision of decisions) {
    const random = new Random((options.seed + Math.imul(decision.episodeId, 0x9e3779b9)) >>> 0);
    const sim = new Simulation({}, world);
    sim.reset(random);
    const walk = new RandomWalk(random, options.persistence);
    const control = sim.controlFor(walk.next());
    assert.deepEqual(decision.observation, sim.observe());
    assert.deepEqual(decision.control, control);
    let collided = false;
    for (let step = 0; step < 50; step++) {
      collided = sim.step(control).collision;
      if (step < 5 && (step === 4 || collided)) assert.deepEqual(decision.nextObservation, sim.observe());
      if (collided) {
        assert.equal(decision.lookaheadElapsedSteps, step + 1);
        break;
      }
    }
    assert.equal(decision.lookaheadCollision, collided);
    if (!collided) assert.equal(decision.lookaheadElapsedSteps, 50);
    assert.equal(decision.terminated, decision.collision);
  }
});
