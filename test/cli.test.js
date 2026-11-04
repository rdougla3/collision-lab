import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const cli = fileURLToPath(new URL('../dist/cli.js', import.meta.url));
const run = args => spawnSync(process.execPath, [cli, ...args], { encoding: 'utf8' });

test('CLI writes reproducible sequential files and a completed manifest, refusing overwrite', () => {
  const temporary = mkdtempSync(join(tmpdir(), 'collision-cli-'));
  try {
    for (const name of ['first', 'repeat']) {
      const result = run(['--out', join(temporary, name), '--episodes', '10', '--max-actions', '2']);
      assert.equal(result.status, 0, result.stderr);
    }
    const first = join(temporary, 'first');
    const manifest = JSON.parse(readFileSync(join(first, 'manifest.json'), 'utf8'));
    assert.equal(manifest.status, 'complete');
    assert.equal(manifest.schemaVersion, 2);
    assert.deepEqual(manifest.prediction, { target: 'lookaheadCollision', steps: 50, continuation: 'hold-world-force' });
    assert.deepEqual(manifest.splitCounts, { train: 8, validation: 1, test: 1 });
    for (const file of ['steps.jsonl', 'decisions.jsonl', 'episodes.jsonl', 'decisions.csv']) {
      assert.equal(readFileSync(join(first, file), 'utf8'), readFileSync(join(temporary, 'repeat', file), 'utf8'));
    }
    const decisions = readFileSync(join(first, 'decisions.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
    const steps = readFileSync(join(first, 'steps.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
    assert.equal(manifest.counts.decisions, decisions.length);
    assert.equal(manifest.counts.lookaheadCollisions, decisions.filter(d => d.lookaheadCollision).length);
    assert.equal(manifest.counts.steps, steps.length);
    const csv = readFileSync(join(first, 'decisions.csv'), 'utf8').trim().split('\n');
    assert.equal(csv.length, decisions.length + 1);
    const columns = csv[0].split(',');
    assert.ok(csv.every(line => line.split(',').length === columns.length));
    for (const [index, decision] of decisions.entries()) {
      const row = csv[index + 1].split(',');
      assert.equal(Number(row[columns.indexOf('collision')]), Number(decision.collision));
      assert.equal(Number(row[columns.indexOf('lookahead_collision')]), Number(decision.lookaheadCollision));
      assert.equal(Number(row[columns.indexOf('lookahead_steps')]), 50);
      assert.equal(Number(row[columns.indexOf('lookahead_elapsed_steps')]), decision.lookaheadElapsedSteps);
      assert.equal(Number(row[columns.indexOf('sensor_0')]), decision.observation.sensors[0]);
    }
    const before = readFileSync(join(first, 'manifest.json'), 'utf8');
    assert.notEqual(run(['--out', first]).status, 0);
    assert.equal(readFileSync(join(first, 'manifest.json'), 'utf8'), before);
    const compact = join(temporary, 'compact');
    assert.equal(run(['--out', compact, '--episodes', '1', '--max-actions', '1', '--decisions-only']).status, 0);
    assert.equal(existsSync(join(compact, 'steps.jsonl')), false);
    assert.equal(JSON.parse(readFileSync(join(compact, 'manifest.json'), 'utf8')).files.steps, null);
    const defaults = join(temporary, 'defaults');
    assert.equal(run(['--out', defaults, '--episodes', '1', '--max-actions', '1', '--decisions-only']).status, 0);
    const config = JSON.parse(readFileSync(join(defaults, 'manifest.json'), 'utf8')).simulation;
    assert.equal(config.actionSteps * config.dt, .1);
    assert.equal(config.acceleration, 140);
    assert.equal(config.maxTurnRate, 3);
    const invalid = join(temporary, 'invalid');
    assert.notEqual(run(['--out', invalid, '--episodes', '-1']).status, 0);
    assert.equal(existsSync(invalid), false);
    assert.notEqual(run(['--out', join(temporary, 'override'), '--action-steps', '25']).status, 0);
    assert.equal(run(['--help']).status, 0);
  } finally {
    rmSync(temporary, { recursive: true, force: true });
  }
});
