import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { restoreExample } from '../scripts/restore-example.mjs';

test('bundled example restores a fresh checkout exactly and protects existing files', { timeout: 30000 }, async () => {
  const root = await mkdtemp(join(tmpdir(), 'collision-example-'));
  try {
    assert.equal(await restoreExample(root), 8);
    assert.equal(await restoreExample(root), 0);
    const pin = JSON.parse(await readFile(new URL('../examples/shared-example.json', import.meta.url), 'utf8'));
    const manifest = JSON.parse(await readFile(join(root, 'data', pin.dataset, 'manifest.json'), 'utf8'));
    const reportPath = join(root, 'models', pin.model, 'report.json');
    const report = JSON.parse(await readFile(reportPath, 'utf8'));
    assert.equal(manifest.counts.episodes, 10000);
    assert.equal(report.datasetFingerprint, pin.datasetFingerprint);
    await writeFile(reportPath, 'Keep this existing file');
    await assert.rejects(restoreExample(root), /Existing example file differs/);
    assert.equal(await readFile(reportPath, 'utf8'), 'Keep this existing file');
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
