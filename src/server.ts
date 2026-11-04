import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { existsSync, createReadStream, createWriteStream } from 'node:fs';
import { readFile, readdir, mkdir, writeFile, rm, rename } from 'node:fs/promises';
import { dirname, extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn, type ChildProcess } from 'node:child_process';
import { randomUUID, createHash } from 'node:crypto';
import { createGzip, createGunzip } from 'node:zlib';
import { Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { validateDecisionTiming } from './simulation.js';
import { validatePrediction } from './prediction.js';
import { validateLayout } from './layout.js';
import type { Artifact, Job } from './artifacts.js';

const project = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const runtime = resolve(process.env.COLLISION_RUNTIME_DIR ?? resolve(project, '.runtime'));
const port = Number(process.env.PORT ?? 5173);
const python = process.env.COLLISION_PYTHON
  ?? ['.venv/bin/python', '.venv/Scripts/python.exe', '../.venv/bin/python'].map(path => resolve(project, path)).find(existsSync)
  ?? 'python3';
const TTL = 24 * 60 * 60 * 1000;
const types: Record<string, string> = { '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.map': 'application/json' };
type InternalJob = Job & { owner: string; createdAt: string; process?: ChildProcess; uploaded: string[] };
const jobs = new Map<string, InternalJob>();
let active: InternalJob | undefined;
const jobPath = (job: InternalJob, ...parts: string[]): string => resolve(runtime, job.id, ...parts);
const exposed = (job: InternalJob): Job => ({ id: job.id, kind: job.kind, status: job.status, log: job.log, datasetId: job.datasetId, result: job.result });
const writes = new WeakMap<InternalJob, Promise<void>>();
const persist = (job: InternalJob): Promise<void> => {
  const { process: _, ...record } = job;
  const snapshot = JSON.stringify(record);
  const next = (writes.get(job) ?? Promise.resolve()).catch(() => {}).then(async () => {
    await writeFile(jobPath(job, 'job.json.tmp'), snapshot);
    await rename(jobPath(job, 'job.json.tmp'), jobPath(job, 'job.json'));
  });
  writes.set(job, next); return next;
};
function json(response: ServerResponse, code: number, value: unknown): void {
  response.writeHead(code, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' }); response.end(JSON.stringify(value));
}
function ownerOf(request: IncomingMessage): string {
  const token = request.headers['x-workspace-token'];
  if (typeof token !== 'string' || !/^[a-f0-9]{64}$/.test(token)) throw new Error('A browser workspace token is required');
  const origin = request.headers.origin;
  if (origin && new URL(origin).host !== request.headers.host) throw new Error('Cross-origin requests are not allowed');
  return createHash('sha256').update(token).digest('hex');
}
function owned(id: string, owner: string): InternalJob {
  const job = jobs.get(id);
  if (!job || job.owner !== owner) throw new Error('Job not found in this workspace');
  return job;
}
async function body(request: IncomingMessage): Promise<Record<string, unknown>> {
  if (!request.headers['content-type']?.startsWith('application/json')) throw new Error('Expected application/json');
  let text = '';
  for await (const chunk of request) { text += chunk.toString(); if (text.length > 65536) throw new Error('Request too large'); }
  const value: unknown = JSON.parse(text);
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Expected a JSON object');
  return value as Record<string, unknown>;
}
const positive = (v: unknown, fallback: number, max: number): number => {
  const n = v ?? fallback;
  if (typeof n !== 'number' || !Number.isInteger(n) || n < 1 || n > max) throw new Error(`Expected integer between 1 and ${max}`);
  return n;
};
const seed = (v: unknown): number => {
  const n = v ?? 42;
  if (typeof n !== 'number' || !Number.isInteger(n) || n < 0 || n > 0xffffffff) throw new Error('Invalid seed');
  return n;
};
const readJSON = async (path: string) => JSON.parse(await readFile(path, 'utf8'));
function checkManifest(manifest: Awaited<ReturnType<typeof readJSON>>): void {
  if (manifest.status !== 'complete' || manifest.schemaVersion !== 2) throw new Error('Expected a completed lookahead dataset');
  validateDecisionTiming(manifest.simulation); validatePrediction(manifest.prediction);
}
const exampleConfig = await readJSON(process.env.COLLISION_EXAMPLE_CONFIG ?? resolve(project, 'examples/shared-example.json'));
const exampleData = resolve(project, 'data', exampleConfig.dataset);
const exampleModel = resolve(project, 'models', exampleConfig.model);
const exampleManifest = await readJSON(resolve(exampleData, 'manifest.json'));
const exampleReport = await readJSON(resolve(exampleModel, 'report.json'));
const { selected: _, ...exampleBalance } = await readJSON(resolve(exampleData, exampleConfig.balance));
checkManifest(exampleManifest);
if (exampleReport.status !== 'complete' || exampleReport.datasetFingerprint !== exampleConfig.datasetFingerprint
  || exampleBalance.datasetFingerprint !== exampleConfig.datasetFingerprint) throw new Error('Shared example artifacts do not match');
const examples: Artifact[] = [
  { id: 'example-data', kind: 'dataset', name: exampleConfig.name, createdAt: exampleManifest.createdAt, shared: true, details: exampleManifest },
  { id: 'example-balance', kind: 'balance', name: 'Example · 50/50 balance', datasetId: 'example-data', createdAt: exampleManifest.createdAt, shared: true, details: exampleBalance },
  { id: 'example-model', kind: 'model', name: 'Example · trained feedforward', datasetId: 'example-data', createdAt: exampleManifest.createdAt, shared: true, details: exampleReport },
];
async function newJob(owner: string): Promise<InternalJob> {
  const job: InternalJob = { id: `job-${randomUUID()}`, owner, createdAt: new Date().toISOString(), uploaded: [], kind: 'upload', status: 'preparing', log: '' };
  await mkdir(jobPath(job, 'input'), { recursive: true }); jobs.set(job.id, job); await persist(job); return job;
}
async function finish(job: InternalJob): Promise<void> {
  const output = jobPath(job, 'output');
  const kind = job.kind === 'collect' ? 'dataset' : job.kind === 'balance' ? 'balance' : 'model';
  const metadataFile = kind === 'dataset' ? 'manifest.json' : kind === 'model' ? 'report.json' : 'balance.json';
  const metadata = await readJSON(resolve(output, metadataFile));
  if (kind === 'dataset') checkManifest(metadata);
  if (kind === 'model' && metadata.status !== 'complete') throw new Error('Incomplete model output');
  const { selected: _, ...details } = metadata;
  const allowed = kind === 'dataset' ? ['manifest.json', 'decisions.jsonl', 'decisions.csv', 'episodes.jsonl', 'steps.jsonl']
    : kind === 'model' ? ['model.json', 'report.json', 'weights.pt', 'balance.json'] : ['balance.json'];
  const files = (await readdir(output)).filter(file => allowed.includes(file));
  const artifact: Artifact = { id: `${kind}-${job.id}`, kind, name: `${kind === 'model' ? metadata.network.kind.toUpperCase() : kind === 'dataset' ? 'Collection' : 'Balance'} · ${job.createdAt}`, createdAt: job.createdAt, datasetId: job.datasetId, details };
  job.result = { artifact, files };
}
async function start(job: InternalJob, command: string, args: string[]): Promise<void> {
  if (active?.process) throw new Error('The server is busy. Try again when the current job finishes.');
  // Reserve the slot before any await to keep simultaneous requests isolated.
  job.status = 'running'; active = job;
  const child = spawn(command, args, { cwd: project, env: { ...process.env, PYTHONUNBUFFERED: '1' }, stdio: ['ignore', 'pipe', 'pipe'] });
  job.process = child;
  const append = (chunk: Buffer) => { job.log = (job.log + chunk.toString()).slice(-30000); };
  child.stdout!.on('data', append); child.stderr!.on('data', append);
  child.on('error', error => { job.log += `\n${error.message}`; });
  child.on('close', async code => {
    try {
      if (job.status !== 'cancelled') {
        if (code !== 0) job.status = 'failed';
        else { await finish(job); job.status = 'complete'; }
      }
    } catch (error) { job.status = 'failed'; job.log += `\n${String(error)}`; }
    finally { delete job.process; await persist(job); }
  });
  await persist(job);
}
async function streamFile(response: ServerResponse, path: string): Promise<void> {
  if (!existsSync(path)) throw new Error('File not found');
  response.writeHead(200, { 'Content-Type': 'application/gzip', 'Cache-Control': 'no-store' });
  await pipeline(createReadStream(path), createGzip({ level: 1 }), response);
}
async function api(request: IncomingMessage, response: ServerResponse, path: string): Promise<void> {
  const owner = ownerOf(request);
  if (request.method === 'GET' && path === '/api/state') {
    json(response, 200, { examples, jobs: [...jobs.values()].filter(j => j.owner === owner).map(exposed), busy: !!active?.process }); return;
  }
  const exampleMatch = path.match(/^\/api\/example\/(example-data|example-balance|example-model)\/([^/]+)$/);
  if (request.method === 'GET' && exampleMatch) {
    const [, id, file] = exampleMatch;
    const allowed = id === 'example-data' ? ['manifest.json','decisions.jsonl','episodes.jsonl','decisions.csv'] : id === 'example-model' ? ['model.json','report.json','weights.pt'] : ['balance.json'];
    if (!allowed.includes(file!)) throw new Error('Unknown example file');
    await streamFile(response, id === 'example-balance' ? resolve(exampleData, exampleConfig.balance) : resolve(id === 'example-data' ? exampleData : exampleModel, file!)); return;
  }
  const match = path.match(/^\/api\/jobs\/([^/]+)(?:\/(files|input)\/([^/]+)|\/(ack|cancel))?$/);
  if (match) {
    const job = owned(match[1]!, owner);
    if (request.method === 'GET' && !match[2] && !match[4]) { json(response, 200, exposed(job)); return; }
    if (request.method === 'GET' && match[2] === 'files') {
      if (job.status !== 'complete' || !job.result?.files.includes(match[3]!)) throw new Error('Result file unavailable');
      await streamFile(response, jobPath(job, 'output', match[3]!)); return;
    }
    if (request.method === 'PUT' && match[2] === 'input') {
      const file = match[3]!;
      if (job.status !== 'preparing' || !['manifest.json','decisions.jsonl','balance.json'].includes(file)) throw new Error('Invalid upload');
      let bytes = 0;
      const limit = new Transform({ transform(chunk: Buffer, _, callback) { bytes += chunk.length; callback(bytes > 1024 * 1024 * 1024 ? new Error('Input file exceeds 1 GB') : null, chunk); } });
      const temporary = jobPath(job, 'input', `${file}.tmp-${randomUUID()}`);
      try {
        await pipeline(request, createGunzip(), limit, createWriteStream(temporary, { flags: 'wx' }));
        if (file === 'manifest.json') checkManifest(await readJSON(temporary));
        await rename(temporary, jobPath(job, 'input', file));
        if (!job.uploaded.includes(file)) job.uploaded.push(file);
        await persist(job); json(response, 200, { uploaded: file });
      } catch (error) { await rm(temporary, { force: true }); throw error; }
      return;
    }
    if (request.method === 'POST' && match[4] === 'cancel') {
      if (job.process) { job.status = 'cancelled'; job.process.kill('SIGTERM'); }
      else { job.status = 'cancelled'; await persist(job); }
      json(response, 200, exposed(job)); return;
    }
    if (request.method === 'POST' && match[4] === 'ack') {
      if (job.process) throw new Error('Job is still running');
      await writes.get(job);
      await rm(jobPath(job), { recursive: true, force: true }); jobs.delete(job.id); json(response, 200, { removed: true }); return;
    }
  }
  if (request.method !== 'POST') throw new Error('Unknown API route');
  const input = await body(request);
  if (path === '/api/prepare') { json(response, 201, exposed(await newJob(owner))); return; }
  if (!['/api/collect', '/api/balance', '/api/train'].includes(path)) throw new Error('Unknown API route');
  if (active?.process) throw new Error('The server is busy. Try again when the current job finishes.');
  const job = input.jobId ? owned(String(input.jobId), owner) : await newJob(owner);
  if (job.status !== 'preparing') throw new Error('Job was already started');
  try {
    if (path === '/api/collect') {
      if (input.actionSteps !== undefined || input.randomLayouts !== undefined) throw new Error('Use the fixed timing and an edited layout');
      const world = validateLayout(input.world);
      await writeFile(jobPath(job, 'world.json'), JSON.stringify(world));
      job.kind = 'collect';
      const args = [resolve(project, 'dist/cli.js'), '--out', jobPath(job, 'output'), '--world', jobPath(job, 'world.json'),
        '--episodes', String(positive(input.episodes, 1000, 100000)), '--max-actions', String(positive(input.maxActions, 50, 1000)), '--seed', String(seed(input.seed))];
      if (input.decisionsOnly !== false) args.push('--decisions-only');
      await start(job, process.execPath, args);
    } else {
      if (typeof input.datasetId !== 'string' || input.datasetId.length > 160) throw new Error('Select a dataset');
      job.datasetId = input.datasetId;
      const dataset = input.datasetId === 'example-data' ? exampleData : jobPath(job, 'input');
      if (dataset !== exampleData && !['manifest.json','decisions.jsonl'].every(file => job.uploaded.includes(file))) throw new Error('Upload the browser dataset before starting');
      checkManifest(await readJSON(resolve(dataset, 'manifest.json')));
      if (path === '/api/balance') {
        if (typeof input.fraction !== 'number' || !Number.isFinite(input.fraction) || input.fraction <= 0 || input.fraction > 1) throw new Error('Collision fraction must be in (0, 1]');
        job.kind = 'balance'; await mkdir(jobPath(job, 'output'));
        await start(job, python, ['-m','training.balance','--dataset',dataset,'--out',jobPath(job,'output','balance.json'),'--collision-fraction',String(input.fraction),'--seed',String(seed(input.seed))]);
      } else {
        job.kind = 'train';
        await writeFile(jobPath(job,'config.json'), JSON.stringify(input.config ?? {}));
        await writeFile(jobPath(job,'network.json'), JSON.stringify(input.network ?? {}));
        const args = ['-m','training.train','--dataset',dataset,'--out',jobPath(job,'output'),'--config',jobPath(job,'config.json'),'--network-config',jobPath(job,'network.json')];
        if (input.balanceId === 'example-balance') {
          if (input.datasetId !== 'example-data') throw new Error('Example balance belongs to the example dataset');
          args.push('--balance-plan',resolve(exampleData, exampleConfig.balance));
        } else if (input.balanceId) {
          if (!job.uploaded.includes('balance.json')) throw new Error('Upload the browser balance plan first');
          args.push('--balance-plan',jobPath(job,'input','balance.json'));
        }
        await start(job, python, args);
      }
    }
    json(response, 202, exposed(job));
  } catch (error) {
    if (!job.process) { await rm(jobPath(job), { recursive: true, force: true }); jobs.delete(job.id); }
    throw error;
  }
}
await mkdir(runtime, { recursive: true });
for (const name of await readdir(runtime)) {
  if (!/^job-[a-f0-9-]+$/.test(name)) continue;
  try {
    const job: InternalJob = await readJSON(resolve(runtime,name,'job.json'));
    if (job.id !== name || !/^[a-f0-9]{64}$/.test(job.owner)) continue;
    if (Date.now() - Date.parse(job.createdAt) > TTL) { await rm(resolve(runtime,name), { recursive: true, force: true }); continue; }
    if (job.status === 'running') { job.status = 'failed'; job.log += '\nServer restarted before this job completed.'; }
    jobs.set(job.id, job); await persist(job);
  } catch { /* Incomplete upload metadata; handled by the next cleanup. */ }
}
const cleanup = setInterval(() => {
  for (const job of jobs.values()) if (!job.process && Date.now() - Date.parse(job.createdAt) > TTL) {
    jobs.delete(job.id); void rm(jobPath(job), { recursive: true, force: true });
  }
}, 60000); cleanup.unref();
const server = createServer(async (request, response) => {
  try {
    const path = decodeURIComponent(new URL(request.url ?? '/', 'http://localhost').pathname);
    if (path.startsWith('/api/')) { await api(request,response,path); return; }
    if (request.method !== 'GET' && request.method !== 'HEAD') { response.writeHead(405); response.end(); return; }
    const modules = path.startsWith('/modules/');
    const base = resolve(project, modules ? 'dist' : 'public');
    const file = resolve(base, modules ? path.slice(9) : path === '/' ? 'index.html' : path.slice(1));
    if (!file.startsWith(base + sep) || !types[extname(file)]) { response.writeHead(404); response.end('Not found'); return; }
    const content = await readFile(file); response.writeHead(200, { 'Content-Type': types[extname(file)]!, 'Cache-Control': 'no-store' }); response.end(request.method === 'HEAD' ? undefined : content);
  } catch (error) {
    if (response.headersSent || response.destroyed) { response.destroy(); return; }
    json(response, request.url?.startsWith('/api/') ? 400 : 404, { error: error instanceof Error ? error.message : String(error) });
  }
});
server.on('error', error => { console.error(error.message); process.exitCode = 1; });
server.listen(port,'127.0.0.1',()=>{ const address=server.address(); console.log(`Collision Lab: http://127.0.0.1:${typeof address === 'object' && address ? address.port : port}\nPython: ${python}`); });
for (const signal of ['SIGINT','SIGTERM'] as const) process.on(signal,()=>{ active?.process?.kill('SIGTERM'); clearInterval(cleanup); server.close(); });
