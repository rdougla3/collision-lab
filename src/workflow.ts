import { BrowserNetwork } from './network.js';
import type { Artifact, Job, ServerState } from './artifacts.js';
import { listArtifacts, saveArtifact, fileFor, deleteArtifact, readJSON, gzipJSON, workspaceToken } from './storage.js';
import { getLayout, setLayout } from './layout.js';
import { freshSeed } from './seeds.js';

const el = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const value = (id: string): string => el<HTMLInputElement>(id).value;
const number = (id: string): number => Number(value(id));
let artifacts: Artifact[] = [];
let currentJob: string | null = null;
let backendReady = false;
let serverBusy = false;
let loaded: BrowserNetwork | undefined;
let starting = false;
const remember = (): void => localStorage.setItem('collision-lab-selection', JSON.stringify({ dataset: value('dataset'), model: value('saved-model'), balance: value('balance-plan') }));
function error(reason: unknown): void { el('workflow-error').textContent = reason instanceof Error ? reason.message : String(reason); }
async function raw(path: string, init: RequestInit = {}): Promise<Response> {
  const response = await fetch(path, { ...init, headers: { ...init.headers, 'X-Workspace-Token': workspaceToken() } });
  if (!response.ok) { const result = await response.json(); throw new Error(result.error ?? `Request failed (${response.status})`); }
  return response;
}
async function api<T>(path: string, data?: unknown): Promise<T> {
  return (await raw(path, data === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) })).json();
}
function busy(): void {
  document.querySelectorAll<HTMLButtonElement>('.job-start').forEach(button => { button.disabled = starting || currentJob !== null || serverBusy || !backendReady; });
  el<HTMLButtonElement>('cancel-job').disabled = currentJob === null;
}
function options(select: HTMLSelectElement, items: Artifact[], empty: string, preferred?: string): void {
  const previous = preferred ?? select.value;
  select.replaceChildren(new Option(empty, ''), ...items.map(item => new Option(`${item.shared ? 'Shared example' : 'This browser'} · ${item.name}`, item.id)));
  select.value = items.some(item => item.id === previous) ? previous : items[0]?.id ?? '';
}
function datasetChanged(preferred?: string): void {
  const dataset = artifacts.find(item => item.id === value('dataset'));
  const counts = dataset?.details.counts;
  el('dataset-summary').textContent = counts
    ? `${counts.episodes} episodes · ${counts.decisions} decisions · ${(100 * counts.lookaheadCollisions / counts.decisions).toFixed(1)}% one-second collision targets · ${counts.collisions} actual collisions`
    : 'Collect a run with at least 10 episodes.';
  const select = el<HTMLSelectElement>('balance-plan');
  const plans = artifacts.filter(item => item.kind === 'balance' && item.datasetId === dataset?.id);
  const old = preferred ?? select.value;
  select.replaceChildren(new Option('All training examples', ''), ...plans.map(plan => new Option(`${plan.shared ? 'Example' : 'This browser'} · ${plan.details.selectedTrainTargets} targets · ${((plan.details.achievedCollisionFraction ?? 0) * 100).toFixed(1)}% collisions`, plan.id)));
  select.value = plans.some(plan => plan.id === old) ? old : plans.find(p => p.shared)?.id ?? '';
  el<HTMLButtonElement>('delete-dataset').disabled = !dataset || !!dataset.shared;
  el<HTMLButtonElement>('use-layout').disabled = !dataset?.details.world;
  trainingSource();
}
async function refresh(preferred: { dataset?: string; model?: string; balance?: string } = {}, resume = true): Promise<void> {
  let server: ServerState | undefined;
  try {
    server = await api<ServerState>('/api/state'); backendReady = true; serverBusy = server.busy;
    localStorage.setItem('collision-lab-examples', JSON.stringify(server.examples));
  } catch { backendReady = false; serverBusy = false; }
  const local = await listArtifacts();
  let examples: Artifact[] = [];
  try { examples = server?.examples ?? JSON.parse(localStorage.getItem('collision-lab-examples') ?? '[]'); } catch { /* Fresh browser. */ }
  artifacts = [...new Map([...local, ...examples].map(item => [item.id,item])).values()];
  let previous: typeof preferred = {};
  try { previous = JSON.parse(localStorage.getItem('collision-lab-selection') ?? '{}'); } catch { /* Fresh browser. */ }
  options(el('dataset'), artifacts.filter(item => item.kind === 'dataset'), 'No datasets', preferred.dataset ?? (value('dataset') || previous.dataset));
  options(el('saved-model'), artifacts.filter(item => item.kind === 'model'), 'No saved models', preferred.model ?? (value('saved-model') || previous.model));
  datasetChanged(preferred.balance ?? (value('balance-plan') || previous.balance));
  el<HTMLButtonElement>('delete-model').disabled = !value('saved-model') || !!artifacts.find(a => a.id === value('saved-model'))?.shared;
  remember(); trainingSource(); busy();
  if (resume && !currentJob && !starting) {
    const pending = server?.jobs.find(job => ['running','complete','failed','cancelled'].includes(job.status));
    if (pending) { monitor(pending); return; }
  }
  if (!backendReady && !currentJob) el('job-status').textContent = 'Server unavailable · browser models still work';
  else if (serverBusy && !currentJob) el('job-status').textContent = 'Server is busy · your library is saved here';
  if (serverBusy && !currentJob) setTimeout(() => void refresh().catch(error), 2000);
}
async function artifactFile(artifact: Artifact, file: string): Promise<Blob> {
  try { return await fileFor(artifact.id, file); }
  catch {
    if (!artifact.shared) throw new Error('This file is no longer saved in this browser.');
    return (await raw(`/api/example/${artifact.id}/${file}`)).blob();
  }
}
async function loadSaved(id: string): Promise<void> {
  const artifact = artifacts.find(item => item.id === id);
  if (!artifact) throw new Error('Select a model');
  const file = await artifactFile(artifact, 'model.json');
  load(await readJSON(file));
  await saveArtifact(artifact, { 'model.json': file });
  remember();
}
function readableLog(log: string): string {
  return log.split("\n").map(line => {
    try {
      const item = JSON.parse(line);
      if (item.event === "epoch") {
        const v = item.validation;
        const percent = (x: number | null): string => x === null ? "N/A" : `${(x * 100).toFixed(1)}%`;
        return `Epoch ${item.epoch} | validation BCE ${v.loss.toFixed(5)} | false positives ${percent(v.falsePositiveRate)} | missed collisions ${percent(v.missedCollisionRate)}${item.bestEpoch === undefined ? "" : ` | best epoch ${item.bestEpoch}`}`;
      }
      if (item.event === "complete") return `Saved best model from epoch ${item.bestEpoch}. Test BCE: ${item.test.loss.toFixed(5)}.`;
      if (item.selectedTrainTargets) return `Balance plan: ${item.selectedTrainTargets}/${item.originalTrainTargets} training targets retained; ${(item.achievedCollisionFraction * 100).toFixed(1)}% collisions. History preserved.`;
    } catch { /* Ordinary process output is displayed as text. */ }
    return line;
  }).join("\n");
}

function monitor(job: Job): void {
  currentJob = job.id; starting = false; busy(); el('workflow-error').textContent = '';
  const poll = async (): Promise<void> => {
    try {
      const next = await api<Job>(`/api/jobs/${job.id}`);
      el('job-status').textContent = `${next.kind}: ${next.status}`;
      el('job-log').textContent = readableLog(next.log) || 'Starting…';
      if (next.status === 'running' || next.status === 'preparing') { setTimeout(() => void poll(), 500); return; }
      const preferred: { dataset?: string; model?: string; balance?: string } = {};
      if (next.status === 'complete' && next.result) {
        const { artifact, files } = next.result;
        // A quota/storage failure leaves the server result available for retry for 24 hours.
        const blobs: Record<string, Blob> = {};
        for (const [index, file] of files.entries()) {
          el('job-status').textContent = `Saving in this browser · file ${index + 1}/${files.length}`;
          blobs[file] = await (await raw(`/api/jobs/${job.id}/files/${file}`)).blob();
        }
        await saveArtifact(artifact, blobs);
        if (artifact.kind === 'dataset') preferred.dataset = artifact.id;
        if (artifact.kind === 'balance') { preferred.dataset = artifact.datasetId; preferred.balance = artifact.id; }
        if (artifact.kind === 'model') preferred.model = artifact.id;
        await api(`/api/jobs/${job.id}/ack`, {});
        currentJob = null;
        await refresh(preferred, false);
        if (artifact.kind === 'model') await loadSaved(artifact.id);
        if (artifact.kind === 'dataset') stage('dataset');
        el('job-status').textContent = `${next.kind} complete · saved in this browser`;
      } else {
        await api(`/api/jobs/${job.id}/ack`, {}); currentJob = null;
        await refresh({}, false);
        if (next.status === 'failed') error('The job failed. Its output is shown above.');
      }
      busy();
    } catch (reason) {
      currentJob = null; starting = false; busy();
      error(`${reason instanceof Error ? reason.message : String(reason)}. Refresh to retry retrieving this job. Unsaved server results expire after 24 hours.`);
    }
  };
  void poll();
}
async function start(path: string, payload: Record<string, unknown>): Promise<void> {
  if (starting || currentJob) return;
  starting = true; busy(); el('workflow-error').textContent = '';
  let prepared: Job | undefined;
  try {
    if (path !== '/api/collect') {
      const dataset = artifacts.find(a => a.id === payload.datasetId);
      if (!dataset) throw new Error('Select a dataset');
      const balance = artifacts.find(a => a.id === payload.balanceId);
      prepared = await api<Job>('/api/prepare', {}); currentJob = prepared.id; busy();
      payload.jobId = prepared.id;
      const upload = async (artifact: Artifact, file: string): Promise<void> => {
        el('job-status').textContent = `Sending ${file} to the server for this job…`;
        await raw(`/api/jobs/${prepared!.id}/input/${file}`, { method: 'PUT', headers: { 'Content-Type': 'application/gzip' }, body: await artifactFile(artifact, file) });
      };
      if (!dataset.shared) { await upload(dataset,'manifest.json'); await upload(dataset,'decisions.jsonl'); }
      if (balance && !balance.shared) await upload(balance,'balance.json');
    }
    monitor(await api<Job>(path,payload));
  } catch (reason) {
    if (prepared) await api(`/api/jobs/${prepared.id}/ack`,{}).catch(() => {});
    currentJob = null; starting = false; busy(); error(reason);
  }
}
function mode(mode: 'train' | 'test'): void {
  el('train-tools').hidden = mode !== 'train'; el('test-tools').hidden = mode !== 'test';
  for (const name of ['train','test']) { el(`mode-${name}`).classList.toggle('primary', name === mode); el(`mode-${name}`).setAttribute('aria-pressed',String(name === mode)); }
  el('mode-description').textContent = mode === 'train' ? 'Collect a walk, balance the data, then train a network.' : 'Run a trained model. Drag the goal to try a new destination.';
  el('arena-mode').textContent = mode === 'train' ? 'RANDOM WALK' : 'GOAL SEEKING';
  document.querySelectorAll<HTMLElement>('.test-stat').forEach(item => { item.hidden = mode !== 'test'; });
  document.querySelectorAll<HTMLElement>('.train-stat').forEach(item => { item.hidden = mode !== 'train'; });
  document.querySelector('.workspace')!.scrollTop = 0;
  document.dispatchEvent(new CustomEvent('lab:mode', { detail: mode }));
}
function load(artifact: unknown): void {
  const candidate = new BrowserNetwork(artifact);
  loaded = candidate;
  const a = candidate.artifact;
  el("model-summary").textContent = `${a.network.kind.toUpperCase()} · history ${a.sequenceLength} · decisions every ${(a.simulation.dt * a.simulation.actionSteps).toFixed(2)} s · prediction horizon ${(a.simulation.dt * a.prediction.steps).toFixed(1)} s · collision cutoff ${a.threshold}`;
  const training = a.training as { bestEpoch?: number; validationLoss?: number; test?: { loss: number; falsePositiveRate: number | null; missedCollisionRate: number | null }; randomLayouts?: boolean } | undefined;
  const rate = (n: number | null | undefined): string => n == null ? "N/A" : `${(n * 100).toFixed(1)}%`;
  el("model-metrics").textContent = training
    ? `One-second held-force predictions. Best epoch: ${training.bestEpoch}. Validation BCE: ${training.validationLoss?.toFixed(5)}. Test BCE: ${training.test?.loss?.toFixed(5)}. Test false positives: ${rate(training.test?.falsePositiveRate)}. Test missed collisions: ${rate(training.test?.missedCollisionRate)}. ${training.randomLayouts ? "Evaluation used disjoint randomized layouts." : "Evaluation used the same fixed layout."}`
    : "This model has no saved evaluation metrics.";
  el<HTMLButtonElement>("export-model").disabled = false;
  document.dispatchEvent(new CustomEvent("lab:model", { detail: candidate }));
}

function trainingSource(): void {
  const dataset = artifacts.find(item => item.id === value('dataset'));
  const plan = artifacts.find(item => item.id === value('balance-plan'));
  el('training-source').textContent = dataset ? `${dataset.name} · ${plan ? `${plan.details.selectedTrainTargets} balanced targets` : 'all training targets'}` : 'Choose a dataset in step 2.';
}
function stage(selected: 'collect' | 'dataset' | 'network'): void {
  for (const name of ['collect', 'dataset', 'network']) {
    el(`${name}-panel`).hidden = name !== selected;
    el(`stage-${name}`).classList.toggle('active', name === selected);
    el(`stage-${name}`).setAttribute('aria-pressed', String(name === selected));
  }
  document.querySelector('.workspace')!.scrollTop = 0;
}
for (const name of ['collect', 'dataset', 'network'] as const) el(`stage-${name}`).addEventListener('click', () => stage(name));
el('continue-network').addEventListener('click', () => stage('network'));

el('mode-train').addEventListener('click',()=>mode('train'));
el('mode-test').addEventListener('click',()=>mode('test'));
el('dataset').addEventListener('change',()=>{ datasetChanged(); remember(); });
el('balance-plan').addEventListener('change',()=>{ remember(); trainingSource(); });
el('saved-model').addEventListener('change',()=>{ remember(); el<HTMLButtonElement>('delete-model').disabled = !!artifacts.find(a=>a.id===value('saved-model'))?.shared; });
el('refresh-artifacts').addEventListener('click',()=>void refresh().catch(error));
el('cancel-job').addEventListener('click',()=>{ if(currentJob) void api(`/api/jobs/${currentJob}/cancel`,{}).catch(error); });
el('collect-data').addEventListener('click',()=>void start('/api/collect', {
  episodes:number('collect-episodes'), maxActions:number('collect-actions'), seed:freshSeed(), world:getLayout(), decisionsOnly:true,
}));
el('balance-data').addEventListener('click',()=>void start('/api/balance',{datasetId:value('dataset'),fraction:number('balance-fraction'),seed:freshSeed()}));
el('train-model').addEventListener('click',()=>{
  const hidden=value('hidden-widths').split(',').map(Number);
  void start('/api/train',{
    datasetId:value('dataset'),balanceId:value('balance-plan') || undefined,
    network:{kind:value('network-kind'),hidden_sizes:hidden,hidden_size:hidden[0],num_layers:number('recurrent-layers')},
    config:{epochs:number('epochs'),batch_size:number('batch-size'),learning_rate:number('learning-rate'),sequence_length:number('history-length'),patience:number('patience'),weight_decay:number('weight-decay'),optimizer:value('optimizer'),threshold:number('threshold'),seed:freshSeed()},
  });
});
el('load-model').addEventListener('click',()=>void loadSaved(value('saved-model')).catch(error));
el('model-file').addEventListener('change',async()=>{
  const file=el<HTMLInputElement>('model-file').files?.[0]; if(!file)return;
  try {
    if(file.size>128*1024*1024)throw new Error('Model file exceeds 128 MB');
    const model=new BrowserNetwork(JSON.parse(await file.text()));
    const artifact: Artifact={id:`model-${crypto.randomUUID()}`,kind:'model',name:file.name,createdAt:new Date().toISOString(),details:{network:model.artifact.network}};
    await saveArtifact(artifact,{'model.json':await gzipJSON(model.artifact)});
    await refresh({model:artifact.id},false); load(model.artifact); el('workflow-error').textContent='';
  } catch(reason){error(reason);}
});
el('export-model').addEventListener('click',()=>{
  if(!loaded)return;
  const url=URL.createObjectURL(new Blob([JSON.stringify(loaded.artifact)],{type:'application/json'}));
  const link=document.createElement('a');link.href=url;link.download=`collision-${loaded.artifact.network.kind}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
el('use-layout').addEventListener('click',()=>{
  try { const world=artifacts.find(a=>a.id===value('dataset'))?.details.world; if(world)setLayout(world); }catch(reason){error(reason);}
});
el('delete-dataset').addEventListener('click',async()=>{
  try { const id=value('dataset');if(artifacts.find(a=>a.id===id)?.shared)return;
    for(const item of artifacts.filter(a=>a.id===id || (a.kind==='balance' && a.datasetId===id && !a.shared)))await deleteArtifact(item.id);
    await refresh({},false);
  }catch(reason){error(reason);}
});
el('delete-model').addEventListener('click',async()=>{
  try { const id=value('saved-model');if(artifacts.find(a=>a.id===id)?.shared)return;await deleteArtifact(id);await refresh({},false); }catch(reason){error(reason);}
});
function dimensions():void {
  const recurrent=value('network-kind')!=='feedforward';
  el('recurrent-settings').hidden=!recurrent;
  el('dimension-help').textContent=recurrent?'First width applies to each recurrent layer.':'Comma-separated layer widths.';
  el<HTMLInputElement>('history-length').disabled=!recurrent;el<HTMLInputElement>('recurrent-layers').disabled=!recurrent;
}
el('network-kind').addEventListener('change',dimensions); dimensions();busy();
void refresh().then(async()=>{if(value('saved-model')&&!loaded)await loadSaved(value('saved-model'));}).catch(error);
