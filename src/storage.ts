import type { Artifact } from './artifacts.js';

const DB = 'collision-lab-library';
let pending: Promise<IDBDatabase> | undefined;
function database(): Promise<IDBDatabase> {
  return pending ??= new Promise((resolve, reject) => {
    const request = indexedDB.open(DB, 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore('artifacts', { keyPath: 'id' });
      request.result.createObjectStore('files');
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
function completed(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error ?? new Error('Local storage transaction aborted'));
  });
}
export async function listArtifacts(): Promise<Artifact[]> {
  const db = await database();
  return new Promise((resolve, reject) => {
    const request = db.transaction('artifacts').objectStore('artifacts').getAll();
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
export async function saveArtifact(artifact: Artifact, files: Record<string, Blob>): Promise<void> {
  const db = await database();
  const transaction = db.transaction(['artifacts', 'files'], 'readwrite');
  const done = completed(transaction);
  try {
    transaction.objectStore('artifacts').put(artifact);
    for (const [name, blob] of Object.entries(files)) transaction.objectStore('files').put(blob, `${artifact.id}/${name}`);
  } catch (error) {
    transaction.abort();
    await done.catch(() => {});
    throw error;
  }
  await done; // The server may delete its temporary copy only after this commits.
}
export async function fileFor(id: string, name: string): Promise<Blob> {
  const db = await database();
  return new Promise((resolve, reject) => {
    const request = db.transaction('files').objectStore('files').get(`${id}/${name}`);
    request.onsuccess = () => request.result instanceof Blob ? resolve(request.result) : reject(new Error('File is no longer saved in this browser'));
    request.onerror = () => reject(request.error);
  });
}
export async function deleteArtifact(id: string): Promise<void> {
  const db = await database();
  const transaction = db.transaction(['artifacts', 'files'], 'readwrite');
  const done = completed(transaction);
  transaction.objectStore('artifacts').delete(id);
  const files = transaction.objectStore('files');
  const cursor = files.openCursor(IDBKeyRange.bound(`${id}/`, `${id}/\uffff`));
  cursor.onsuccess = () => { if (cursor.result) { cursor.result.delete(); cursor.result.continue(); } };
  await done;
}
export async function gzipJSON(value: unknown): Promise<Blob> {
  return new Response(new Blob([JSON.stringify(value)]).stream().pipeThrough(new CompressionStream('gzip'))).blob();
}
export async function readJSON(blob: Blob): Promise<unknown> {
  return new Response(blob.stream().pipeThrough(new DecompressionStream('gzip'))).json();
}
export function workspaceToken(): string {
  const key = 'collision-lab-workspace';
  let token = localStorage.getItem(key);
  if (!token || !/^[a-f0-9]{64}$/.test(token)) {
    token = Array.from(crypto.getRandomValues(new Uint8Array(32)), n => n.toString(16).padStart(2, '0')).join('');
    localStorage.setItem(key, token);
  }
  return token;
}
