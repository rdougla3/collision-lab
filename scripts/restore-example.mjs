import { createReadStream, createWriteStream } from 'node:fs';
import { readFile, mkdir, rename, rm } from 'node:fs/promises';
import { dirname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash, randomUUID } from 'node:crypto';
import { createGunzip } from 'node:zlib';
import { Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';

const project = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const bundle = resolve(project, 'examples/bundle');

async function digest(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

/** Restore only the pinned example. Existing, different files are never replaced. */
export async function restoreExample(destination = project) {
  const manifest = JSON.parse(await readFile(resolve(bundle, 'manifest.json'), 'utf8'));
  if (manifest.version !== 1 || !Array.isArray(manifest.files)) throw new Error('Invalid example bundle manifest');
  let restored = 0;
  for (const entry of manifest.files) {
    if (!/^[\w.-]+\.gz$/.test(entry.file) || !/^(data|models)\/[\w.-]+\/[\w.-]+$/.test(entry.target)
      || !/^[a-f0-9]{64}$/.test(entry.sha256) || !Number.isSafeInteger(entry.bytes) || entry.bytes < 0) {
      throw new Error('Invalid example bundle entry');
    }
    const target = resolve(destination, entry.target);
    if (!target.startsWith(resolve(destination) + sep)) throw new Error('Invalid example path');
    try {
      if (await digest(target) !== entry.sha256) throw new Error(`Existing example file differs from the bundle: ${target}. Move it aside before restoring.`);
      continue;
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
    await mkdir(dirname(target), { recursive: true });
    const temporary = `${target}.${randomUUID()}.tmp`;
    const hash = createHash('sha256');
    let bytes = 0;
    const verify = new Transform({ transform(chunk, _encoding, callback) {
      bytes += chunk.length;
      if (bytes > entry.bytes) return callback(new Error(`Example size mismatch: ${entry.target}`));
      hash.update(chunk); callback(null, chunk);
    } });
    try {
      await pipeline(createReadStream(resolve(bundle, entry.file)), createGunzip(), verify, createWriteStream(temporary, { flags: 'wx' }));
      if (bytes !== entry.bytes || hash.digest('hex') !== entry.sha256) throw new Error(`Example checksum mismatch: ${entry.target}`);
      await rename(temporary, target);
      restored++;
    } finally {
      await rm(temporary, { force: true });
    }
  }
  return restored;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  console.log(`Shared example ready (${await restoreExample()} files restored).`);
}
