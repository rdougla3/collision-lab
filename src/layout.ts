import { createWorld, Simulation } from './simulation.js';
import type { World } from './types.js';

export const BOUNDARIES = ['south', 'east', 'north', 'west'];
/** Validate editable geometry at both the browser and API boundary. */
export function validateLayout(value: unknown): World {
  const world = value as World;
  const base = createWorld();
  if (!world || world.width !== base.width || world.height !== base.height || !Array.isArray(world.walls)
      || world.walls.length < 4 || world.walls.length > 44 || typeof world.id !== 'string' || world.id.length > 120) {
    throw new Error('Use the fixed arena bounds and at most 40 obstacles.');
  }
  const ids = new Set<string>();
  for (const wall of world.walls) {
    if (!wall || typeof wall.id !== 'string' || ids.has(wall.id) || wall.id.length > 120) throw new Error('Obstacle IDs must be unique.');
    ids.add(wall.id);
    if (BOUNDARIES.includes(wall.id)) {
      const original = base.walls.find(w => w.id === wall.id)!;
      if (wall.radius !== original.radius || wall.start?.x !== original.start.x || wall.start?.y !== original.start.y
        || wall.end?.x !== original.end.x || wall.end?.y !== original.end.y) throw new Error('Arena boundary walls cannot be edited.');
    } else if (!Number.isFinite(wall.radius) || wall.radius < 5 || wall.radius > 80
      || ![wall.start, wall.end].every(p => p && Number.isFinite(p.x) && Number.isFinite(p.y)
        && p.x >= 0 && p.x <= world.width
        && p.y >= 0 && p.y <= world.height)) {
      throw new Error('Keep obstacles inside the arena, with thickness between 10 and 160.');
    }
  }
  if (!BOUNDARIES.every(id => ids.has(id))) throw new Error('Arena boundary walls cannot be removed.');
  new Simulation({}, world); // Also verify that a robot can spawn.
  return structuredClone(world);
}
const KEY = 'collision-lab-layout';
let current: World | undefined;
export function getLayout(): World {
  if (!current) {
    try { current = validateLayout(JSON.parse(localStorage.getItem(KEY) ?? 'null')); }
    catch { current = createWorld(); }
  }
  return structuredClone(current);
}
export function setLayout(value: World): void {
  const next = validateLayout(value);
  localStorage.setItem(KEY, JSON.stringify(next));
  current = next;
  document.dispatchEvent(new CustomEvent('lab:layout', { detail: structuredClone(next) }));
}
