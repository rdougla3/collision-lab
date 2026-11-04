import type { Vec2, Wall } from "./types.js";

export function distanceToSegment(point: Vec2, start: Vec2, end: Vec2): number {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;
  const t = lengthSquared === 0 ? 0 : Math.max(0, Math.min(1,
    ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared));
  return Math.hypot(point.x - start.x - t * dx, point.y - start.y - t * dy);
}

function rayCircle(origin: Vec2, direction: Vec2, center: Vec2, radius: number): number {
  const x = origin.x - center.x;
  const y = origin.y - center.y;
  const b = x * direction.x + y * direction.y;
  const c = x * x + y * y - radius * radius;
  if (c <= 0) return 0;
  const discriminant = b * b - c;
  if (discriminant < 0) return Infinity;
  const distance = -b - Math.sqrt(discriminant);
  return distance >= 0 ? distance : Infinity;
}

/** Distance along a unit ray to a capsule (a thick line with round ends). */
export function rayWall(origin: Vec2, direction: Vec2, wall: Wall, expansion = 0): number {
  const radius = wall.radius + expansion;
  if (distanceToSegment(origin, wall.start, wall.end) <= radius) return 0;
  const dx = wall.end.x - wall.start.x;
  const dy = wall.end.y - wall.start.y;
  const length = Math.hypot(dx, dy);
  let nearest = Math.min(
    rayCircle(origin, direction, wall.start, radius),
    rayCircle(origin, direction, wall.end, radius),
  );
  if (length === 0) return nearest;

  // Intersect the central rectangle in coordinates aligned with the wall.
  const ux = dx / length;
  const uy = dy / length;
  const px = origin.x - wall.start.x;
  const py = origin.y - wall.start.y;
  const axes = [
    { p: px * ux + py * uy, d: direction.x * ux + direction.y * uy, min: 0, max: length },
    { p: -px * uy + py * ux, d: -direction.x * uy + direction.y * ux, min: -radius, max: radius },
  ];
  let entry = 0;
  let exit = Infinity;
  for (const axis of axes) {
    if (Math.abs(axis.d) < 1e-12) {
      if (axis.p < axis.min || axis.p > axis.max) return nearest;
    } else {
      const first = (axis.min - axis.p) / axis.d;
      const second = (axis.max - axis.p) / axis.d;
      entry = Math.max(entry, Math.min(first, second));
      exit = Math.min(exit, Math.max(first, second));
      if (entry > exit) return nearest;
    }
  }
  if (exit >= 0) nearest = Math.min(nearest, entry);
  return nearest;
}

export function wrapAngle(angle: number): number {
  return Math.atan2(Math.sin(angle), Math.cos(angle));
}
