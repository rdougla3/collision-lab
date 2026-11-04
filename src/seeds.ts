/** Choose a new seed per run; saved artifacts retain it for reproducibility. */
export function freshSeed(): number {
  return crypto.getRandomValues(new Uint32Array(1))[0]!;
}
