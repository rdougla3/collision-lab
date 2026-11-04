/** Small seeded PRNG shared by the browser and Node; no use of Math.random. */
export class Random {
  private state: number;

  constructor(seed: number) {
    if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffffffff) {
      throw new Error("Seed must be an integer between 0 and 4294967295.");
    }
    this.state = seed >>> 0;
  }

  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let value = this.state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  }

  integer(min: number, max: number): number {
    return min + Math.floor(this.next() * (max - min + 1));
  }
}

export const ACTIONS = Object.freeze(Array.from({ length: 11 }, (_, i) => i - 5));

/** Persistent random steering, with uniform exploration of all 11 actions. */
export class RandomWalk {
  private previous: number | undefined;

  constructor(private readonly random: Random, readonly persistence = 0.65) {
    if (!Number.isFinite(persistence) || persistence < 0 || persistence > 1) {
      throw new Error("Persistence must be between 0 and 1.");
    }
  }

  next(): number {
    if (this.previous === undefined || this.random.next() >= this.persistence) {
      this.previous = this.random.integer(-5, 5);
    }
    return this.previous;
  }

  reset(): void {
    this.previous = undefined;
  }
}
