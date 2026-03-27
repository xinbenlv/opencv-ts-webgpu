/**
 * Unit tests for demo/realtime/depth-renderer.ts
 *
 * Only `depthColormap` is tested here — it is a pure function.
 * `DepthRenderer` requires a DOM canvas so it is not tested at unit level.
 *
 * Coverage:
 *  1. depthColormap(0)   → deep blue (near)
 *  2. depthColormap(1)   → yellow/red (far)
 *  3. depthColormap(0.5) → peak green (mid)
 *  4. depthColormap      → all channels are integers in [0, 255]
 *  5. depthColormap      → monotonically increasing R as norm increases
 *  6. depthColormap      → monotonically decreasing B as norm increases
 */

import { describe, it, expect } from 'vitest';
import { depthColormap } from '../../demo/realtime/depth-renderer.ts';

describe('depthColormap', () => {
  it('returns deep blue at norm=0 (near)', () => {
    const [r, g, b] = depthColormap(0);
    expect(r).toBe(0);   // no red at near
    expect(b).toBeGreaterThan(200); // strong blue
    expect(g).toBe(0);   // no green at zero
  });

  it('returns yellow/red at norm=1 (far)', () => {
    const [r, , b] = depthColormap(1);
    expect(r).toBeGreaterThan(200); // strong red component
    expect(b).toBe(0);              // no blue at far
  });

  it('peaks green near norm=0.5', () => {
    const [, g05] = depthColormap(0.5);
    const [, g00] = depthColormap(0.0);
    const [, g10] = depthColormap(1.0);
    expect(g05).toBeGreaterThan(g00);
    expect(g05).toBeGreaterThan(g10);
    expect(g05).toBeGreaterThan(200);
  });

  it('returns integer RGB values in [0, 255]', () => {
    for (const norm of [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]) {
      const [r, g, b] = depthColormap(norm);
      for (const ch of [r, g, b]) {
        expect(ch).toBeGreaterThanOrEqual(0);
        expect(ch).toBeLessThanOrEqual(255);
        expect(Number.isInteger(ch)).toBe(true);
      }
    }
  });

  it('R channel is monotonically non-decreasing as norm increases', () => {
    const samples = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let prevR = -1;
    for (const n of samples) {
      const [r] = depthColormap(n);
      expect(r).toBeGreaterThanOrEqual(prevR);
      prevR = r;
    }
  });

  it('B channel is monotonically non-increasing as norm increases', () => {
    const samples = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let prevB = 256;
    for (const n of samples) {
      const [, , b] = depthColormap(n);
      expect(b).toBeLessThanOrEqual(prevB);
      prevB = b;
    }
  });
});
