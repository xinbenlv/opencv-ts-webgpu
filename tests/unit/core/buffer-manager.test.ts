import { describe, it, expect } from 'vitest';
import { computeBufferLayout, dim } from '@core/types.ts';
import type { Shape2D } from '@core/types.ts';

describe('computeBufferLayout', () => {
  it('should compute correct layout for f32 2D tensor', () => {
    const layout = computeBufferLayout([dim(480), dim(640)] as Shape2D, 'f32');

    expect(layout.dtype).toBe('f32');
    expect(layout.byteLength).toBe(480 * 640 * 4);
    expect(layout.strides).toEqual([640 * 4, 4]);
  });

  it('should compute correct layout for u8 3D tensor', () => {
    const layout = computeBufferLayout(
      [dim(480), dim(640), dim(3)] as readonly [ReturnType<typeof dim>, ReturnType<typeof dim>, ReturnType<typeof dim>],
      'u8',
    );

    expect(layout.dtype).toBe('u8');
    expect(layout.byteLength).toBe(480 * 640 * 3);
    expect(layout.strides).toEqual([640 * 3, 3, 1]);
  });

  it('should compute correct layout for 1D tensor', () => {
    const layout = computeBufferLayout(
      [dim(100)] as readonly [ReturnType<typeof dim>],
      'f32',
    );

    expect(layout.byteLength).toBe(100 * 4);
    expect(layout.strides).toEqual([4]);
  });
});

describe('dim', () => {
  it('should create a valid dimension', () => {
    expect(dim(1)).toBe(1);
    expect(dim(1920)).toBe(1920);
  });

  it('should reject non-positive values', () => {
    expect(() => dim(0)).toThrow(RangeError);
    expect(() => dim(-1)).toThrow(RangeError);
    expect(() => dim(1.5)).toThrow(RangeError);
  });
});
