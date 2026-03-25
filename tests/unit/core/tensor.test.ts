import { describe, it, expect } from 'vitest';
import { CpuTensor } from '@core/tensor.ts';
import { dim } from '@core/types.ts';
import type { Shape2D, Shape3D } from '@core/types.ts';

describe('CpuTensor', () => {
  it('should create a 2D tensor with correct shape and layout', () => {
    const shape = [dim(480), dim(640)] as Shape2D;
    const tensor = new CpuTensor(shape, 'f32');

    expect(tensor.rank).toBe(2);
    expect(tensor.numel).toBe(480 * 640);
    expect(tensor.byteLength).toBe(480 * 640 * 4);
    expect(tensor.layout.dtype).toBe('f32');
    expect(tensor.layout.strides).toEqual([640 * 4, 4]);
  });

  it('should create a 3D tensor (image with channels)', () => {
    const shape = [dim(480), dim(640), dim(3)] as Shape3D;
    const tensor = new CpuTensor(shape, 'u8');

    expect(tensor.rank).toBe(3);
    expect(tensor.numel).toBe(480 * 640 * 3);
    expect(tensor.byteLength).toBe(480 * 640 * 3);
    expect(tensor.layout.strides).toEqual([640 * 3, 3, 1]);
  });

  it('should compute correct offset for multi-dimensional indexing', () => {
    const shape = [dim(10), dim(20)] as Shape2D;
    const tensor = new CpuTensor(shape, 'f32');

    // Row 3, Column 5 → flat index = 3 * 20 + 5 = 65
    expect(tensor.offset(3, 5)).toBe(65);
    expect(tensor.offset(0, 0)).toBe(0);
    expect(tensor.offset(9, 19)).toBe(9 * 20 + 19);
  });

  it('should throw on wrong number of indices', () => {
    const shape = [dim(10), dim(20)] as Shape2D;
    const tensor = new CpuTensor(shape, 'f32');

    expect(() => tensor.offset(3)).toThrow('Expected 2 indices, got 1');
    expect(() => tensor.offset(3, 5, 7)).toThrow('Expected 2 indices, got 3');
  });

  it('should create from existing data buffer', () => {
    const data = new ArrayBuffer(12); // 3 × f32
    const view = new Float32Array(data);
    view[0] = 1.0;
    view[1] = 2.0;
    view[2] = 3.0;

    const tensor = new CpuTensor([dim(3)] as readonly [ReturnType<typeof dim>], 'f32', data);
    const f32Data = tensor.data as Float32Array;
    expect(f32Data[0]).toBe(1.0);
    expect(f32Data[1]).toBe(2.0);
    expect(f32Data[2]).toBe(3.0);
  });

  it('should handle different dtypes', () => {
    const u8Tensor = new CpuTensor([dim(4)] as readonly [ReturnType<typeof dim>], 'u8');
    expect(u8Tensor.byteLength).toBe(4);

    const i32Tensor = new CpuTensor([dim(4)] as readonly [ReturnType<typeof dim>], 'i32');
    expect(i32Tensor.byteLength).toBe(16);
  });
});
