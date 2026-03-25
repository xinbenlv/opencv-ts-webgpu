import type { BufferLayout, DType, TensorShape } from './types.ts';
import { DTYPE_BYTES, computeBufferLayout } from './types.ts';
import type { Disposable } from './resource-tracker.ts';

// TypedArray type mapping for each DType
type TypedArrayFor<D extends DType> = D extends 'f32'
  ? Float32Array
  : D extends 'f16'
    ? Uint16Array // f16 stored as raw u16 bits
    : D extends 'u8'
      ? Uint8Array
      : D extends 'u32'
        ? Uint32Array
        : D extends 'i32'
          ? Int32Array
          : never;

function createTypedArray(dtype: DType, buffer: ArrayBuffer): ArrayBufferView {
  switch (dtype) {
    case 'f32':
      return new Float32Array(buffer);
    case 'f16':
      return new Uint16Array(buffer);
    case 'u8':
      return new Uint8Array(buffer);
    case 'u32':
      return new Uint32Array(buffer);
    case 'i32':
      return new Int32Array(buffer);
  }
}

/**
 * CPU-resident tensor with typed data and shape metadata.
 */
export class CpuTensor<S extends TensorShape = TensorShape, D extends DType = DType>
  implements Disposable
{
  readonly layout: BufferLayout;
  readonly data: TypedArrayFor<D>;
  private _disposed = false;

  constructor(
    readonly shape: S,
    readonly dtype: D,
    data?: ArrayBuffer,
  ) {
    this.layout = computeBufferLayout(shape, dtype);
    const buf = data ?? new ArrayBuffer(this.layout.byteLength);
    this.data = createTypedArray(dtype, buf) as TypedArrayFor<D>;
  }

  get rank(): number {
    return this.shape.length;
  }

  get numel(): number {
    return this.shape.reduce<number>((acc, d) => acc * d, 1);
  }

  get byteLength(): number {
    return this.layout.byteLength;
  }

  /**
   * Get the flat index for a multi-dimensional coordinate.
   */
  offset(...indices: number[]): number {
    if (indices.length !== this.shape.length) {
      throw new RangeError(
        `Expected ${this.shape.length} indices, got ${indices.length}`,
      );
    }
    let off = 0;
    const elemBytes = DTYPE_BYTES[this.dtype];
    for (let i = 0; i < indices.length; i++) {
      off += indices[i]! * (this.layout.strides[i]! / elemBytes);
    }
    return off;
  }

  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;
    // ArrayBuffer will be GC'd — nothing explicit to free
  }
}

/**
 * Handle to a GPU-resident tensor. Data lives in a GPUBuffer managed by BufferManager.
 * This is a lightweight descriptor — the actual buffer is owned by BufferManager.
 */
export class GpuTensor<S extends TensorShape = TensorShape, D extends DType = DType>
  implements Disposable
{
  readonly layout: BufferLayout;
  private _disposed = false;

  constructor(
    readonly shape: S,
    readonly dtype: D,
    readonly gpuBuffer: GPUBuffer,
  ) {
    this.layout = computeBufferLayout(shape, dtype);
  }

  get rank(): number {
    return this.shape.length;
  }

  get numel(): number {
    return this.shape.reduce<number>((acc, d) => acc * d, 1);
  }

  get byteLength(): number {
    return this.layout.byteLength;
  }

  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;
    this.gpuBuffer.destroy();
  }
}
