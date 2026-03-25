// ─── Branded type foundation ───
// Prevents accidental mixing of structurally identical types at compile time.

declare const __brand: unique symbol;
type Brand<T, B extends string> = T & { readonly [__brand]: B };

// ─── Dimension ───
// A positive integer representing a single axis size.

export type Dim = Brand<number, 'Dim'>;

export function dim(n: number): Dim {
  if (!Number.isInteger(n) || n <= 0) {
    throw new RangeError(`Invalid dimension: ${n}. Must be a positive integer.`);
  }
  return n as Dim;
}

// ─── Tensor shapes ───
// Tuple length encodes rank; each element is a Dim.

export type Shape1D = readonly [Dim];
export type Shape2D = readonly [Dim, Dim]; // [H, W]
export type Shape3D = readonly [Dim, Dim, Dim]; // [H, W, C]
export type Shape4D = readonly [Dim, Dim, Dim, Dim]; // [N, H, W, C]
export type TensorShape = Shape1D | Shape2D | Shape3D | Shape4D;

// ─── Scalar element types ───

export type DType = 'f32' | 'f16' | 'u8' | 'u32' | 'i32';

export const DTYPE_BYTES: Readonly<Record<DType, number>> = {
  f32: 4,
  f16: 2,
  u8: 1,
  u32: 4,
  i32: 4,
};

// ─── Buffer layout descriptor ───
// Fully describes the memory layout of a tensor for GPU binding.

export interface BufferLayout {
  readonly dtype: DType;
  readonly shape: TensorShape;
  readonly strides: readonly number[];
  readonly byteLength: number;
}

/**
 * Compute a row-major (C-contiguous) BufferLayout from shape and dtype.
 */
export function computeBufferLayout(shape: TensorShape, dtype: DType): BufferLayout {
  const elemBytes = DTYPE_BYTES[dtype];
  const strides: number[] = new Array(shape.length);
  let stride = elemBytes;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i]!;
  }
  const numel = shape.reduce<number>((acc, d) => acc * d, 1);
  return {
    dtype,
    shape,
    strides,
    byteLength: numel * elemBytes,
  };
}

// ─── Branded identifiers ───

export type BufferId = Brand<string, 'BufferId'>;
export type NodeId = Brand<string, 'NodeId'>;
export type GraphId = Brand<string, 'GraphId'>;

let _idCounter = 0;

export function createBufferId(label?: string): BufferId {
  return `buf_${label ?? ''}${_idCounter++}` as BufferId;
}

export function createNodeId(name: string): NodeId {
  return `node_${name}_${_idCounter++}` as NodeId;
}

export function createGraphId(name: string): GraphId {
  return `graph_${name}_${_idCounter++}` as GraphId;
}
