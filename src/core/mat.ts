import type { DType, Dim, Shape2D, Shape3D } from './types.ts';
import { dim } from './types.ts';
import { CpuTensor } from './tensor.ts';
import type { Disposable } from './resource-tracker.ts';

/**
 * Image channel count constants.
 */
export const CV_GRAY = 1;
export const CV_RGB = 3;
export const CV_RGBA = 4;

export type Channels = typeof CV_GRAY | typeof CV_RGB | typeof CV_RGBA;

/**
 * cv.Mat compatible wrapper built on top of CpuTensor.
 *
 * Provides familiar OpenCV-style accessors (rows, cols, channels)
 * while delegating storage to the tensor system.
 */
export class Mat implements Disposable {
  private readonly _tensor: CpuTensor<Shape2D | Shape3D>;
  private _disposed = false;

  private constructor(tensor: CpuTensor<Shape2D | Shape3D>) {
    this._tensor = tensor;
  }

  /**
   * Create a new Mat with the given dimensions and type.
   */
  static create(rows: number, cols: number, dtype: DType, channels: Channels = CV_GRAY): Mat {
    const h = dim(rows);
    const w = dim(cols);
    if (channels === CV_GRAY) {
      return new Mat(new CpuTensor([h, w] as Shape2D, dtype));
    }
    return new Mat(new CpuTensor([h, w, dim(channels)] as Shape3D, dtype));
  }

  /**
   * Create a Mat from existing pixel data.
   */
  static fromData(
    rows: number,
    cols: number,
    dtype: DType,
    channels: Channels,
    data: ArrayBuffer,
  ): Mat {
    const h = dim(rows);
    const w = dim(cols);
    if (channels === CV_GRAY) {
      return new Mat(new CpuTensor([h, w] as Shape2D, dtype, data));
    }
    return new Mat(new CpuTensor([h, w, dim(channels)] as Shape3D, dtype, data));
  }

  get rows(): Dim {
    return this._tensor.shape[0];
  }

  get cols(): Dim {
    return this._tensor.shape[1];
  }

  get channels(): number {
    return this._tensor.shape.length === 3 ? this._tensor.shape[2] : 1;
  }

  get dtype(): DType {
    return this._tensor.dtype;
  }

  get data(): ArrayBufferView {
    return this._tensor.data;
  }

  get byteLength(): number {
    return this._tensor.byteLength;
  }

  get tensor(): CpuTensor<Shape2D | Shape3D> {
    return this._tensor;
  }

  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;
    this._tensor.dispose();
  }
}
