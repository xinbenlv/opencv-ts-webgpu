import { WasmError } from '../../core/errors.ts';

/**
 * SharedArrayBuffer bridge between WASM and WebGPU.
 *
 * Enables zero-copy data sharing between the L-BFGS optimizer (WASM)
 * and gradient computation kernels (WebGPU) in the JOSH solver.
 *
 * Requirements:
 * - Cross-Origin Isolation headers must be set (COOP + COEP)
 * - SharedArrayBuffer must be available (checked at construction)
 */
export class SharedMemoryBridge {
  private readonly _buffer: SharedArrayBuffer;
  private readonly _f64View: Float64Array;
  private readonly _f32View: Float32Array;

  constructor(byteLength: number) {
    if (typeof SharedArrayBuffer === 'undefined') {
      throw new WasmError(
        'SharedArrayBuffer is not available. ' +
          'Ensure Cross-Origin Isolation headers are set: ' +
          'Cross-Origin-Opener-Policy: same-origin, ' +
          'Cross-Origin-Embedder-Policy: require-corp',
      );
    }

    // Align to 8 bytes for Float64Array compatibility
    const aligned = Math.ceil(byteLength / 8) * 8;
    this._buffer = new SharedArrayBuffer(aligned);
    this._f64View = new Float64Array(this._buffer);
    this._f32View = new Float32Array(this._buffer);
  }

  /**
   * Get the underlying SharedArrayBuffer for passing to WASM.
   */
  get buffer(): SharedArrayBuffer {
    return this._buffer;
  }

  /**
   * Float64 view — used by L-BFGS optimizer (double precision).
   */
  get f64(): Float64Array {
    return this._f64View;
  }

  /**
   * Float32 view — used by GPU readback (single precision).
   */
  get f32(): Float32Array {
    return this._f32View;
  }

  /**
   * Write GPU readback data (f32) into the shared buffer.
   * The WASM side reads this as f64 with implicit widening.
   */
  writeFromGpu(data: Float32Array, offsetF32 = 0): void {
    this._f32View.set(data, offsetF32);
  }

  /**
   * Read WASM optimizer output (f64) for upload to GPU.
   * Narrows f64 → f32 for GPU consumption.
   */
  readForGpu(lengthF64: number, offsetF64 = 0): Float32Array {
    const result = new Float32Array(lengthF64);
    for (let i = 0; i < lengthF64; i++) {
      result[i] = this._f64View[offsetF64 + i]!;
    }
    return result;
  }

  get byteLength(): number {
    return this._buffer.byteLength;
  }
}
