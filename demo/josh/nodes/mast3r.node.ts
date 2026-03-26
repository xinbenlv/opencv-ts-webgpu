/**
 * Phase 1B: MASt3R dense stereo matching node.
 *
 * Loads the MASt3R ViT-Large ONNX model and runs pairwise inference on two
 * 512×512 RGB frames, producing dense 3D point maps and per-pixel confidence
 * scores.
 *
 * Model I/O (as exported by scripts/export-mast3r.py):
 *   Inputs:
 *     image1  [1, 3, 512, 512] float32 in [-1, 1]  (NCHW)
 *     image2  [1, 3, 512, 512] float32 in [-1, 1]  (NCHW)
 *   Outputs:
 *     pts3d_1  [1, 512, 512, 3]  point map for frame 1
 *     pts3d_2  [1, 512, 512, 3]  point map for frame 2
 *     conf_1   [1, 512, 512]     confidence for frame 1
 *     conf_2   [1, 512, 512]     confidence for frame 2
 *
 * Large-file loading strategy:
 *   - First load: streams the ONNX file from the server with progress callback
 *   - Subsequent loads: served instantly from the Cache API (persistent on disk)
 *
 * Execution provider:
 *   - Tries WebGPU EP first; falls back to WASM if unavailable
 */

import { cachedFetchModel } from '../models/cached-fetch.ts';

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface MASt3ROutput {
  /** Flattened XYZ point map for frame 1, length H*W*3 */
  pts3d_1: Float32Array;
  /** Flattened XYZ point map for frame 2, length H*W*3 */
  pts3d_2: Float32Array;
  /** Per-pixel confidence for frame 1, length H*W */
  conf_1: Float32Array;
  /** Per-pixel confidence for frame 2, length H*W */
  conf_2: Float32Array;
  width: number;
  height: number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MODEL_SIZE = 512; // MASt3R canonical resolution

// ---------------------------------------------------------------------------
// MASt3RNode
// ---------------------------------------------------------------------------

export class MASt3RNode {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private _session: any = null;
  private readonly _modelPath: string;

  constructor(modelPath = '/models/mast3r-vit-large-fp32.onnx') {
    this._modelPath = modelPath;
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Lazy-load the ONNX model.
   *
   * Uses the Cache API so the ~600 MB file is only downloaded once.
   * An optional progress callback receives a [0, 1] fraction during download.
   *
   * @param onProgress  Called with download fraction in [0, 1] during first fetch.
   */
  async load(onProgress?: (fraction: number) => void): Promise<void> {
    if (this._session !== null) return;

    const ort = await import('onnxruntime-web');

    // Match the wasmPaths used by the rest of the pipeline
    ort.env.wasm.wasmPaths = '/assets/ort/';
    ort.env.wasm.numThreads = 1;

    // Build a StatusFn-compatible wrapper around the simpler progress callback
    // so we can reuse cachedFetchModel (which has retry logic and Cache API
    // persistence built in).
    const statusFn = onProgress
      ? (_id: string, _status: string, text: string) => {
          // Extract percentage from the status text produced by cachedFetchModel
          const match = text.match(/(\d+)%/);
          if (match) {
            const pct = parseInt(match[1]!, 10);
            onProgress(pct / 100);
          }
        }
      : undefined;

    console.log('[MASt3RNode] Loading model from:', this._modelPath);

    const modelBuf = await cachedFetchModel(
      this._modelPath,
      'mast3rModel',
      'MASt3R ViT-Large stereo model',
      '~600 MB',
      statusFn,
    );

    // Prefer WebGPU execution provider for GPU-accelerated inference
    let ep = 'wasm';
    try {
      this._session = await ort.InferenceSession.create(modelBuf, {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
      });
      ep = 'webgpu';
      console.log('[MASt3RNode] Using WebGPU execution provider');
    } catch (e) {
      console.warn('[MASt3RNode] WebGPU EP unavailable, falling back to WASM:', e);
      this._session = await ort.InferenceSession.create(modelBuf, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
    }

    onProgress?.(1);
    console.log(`[MASt3RNode] Model ready (${ep.toUpperCase()})`);
  }

  /**
   * Run MASt3R on a pair of frames.
   *
   * Both images are resized to 512×512, normalized to [-1, 1], and packed
   * into NCHW tensors before inference.
   *
   * @param image1  First frame — ImageData or HTMLCanvasElement at any resolution
   * @param image2  Second frame — ImageData or HTMLCanvasElement at any resolution
   */
  async process(
    image1: ImageData | HTMLCanvasElement,
    image2: ImageData | HTMLCanvasElement,
  ): Promise<MASt3ROutput> {
    if (this._session === null) {
      throw new Error('[MASt3RNode] Model not loaded — call load() first');
    }

    const ort = await import('onnxruntime-web');

    const tensor1 = this._preprocess(image1);
    const tensor2 = this._preprocess(image2);

    const feeds: Record<string, InstanceType<typeof ort.Tensor>> = {
      image1: new ort.Tensor('float32', tensor1, [1, 3, MODEL_SIZE, MODEL_SIZE]),
      image2: new ort.Tensor('float32', tensor2, [1, 3, MODEL_SIZE, MODEL_SIZE]),
    };

    const results = await this._session.run(feeds);

    // Output tensors: pts3d_1 [1,H,W,3], pts3d_2 [1,H,W,3],
    //                 conf_1  [1,H,W],   conf_2  [1,H,W]
    const pts3d_1 = (results['pts3d_1']!.data as Float32Array).slice();
    const pts3d_2 = (results['pts3d_2']!.data as Float32Array).slice();
    const conf_1  = (results['conf_1']!.data  as Float32Array).slice();
    const conf_2  = (results['conf_2']!.data  as Float32Array).slice();

    return {
      pts3d_1,
      pts3d_2,
      conf_1,
      conf_2,
      width: MODEL_SIZE,
      height: MODEL_SIZE,
    };
  }

  /** Whether the model session has been loaded. */
  isLoaded(): boolean {
    return this._session !== null;
  }

  /** Release the ONNX session and free memory. */
  dispose(): void {
    if (this._session !== null) {
      try {
        // eslint-disable-next-line @typescript-eslint/no-unsafe-call
        this._session.release();
      } catch {
        /* best-effort */
      }
      this._session = null;
    }
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  /**
   * Preprocess an image for MASt3R:
   *   1. Rasterize to 512×512 pixels via a temporary canvas
   *   2. Normalize RGB from [0, 255] to [-1, 1]
   *   3. Rearrange HWC → NCHW and return as Float32Array [1*3*512*512]
   */
  private _preprocess(image: ImageData | HTMLCanvasElement): Float32Array {
    const H = MODEL_SIZE;
    const W = MODEL_SIZE;

    // Obtain an ImageData at the target resolution
    const imageData = this._toImageData(image, W, H);

    const { data } = imageData; // Uint8ClampedArray RGBA
    const out = new Float32Array(3 * H * W);

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const pixelBase = (y * W + x) * 4;
        const r = (data[pixelBase]!     / 127.5) - 1.0;
        const g = (data[pixelBase + 1]! / 127.5) - 1.0;
        const b = (data[pixelBase + 2]! / 127.5) - 1.0;

        // NCHW layout: channel c at offset c*H*W + y*W + x
        const pixelIdx = y * W + x;
        out[0 * H * W + pixelIdx] = r;
        out[1 * H * W + pixelIdx] = g;
        out[2 * H * W + pixelIdx] = b;
      }
    }

    return out;
  }

  /**
   * Convert ImageData or HTMLCanvasElement to an ImageData at the given size.
   * If the source is already the right size this avoids an extra copy.
   */
  private _toImageData(
    source: ImageData | HTMLCanvasElement,
    targetW: number,
    targetH: number,
  ): ImageData {
    if (source instanceof ImageData) {
      if (source.width === targetW && source.height === targetH) {
        return source;
      }
      // Need to resize: paint onto an off-screen canvas
      const canvas = document.createElement('canvas');
      canvas.width = targetW;
      canvas.height = targetH;
      const ctx = canvas.getContext('2d')!;

      // Put the source ImageData into a temporary canvas first
      const tmp = document.createElement('canvas');
      tmp.width = source.width;
      tmp.height = source.height;
      const tmpCtx = tmp.getContext('2d')!;
      tmpCtx.putImageData(source, 0, 0);

      ctx.drawImage(tmp, 0, 0, targetW, targetH);
      return ctx.getImageData(0, 0, targetW, targetH);
    }

    // HTMLCanvasElement — draw with scaling
    const canvas = document.createElement('canvas');
    canvas.width = targetW;
    canvas.height = targetH;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(source, 0, 0, targetW, targetH);
    return ctx.getImageData(0, 0, targetW, targetH);
  }
}
