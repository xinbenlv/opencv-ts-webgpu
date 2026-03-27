/**
 * MASt3R-compatible node backed by MiDaS monocular depth estimation.
 *
 * Preserves the exact public interface (MASt3ROutput + MASt3RNode class) so
 * batch-pipeline.ts requires zero changes.  Instead of loading a 2.75 GB
 * stereo ONNX model, this implementation:
 *
 *   1. Runs MiDaS v2.1-small (64 MB) on each image independently.
 *   2. Converts the relative inverse-depth output into metric-ish depth via
 *      Z = scale / midas_value  (default scale = 5.0 → ~5 m average scene).
 *   3. Unprojects each pixel (u, v) with depth Z into 3-D:
 *        X = (u - cx) * Z / f
 *        Y = (v - cy) * Z / f
 *   4. Builds a confidence map:  conf = min(1, Z_normalised) for valid pixels.
 *
 * Model path / caching:
 *   Loads from `./assets/models/midas-v2.1-small-256.onnx` by default,
 *   matching the path used by DepthEstimationNode.  Falls back to synthetic
 *   gradient depth if the model cannot be fetched.
 *
 * Execution providers:
 *   Tries WebGPU first; falls back to WASM.
 */

import { cachedFetchModel } from '../models/cached-fetch.ts';

// ---------------------------------------------------------------------------
// Public types  (interface MUST stay identical to original)
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

/** MiDaS v2.1-small canonical inference resolution */
const MIDAS_H = 256;
const MIDAS_W = 256;

/**
 * Output resolution for pts3d / conf arrays.
 *
 * Must match the hardcoded `pmW = 512` in batch-pipeline.ts Phase D so that
 * focal recovery reconstructs pixel coordinates correctly.  MiDaS inference
 * runs at 256×256 and the depth map is bilinearly upsampled to this size.
 */
const OUT_H = 512;
const OUT_W = 512;

/** ImageNet normalization constants used by MiDaS */
const MIDAS_MEAN = [0.485, 0.456, 0.406] as const;
const MIDAS_STD  = [0.229, 0.224, 0.225] as const;

/**
 * Scale factor: Z = DEPTH_SCALE / midas_value.
 * MiDaS outputs relative inverse depth (higher = closer), so dividing the
 * scale by the raw value converts it to a metric-ish absolute depth in metres.
 * A value of 5.0 places the average scene roughly 5 m from the camera.
 */
const DEPTH_SCALE = 5.0;

// ---------------------------------------------------------------------------
// MASt3RNode
// ---------------------------------------------------------------------------

export class MASt3RNode {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private midasSession: any = null;
  private readonly modelPath: string;
  private focalLength: number;

  constructor(
    modelPath = './assets/models/midas-v2.1-small-256.onnx',
    focalLength = 500,
  ) {
    this.modelPath = modelPath;
    this.focalLength = focalLength;
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Load the MiDaS ONNX model.
   *
   * Uses the Cache API so the ~64 MB file is only downloaded once.
   * Falls back to synthetic depth if the model cannot be fetched — process()
   * will still return valid (synthetic) point maps.
   *
   * @param onProgress  Optional callback receiving a [0, 1] download fraction.
   */
  async load(onProgress?: (fraction: number) => void): Promise<void> {
    if (this.midasSession !== null) return;

    const ort = await import('onnxruntime-web');

    ort.env.wasm.wasmPaths = '/assets/ort/';
    ort.env.wasm.numThreads = 1;

    const statusFn = onProgress
      ? (_id: string, _status: string, text: string) => {
          const match = text.match(/(\d+)%/);
          if (match) {
            onProgress(parseInt(match[1]!, 10) / 100);
          }
        }
      : undefined;

    console.log('[MASt3RNode] Loading MiDaS model from:', this.modelPath);

    let modelBuf: ArrayBuffer;
    try {
      modelBuf = await cachedFetchModel(
        this.modelPath,
        'midasModelForMast3r',
        'MiDaS v2.1-small depth model',
        '64 MB',
        statusFn,
      );
    } catch (e) {
      console.warn(
        '[MASt3RNode] Could not fetch MiDaS model — will use synthetic depth.',
        e,
      );
      onProgress?.(1);
      return; // midasSession stays null; process() uses fallback
    }

    let ep = 'wasm';
    try {
      this.midasSession = await ort.InferenceSession.create(modelBuf, {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
      });
      ep = 'webgpu';
      console.log('[MASt3RNode] Using WebGPU execution provider');
    } catch (e) {
      console.warn('[MASt3RNode] WebGPU EP unavailable, falling back to WASM:', e);
      this.midasSession = await ort.InferenceSession.create(modelBuf, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
    }

    onProgress?.(1);
    console.log(`[MASt3RNode] MiDaS model ready (${ep.toUpperCase()})`);
  }

  /**
   * Produce dense 3-D point maps for a pair of frames.
   *
   * Each image is processed independently through MiDaS.  The unprojection
   * uses the current focalLength and assumes the principal point at image
   * centre.  Output width/height match the input images (after resizing to
   * MIDAS_W × MIDAS_H internally).
   *
   * @param image1  First frame  — ImageData or HTMLCanvasElement at any resolution.
   * @param image2  Second frame — ImageData or HTMLCanvasElement at any resolution.
   */
  async process(
    image1: ImageData | HTMLCanvasElement,
    image2: ImageData | HTMLCanvasElement,
  ): Promise<MASt3ROutput> {
    const [pts3d_1, conf_1] = await this._processOne(image1);
    const [pts3d_2, conf_2] = await this._processOne(image2);

    return {
      pts3d_1,
      pts3d_2,
      conf_1,
      conf_2,
      width: OUT_W,
      height: OUT_H,
    };
  }

  /**
   * Update the focal length used for unprojection.
   * Called by batch-pipeline.ts after focal-length recovery.
   */
  setFocalLength(f: number): void {
    this.focalLength = f;
  }

  /** Whether the MiDaS session has been loaded. */
  isLoaded(): boolean {
    return this.midasSession !== null;
  }

  /** Release the ONNX session and free memory. */
  dispose(): void {
    if (this.midasSession !== null) {
      try {
        // eslint-disable-next-line @typescript-eslint/no-unsafe-call
        this.midasSession.release();
      } catch {
        /* best-effort */
      }
      this.midasSession = null;
    }
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  /**
   * Run MiDaS on a single image and return a [pts3d, conf] pair.
   *
   * If midasSession is null the fallback produces a synthetic gradient depth
   * (darker = farther) so the pipeline can still run end-to-end without a
   * model file.
   *
   * The 256×256 MiDaS output is bilinearly upsampled to OUT_W × OUT_H
   * (512×512) so that the pts3d / conf arrays are compatible with the
   * hardcoded `pmW = 512` expectation in batch-pipeline.ts.
   */
  private async _processOne(
    image: ImageData | HTMLCanvasElement,
  ): Promise<[Float32Array, Float32Array]> {
    const imageData = this._toImageData(image, MIDAS_W, MIDAS_H);

    let rawDepth256: Float32Array; // [MIDAS_H * MIDAS_W], relative inverse depth

    if (this.midasSession !== null) {
      rawDepth256 = await this._runMiDaS(imageData);
    } else {
      rawDepth256 = this._syntheticDepth(imageData);
    }

    // Upsample depth from MIDAS resolution to output resolution
    const rawDepth = this._upsampleDepth(rawDepth256, MIDAS_W, MIDAS_H, OUT_W, OUT_H);

    return this._unprojectDepth(rawDepth, OUT_W, OUT_H);
  }

  /**
   * Run a single MiDaS forward pass.
   *
   * Input:  [1, 3, MIDAS_H, MIDAS_W] float32 (ImageNet-normalised, NCHW)
   * Output: [1, MIDAS_H, MIDAS_W]    float32 (relative inverse depth)
   */
  private async _runMiDaS(imageData: ImageData): Promise<Float32Array> {
    const ort = await import('onnxruntime-web');

    const { data } = imageData; // Uint8ClampedArray RGBA
    const nchwInput = new Float32Array(3 * MIDAS_H * MIDAS_W);

    for (let y = 0; y < MIDAS_H; y++) {
      for (let x = 0; x < MIDAS_W; x++) {
        const pixelBase = (y * MIDAS_W + x) * 4;
        const pixelIdx  = y * MIDAS_W + x;

        for (let c = 0; c < 3; c++) {
          const raw = data[pixelBase + c]! / 255.0;
          nchwInput[c * MIDAS_H * MIDAS_W + pixelIdx] =
            (raw - MIDAS_MEAN[c]!) / MIDAS_STD[c]!;
        }
      }
    }

    const inputTensor = new ort.Tensor('float32', nchwInput, [1, 3, MIDAS_H, MIDAS_W]);
    const feeds: Record<string, InstanceType<typeof ort.Tensor>> = {};
    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
    feeds[this.midasSession.inputNames[0] as string] = inputTensor;

    // eslint-disable-next-line @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access
    const results = await this.midasSession.run(feeds);
    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
    const outputTensor = results[this.midasSession.outputNames[0] as string]!;
    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
    return outputTensor.data as Float32Array;
  }

  /**
   * Fallback: synthetic gradient depth based on image brightness.
   * Brighter pixels → smaller inverse-depth value → farther away.
   * This allows the pipeline to produce plausible (but non-metric) point maps
   * even when the MiDaS model is unavailable.
   */
  private _syntheticDepth(imageData: ImageData): Float32Array {
    const { data } = imageData;
    const N = MIDAS_H * MIDAS_W;
    const depth = new Float32Array(N);

    for (let i = 0; i < N; i++) {
      const base = i * 4;
      const luma =
        0.2126 * (data[base]!     / 255.0) +
        0.7152 * (data[base + 1]! / 255.0) +
        0.0722 * (data[base + 2]! / 255.0);
      // Invert so dark (foreground) → high value → closer
      depth[i] = 1.0 - luma + 0.01; // add small epsilon to avoid division by zero
    }

    return depth;
  }

  /**
   * Bilinear upsampling of a flat depth map from srcW×srcH to dstW×dstH.
   */
  private _upsampleDepth(
    src: Float32Array,
    srcW: number,
    srcH: number,
    dstW: number,
    dstH: number,
  ): Float32Array {
    if (srcW === dstW && srcH === dstH) return src;

    const dst = new Float32Array(dstH * dstW);

    for (let y = 0; y < dstH; y++) {
      for (let x = 0; x < dstW; x++) {
        const sx = (x / dstW) * srcW;
        const sy = (y / dstH) * srcH;
        const x0 = Math.floor(sx);
        const y0 = Math.floor(sy);
        const x1 = Math.min(x0 + 1, srcW - 1);
        const y1 = Math.min(y0 + 1, srcH - 1);
        const fx = sx - x0;
        const fy = sy - y0;

        const v00 = src[y0 * srcW + x0]!;
        const v10 = src[y0 * srcW + x1]!;
        const v01 = src[y1 * srcW + x0]!;
        const v11 = src[y1 * srcW + x1]!;

        dst[y * dstW + x] =
          v00 * (1 - fx) * (1 - fy) +
          v10 * fx       * (1 - fy) +
          v01 * (1 - fx) * fy +
          v11 * fx       * fy;
      }
    }

    return dst;
  }

  /**
   * Unproject a flat depth map into a 3-D point cloud.
   *
   * MiDaS outputs relative inverse depth d where higher d = closer.
   * We convert to metric depth:  Z = DEPTH_SCALE / d
   *
   * Then for each pixel (u, v):
   *   X = (u - cx) * Z / f
   *   Y = (v - cy) * Z / f
   *
   * Confidence is set to min(1, Z_norm) where Z_norm is Z normalised by the
   * maximum depth in the map.  Pixels with zero or invalid depth get conf = 0.
   *
   * @returns [pts3d, conf]  Flat arrays of length H*W*3 and H*W respectively.
   */
  private _unprojectDepth(
    rawDepth: Float32Array,
    W: number,
    H: number,
  ): [Float32Array, Float32Array] {
    const N = H * W;
    const pts3d = new Float32Array(N * 3);
    const conf  = new Float32Array(N);

    const f  = this.focalLength;
    const cx = W / 2;
    const cy = H / 2;

    // Convert inverse-depth → metric depth and find max for normalisation
    const metricDepth = new Float32Array(N);
    let maxZ = 0;
    for (let i = 0; i < N; i++) {
      const d = rawDepth[i]!;
      const Z = d > 0 ? DEPTH_SCALE / d : 0;
      metricDepth[i] = Z;
      if (Z > maxZ) maxZ = Z;
    }

    const invMaxZ = maxZ > 0 ? 1.0 / maxZ : 1.0;

    for (let v = 0; v < H; v++) {
      for (let u = 0; u < W; u++) {
        const idx = v * W + u;
        const Z   = metricDepth[idx]!;

        if (Z <= 0) {
          // Invalid depth — leave pts3d at [0,0,0], conf at 0
          continue;
        }

        pts3d[idx * 3]     = (u - cx) * Z / f; // X
        pts3d[idx * 3 + 1] = (v - cy) * Z / f; // Y
        pts3d[idx * 3 + 2] = Z;                 // Z

        // Confidence: reward closer (lower Z) points
        conf[idx] = Math.min(1.0, Z * invMaxZ);
      }
    }

    return [pts3d, conf];
  }

  /**
   * Convert ImageData or HTMLCanvasElement to an ImageData at the given size.
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
      const tmp = document.createElement('canvas');
      tmp.width = source.width;
      tmp.height = source.height;
      tmp.getContext('2d')!.putImageData(source, 0, 0);

      const canvas = document.createElement('canvas');
      canvas.width = targetW;
      canvas.height = targetH;
      canvas.getContext('2d')!.drawImage(tmp, 0, 0, targetW, targetH);
      return canvas.getContext('2d')!.getImageData(0, 0, targetW, targetH);
    }

    // HTMLCanvasElement
    const canvas = document.createElement('canvas');
    canvas.width = targetW;
    canvas.height = targetH;
    canvas.getContext('2d')!.drawImage(source, 0, 0, targetW, targetH);
    return canvas.getContext('2d')!.getImageData(0, 0, targetW, targetH);
  }
}
