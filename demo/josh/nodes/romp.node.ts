/**
 * ROMPNode — SMPL parameter estimation from a single image using a ROMP/BEV
 * ONNX model served at /models/romp-bev-fp32.onnx.
 *
 * ONNX interface
 * --------------
 *   Input  : "image"  [1, 3, 512, 512]  normalised RGB (float32)
 *   Outputs: "pose"   [1, 72]           axis-angle pose (24 joints × 3)
 *            "betas"  [1, 10]           SMPL shape coefficients
 *            "cam"    [1, 3]            weak-perspective camera (scale, tx, ty)
 *
 * Usage
 * -----
 *   const node = new ROMPNode();
 *   await node.load();
 *   const result = await node.estimate(imageData);
 *   if (result) { ... }
 */

import * as ort from 'onnxruntime-web';
import { JOSH_CONFIG } from '../config.ts';
import { cachedFetchModel } from '../models/cached-fetch.ts';

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface ROMPOutput {
  /** Axis-angle pose parameters for 24 SMPL joints [72] */
  pose: Float32Array;
  /** SMPL shape coefficients [10] */
  betas: Float32Array;
  /** Weak-perspective camera [scale, tx, ty] */
  cam: Float32Array;
  /**
   * Confidence of the top-1 detection in [0, 1].
   * Derived from the maximum centre-map value returned by the model, OR 1.0
   * when the model outputs flat parameter tensors directly.
   */
  confidence: number;
}

// ---------------------------------------------------------------------------
// ImageData extraction helper (works on both ImageData and HTMLCanvasElement)
// ---------------------------------------------------------------------------

function getImageData(src: ImageData | HTMLCanvasElement): ImageData {
  if (src instanceof ImageData) return src;
  const ctx = src.getContext('2d');
  if (!ctx) throw new Error('[ROMPNode] Cannot get 2D context from canvas');
  return ctx.getImageData(0, 0, src.width, src.height);
}

// ---------------------------------------------------------------------------
// ORT lazy loader
// ---------------------------------------------------------------------------

let _ortModule: typeof ort | null = null;

async function getOrt(): Promise<typeof ort> {
  if (!_ortModule) {
    _ortModule = await import('onnxruntime-web');
  }
  return _ortModule;
}

// ---------------------------------------------------------------------------
// ROMPNode
// ---------------------------------------------------------------------------

const ROMP_SIZE = JOSH_CONFIG.rompInputSize; // 512

/** ImageNet normalisation constants (matching ROMP preprocessing). */
const MEAN = [0.485, 0.456, 0.406] as const;
const STD  = [0.229, 0.224, 0.225] as const;

export class ROMPNode {
  private _session: ort.InferenceSession | null = null;
  private _loading = false;
  private _loadError: string | null = null;

  constructor(private readonly modelPath = '/models/romp-bev-fp32.onnx') {}

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Lazy-load the ONNX model.
   * Safe to call multiple times — subsequent calls are no-ops once loaded.
   * Throws if the model cannot be fetched or the session cannot be created.
   */
  async load(): Promise<void> {
    if (this._session !== null) return;
    if (this._loading) return; // concurrent call — do nothing, let first caller finish
    if (this._loadError !== null) throw new Error(this._loadError);

    this._loading = true;
    try {
      const o = await getOrt();
      o.env.wasm.wasmPaths = './assets/ort/';
      o.env.wasm.numThreads = 1;

      const statusFn: ((id: string, s: string, t: string) => void) | undefined =
        (globalThis as Record<string, unknown>).__joshLoadingStatus as
          | ((id: string, s: string, t: string) => void)
          | undefined;

      const modelBuffer = await cachedFetchModel(
        this.modelPath,
        'rompModel',
        'ROMP: SMPL pose estimator',
        '~150 MB',
        statusFn,
      );

      statusFn?.('rompModel', 'active', 'ROMP: creating ONNX session…');

      this._session = await o.InferenceSession.create(modelBuffer, {
        executionProviders: ['webgpu', 'wasm'],
        graphOptimizationLevel: 'all',
      });

      statusFn?.('rompModel', 'done', 'ROMP: ready');
      console.log(
        '[ROMPNode] model loaded —',
        this._session.inputNames,
        '→',
        this._session.outputNames,
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this._loadError = msg;
      console.warn('[ROMPNode] Failed to load ONNX model:', msg);
      throw err;
    } finally {
      this._loading = false;
    }
  }

  /** Returns true once the ONNX session is ready to run inference. */
  isLoaded(): boolean {
    return this._session !== null;
  }

  /**
   * Estimate SMPL parameters for the dominant person in the image.
   *
   * @param image  512×512 (or any size — will be resized) RGB image source.
   * @returns      ROMPOutput for the top-1 detection, or null if the model is
   *               not loaded or no person was detected (confidence < 0.01).
   */
  async estimate(image: ImageData | HTMLCanvasElement): Promise<ROMPOutput | null> {
    if (this._session === null) {
      console.warn('[ROMPNode] estimate() called before load() — returning null');
      return null;
    }

    const o = await getOrt();

    // Preprocess
    const inputData = this.preprocess(image);
    const inputTensor = new o.Tensor('float32', inputData, [1, 3, ROMP_SIZE, ROMP_SIZE]);

    const feeds: Record<string, ort.Tensor> = {};
    feeds[this._session.inputNames[0]!] = inputTensor;

    const results = await this._session.run(feeds);

    return this._parseOutputs(results);
  }

  /**
   * Release the ONNX inference session and free associated resources.
   * The node can be re-loaded after disposal by calling load() again.
   */
  dispose(): void {
    this._session?.release();
    this._session = null;
    this._loadError = null;
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  /**
   * Convert an image source to a Float32Array of shape [1, 3, 512, 512]
   * (NCHW) with ImageNet normalisation applied.
   *
   * Steps:
   *   1. Extract raw RGBA pixels
   *   2. Bilinear-resize to ROMP_SIZE × ROMP_SIZE
   *   3. Convert to float, divide by 255
   *   4. Normalise: (x - mean) / std
   *   5. Reorder HWC → CHW
   */
  preprocess(image: ImageData | HTMLCanvasElement): Float32Array {
    const src = getImageData(image);
    const resized = _resizeCpu(src, ROMP_SIZE);

    const numPixels = ROMP_SIZE * ROMP_SIZE;
    const out = new Float32Array(3 * numPixels); // [C, H, W]

    for (let i = 0; i < numPixels; i++) {
      const r = resized.data[i * 4    ]! / 255;
      const g = resized.data[i * 4 + 1]! / 255;
      const b = resized.data[i * 4 + 2]! / 255;

      out[0 * numPixels + i] = (r - MEAN[0]) / STD[0];
      out[1 * numPixels + i] = (g - MEAN[1]) / STD[1];
      out[2 * numPixels + i] = (b - MEAN[2]) / STD[2];
    }

    return out;
  }

  /**
   * Parse the raw ONNX output tensors into a typed ROMPOutput.
   *
   * The exported ONNX model already extracts the top-1 detection, so outputs
   * are flat [1, N] tensors.  We just slice the data.
   */
  private _parseOutputs(
    results: Record<string, ort.Tensor>,
  ): ROMPOutput | null {
    const poseOut  = results['pose']  ?? results[this._session!.outputNames[0]!];
    const betasOut = results['betas'] ?? results[this._session!.outputNames[1]!];
    const camOut   = results['cam']   ?? results[this._session!.outputNames[2]!];

    if (!poseOut || !betasOut || !camOut) {
      console.warn('[ROMPNode] Unexpected output names:', Object.keys(results));
      return null;
    }

    const poseData  = poseOut.data  as Float32Array;
    const betasData = betasOut.data as Float32Array;
    const camData   = camOut.data   as Float32Array;

    // Confidence: use the camera scale (first cam value) as a proxy.
    // A scale near zero implies no meaningful detection.
    const scale = camData[0] ?? 0;
    const confidence = Math.min(1, Math.max(0, scale));

    if (confidence < 0.01) {
      return null;
    }

    return {
      pose:       new Float32Array(poseData.buffer,  poseData.byteOffset,  72),
      betas:      new Float32Array(betasData.buffer, betasData.byteOffset, 10),
      cam:        new Float32Array(camData.buffer,   camData.byteOffset,   3),
      confidence,
    };
  }
}

// ---------------------------------------------------------------------------
// CPU bilinear resize (same implementation as pose-2d.node.ts)
// ---------------------------------------------------------------------------

function _resizeCpu(src: ImageData, targetSize: number): ImageData {
  const sw = src.width;
  const sh = src.height;
  const tw = targetSize;
  const th = targetSize;
  const dst = new Uint8ClampedArray(tw * th * 4);

  for (let ty = 0; ty < th; ty++) {
    for (let tx = 0; tx < tw; tx++) {
      const srcX = (tx / tw) * sw;
      const srcY = (ty / th) * sh;
      const x0 = Math.floor(srcX);
      const y0 = Math.floor(srcY);
      const x1 = Math.min(x0 + 1, sw - 1);
      const y1 = Math.min(y0 + 1, sh - 1);
      const fx = srcX - x0;
      const fy = srcY - y0;

      for (let c = 0; c < 4; c++) {
        const v00 = src.data[(y0 * sw + x0) * 4 + c]!;
        const v10 = src.data[(y0 * sw + x1) * 4 + c]!;
        const v01 = src.data[(y1 * sw + x0) * 4 + c]!;
        const v11 = src.data[(y1 * sw + x1) * 4 + c]!;
        dst[(ty * tw + tx) * 4 + c] = Math.round(
          v00 * (1 - fx) * (1 - fy) +
          v10 * fx       * (1 - fy) +
          v01 * (1 - fx) * fy       +
          v11 * fx       * fy,
        );
      }
    }
  }

  return new ImageData(dst, tw, th);
}
