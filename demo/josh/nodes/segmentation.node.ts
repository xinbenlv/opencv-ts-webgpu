/**
 * Phase 1A: Human foreground segmentation.
 *
 * Strategy pattern:
 *  - Primary:  @mediapipe/selfie_segmentation (dynamically imported if available)
 *  - Fallback: HSV skin-colour detection heuristic
 */

export interface SegmentationResult {
  /** Binary mask: 0=background, 1=foreground (person), shape [H, W] */
  mask: Uint8Array;
  width: number;
  height: number;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Convert a single sRGB pixel to HSV.
 * R, G, B are in [0, 1].
 * Returns [H in degrees (0–360), S in [0,1], V in [0,1]].
 */
function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const delta = max - min;

  const v = max;
  const s = max === 0 ? 0 : delta / max;

  let h = 0;
  if (delta !== 0) {
    if (max === r) {
      h = ((g - b) / delta) % 6;
    } else if (max === g) {
      h = (b - r) / delta + 2;
    } else {
      h = (r - g) / delta + 4;
    }
    h = h * 60;
    if (h < 0) h += 360;
  }

  return [h, s, v];
}

/**
 * Per-pixel skin-colour heuristic based on HSV thresholds.
 * Conservative thresholds — reduces false-positives on warm backgrounds.
 */
function skinMask(imageData: ImageData): Uint8Array {
  const { data, width, height } = imageData;
  const mask = new Uint8Array(width * height);

  for (let i = 0; i < width * height; i++) {
    const base = i * 4;
    const r = (data[base]! / 255) as number;
    const g = (data[base + 1]! / 255) as number;
    const b = (data[base + 2]! / 255) as number;

    const [h, s, v] = rgbToHsv(r, g, b);
    // Skin tone: hue 0°–35°, reasonable saturation and brightness
    const isSkin = h >= 0 && h <= 35 && s > 0.2 && v > 0.3;
    mask[i] = isSkin ? 1 : 0;
  }

  return mask;
}

// ---------------------------------------------------------------------------
// SegmentationNode
// ---------------------------------------------------------------------------

export class SegmentationNode {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private _mediapipe: any = null;
  private _usingMediaPipe = false;

  /**
   * Initialize segmentation — tries MediaPipe first, falls back to heuristic.
   */
  async initialize(width: number, height: number): Promise<void> {
    try {
      // Attempt to load MediaPipe Selfie Segmentation dynamically.
      // The package may not be installed in all environments.
      const mp = await import('@mediapipe/selfie_segmentation' as string);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const SelfieSegmentation = (mp as any).SelfieSegmentation;
      if (typeof SelfieSegmentation !== 'function') throw new Error('SelfieSegmentation not found');

      this._mediapipe = new SelfieSegmentation({
        locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`,
      });
      this._mediapipe.setOptions({ modelSelection: 1 /* landscape */ });

      await new Promise<void>((resolve, reject) => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        this._mediapipe.onResults((_results: any) => resolve());
        // Send a tiny dummy frame to confirm the model loaded.
        const dummy = new ImageData(width, height);
        this._mediapipe.send({ image: dummy }).catch(reject);
      });

      this._usingMediaPipe = true;
      console.log('[SegmentationNode] Using MediaPipe Selfie Segmentation');
    } catch {
      this._usingMediaPipe = false;
      console.info('[SegmentationNode] MediaPipe unavailable — using HSV skin-colour heuristic');
    }
  }

  /** Segment a frame — returns binary mask. */
  async segment(imageData: ImageData): Promise<SegmentationResult> {
    const { width, height } = imageData;

    if (this._usingMediaPipe && this._mediapipe !== null) {
      return this._segmentMediaPipe(imageData);
    }

    return {
      mask: skinMask(imageData),
      width,
      height,
    };
  }

  /** Apply mask to image: set masked-out pixels to black. Returns a new ImageData. */
  applyMask(imageData: ImageData, mask: Uint8Array): ImageData {
    const { width, height } = imageData;
    const outData = new Uint8ClampedArray(imageData.data);
    // Construct ImageData — use the constructor when available (browser), otherwise
    // fall back to a plain structural object that satisfies the interface (Node.js tests).
    const out: ImageData =
      typeof ImageData !== 'undefined'
        ? new ImageData(outData, width, height)
        : ({ data: outData, width, height } as unknown as ImageData);

    for (let i = 0; i < width * height; i++) {
      if (mask[i] === 0) {
        out.data[i * 4] = 0;
        out.data[i * 4 + 1] = 0;
        out.data[i * 4 + 2] = 0;
        out.data[i * 4 + 3] = 255;
      }
    }

    return out;
  }

  dispose(): void {
    if (this._mediapipe !== null) {
      try {
        // eslint-disable-next-line @typescript-eslint/no-unsafe-call
        this._mediapipe.close();
      } catch {
        /* best-effort */
      }
      this._mediapipe = null;
    }
    this._usingMediaPipe = false;
  }

  // ---------------------------------------------------------------------------
  // Private — MediaPipe path
  // ---------------------------------------------------------------------------

  private async _segmentMediaPipe(imageData: ImageData): Promise<SegmentationResult> {
    const { width, height } = imageData;

    return new Promise<SegmentationResult>((resolve, reject) => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      this._mediapipe.onResults((results: any) => {
        try {
          // results.segmentationMask is a HTMLCanvasElement or ImageBitmap
          const canvas = document.createElement('canvas');
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d')!;
          ctx.drawImage(results.segmentationMask, 0, 0, width, height);
          const raw = ctx.getImageData(0, 0, width, height);

          const mask = new Uint8Array(width * height);
          for (let i = 0; i < width * height; i++) {
            // MediaPipe mask encodes foreground probability in the red channel
            mask[i] = raw.data[i * 4]! > 127 ? 1 : 0;
          }
          resolve({ mask, width, height });
        } catch (err) {
          reject(err);
        }
      });

      this._mediapipe.send({ image: imageData }).catch(reject);
    });
  }
}
