/**
 * Depth map colourisation and canvas rendering for the realtime Node A panel.
 *
 * `depthColormap` is a pure function and can be unit-tested without a DOM.
 * `DepthRenderer` wraps an offscreen canvas that is allocated once and reused
 * every frame to avoid per-frame allocations.
 */

import { PROC_W, PROC_H } from '../../src/core/dimensions.ts';

// ---------------------------------------------------------------------------
// Pure colormap
// ---------------------------------------------------------------------------

/**
 * Map a normalised depth value [0, 1] to an RGB triple using a
 * viridis-inspired blue → green → yellow palette.
 *
 * @param norm  Depth in [0, 1] (0 = near/close, 1 = far)
 * @returns     [r, g, b] each in [0, 255]
 */
export function depthColormap(norm: number): [number, number, number] {
  const clamp = (v: number) => Math.min(255, Math.max(0, Math.round(v)));
  const r = clamp((norm * 3 - 1) * 255);
  const g = clamp(Math.sin(norm * Math.PI) * 255);
  const b = clamp((1 - norm * 2) * 255);
  return [r, g, b];
}

// ---------------------------------------------------------------------------
// Renderer class
// ---------------------------------------------------------------------------

/**
 * Renders a Float32Array depth map to a 2D canvas context.
 * Allocates one offscreen canvas at construction and reuses it.
 */
export class DepthRenderer {
  private readonly _offscreen: HTMLCanvasElement;
  private readonly _offCtx: CanvasRenderingContext2D;
  private readonly _frameW: number;
  private readonly _frameH: number;

  constructor(frameW = PROC_W as number, frameH = PROC_H as number) {
    this._frameW = frameW;
    this._frameH = frameH;
    this._offscreen = document.createElement('canvas');
    this._offscreen.width = frameW;
    this._offscreen.height = frameH;
    this._offCtx = this._offscreen.getContext('2d')!;
  }

  /**
   * Render `depthData` (FRAME_W × FRAME_H floats, arbitrary range) to
   * `ctx2d` at the given destination size, letterboxing if aspect ratios differ.
   */
  render(
    depthData: Float32Array,
    ctx2d: CanvasRenderingContext2D,
    dstW: number,
    dstH: number,
  ): void {
    // Normalise depth range
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < depthData.length; i++) {
      const v = depthData[i]!;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = max - min || 1;

    const imgData = this._offCtx.createImageData(this._frameW, this._frameH);
    const px = imgData.data;
    for (let i = 0; i < depthData.length; i++) {
      const [r, g, b] = depthColormap((depthData[i]! - min) / range);
      px[i * 4]     = r;
      px[i * 4 + 1] = g;
      px[i * 4 + 2] = b;
      px[i * 4 + 3] = 255;
    }
    this._offCtx.putImageData(imgData, 0, 0);

    // Letterbox scale to destination
    const srcAR = this._frameW / this._frameH;
    const dstAR = dstW / dstH;
    let sx = 0, sy = 0, sw = this._frameW, sh = this._frameH;
    if (Math.abs(srcAR - dstAR) > 0.01) {
      if (srcAR > dstAR) { sw = Math.round(this._frameH * dstAR); sx = Math.round((this._frameW - sw) / 2); }
      else               { sh = Math.round(this._frameW / dstAR); sy = Math.round((this._frameH - sh) / 2); }
    }
    ctx2d.drawImage(this._offscreen, sx, sy, sw, sh, 0, 0, dstW, dstH);
  }
}
