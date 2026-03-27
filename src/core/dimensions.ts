/**
 * Branded numeric types for dimension / coordinate conversions in the pipeline.
 *
 * Using distinct brands prevents mixing up coordinate spaces at compile time —
 * e.g. accidentally passing a normalized [0,1] value where a pixel offset is
 * expected, or confusing the internal 384×384 processing resolution with
 * display canvas dimensions or world-space metric units.
 *
 * Usage
 * -----
 *   import { type NormalizedCoord, type ProcessingPx, PROC_W, normalizedToProcessing } from '../../src/core/dimensions.ts';
 *
 *   const nx = 0.5 as NormalizedCoord;
 *   const px = normalizedToProcessing(nx, PROC_W); // ProcessingPx — correct
 *   const bad: ProcessingPx = nx;                  // TS error — caught at compile time
 */

declare const __brand: unique symbol;
type Brand<T, B> = T & { readonly [__brand]: B };

// ── Branded coordinate / dimension types ─────────────────────────────────────

/** Integer pixel coordinate in the internal 384 × 384 GPU processing space */
export type ProcessingPx = Brand<number, 'ProcessingPx'>;

/** Integer pixel coordinate in the display / output canvas space (video AR) */
export type DisplayPx = Brand<number, 'DisplayPx'>;

/** Normalised spatial coordinate in [0, 1] relative to image dimensions */
export type NormalizedCoord = Brand<number, 'NormalizedCoord'>;

/** Absolute world-space coordinate, SI metres, Y-up right-handed */
export type MetricMeters = Brand<number, 'MetricMeters'>;

/** Relative depth value as produced by MiDaS (unitless, inverse proportional) */
export type RelativeDepth = Brand<number, 'RelativeDepth'>;

// ── Processing-resolution constants ──────────────────────────────────────────

/** Internal processing width (pixels) — matches GPU buffer layout */
export const PROC_W = 384 as ProcessingPx;

/** Internal processing height (pixels) — matches GPU buffer layout */
export const PROC_H = 384 as ProcessingPx;

// ── Conversion helpers ────────────────────────────────────────────────────────

/** NormalizedCoord [0,1]  →  ProcessingPx [0, dim) */
export function normalizedToProcessing(n: NormalizedCoord, dim: ProcessingPx): ProcessingPx {
  return Math.round(n * (dim as number)) as ProcessingPx;
}

/** ProcessingPx  →  NormalizedCoord [0, 1] */
export function processingToNormalized(p: ProcessingPx, dim: ProcessingPx): NormalizedCoord {
  return ((p as number) / (dim as number)) as NormalizedCoord;
}

/** NormalizedCoord [0,1]  →  DisplayPx [0, dim) */
export function normalizedToDisplay(n: NormalizedCoord, dim: DisplayPx): DisplayPx {
  return Math.round(n * (dim as number)) as DisplayPx;
}

/** DisplayPx  →  NormalizedCoord [0, 1] */
export function displayToNormalized(p: DisplayPx, dim: DisplayPx): NormalizedCoord {
  return ((p as number) / (dim as number)) as NormalizedCoord;
}

/** ProcessingPx  →  DisplayPx (scales processing-space pixel to display space) */
export function processingToDisplay(
  p: ProcessingPx,
  procDim: ProcessingPx,
  displayDim: DisplayPx,
): DisplayPx {
  return Math.round(((p as number) / (procDim as number)) * (displayDim as number)) as DisplayPx;
}

/**
 * Compute a display canvas size that preserves the source video's aspect ratio
 * while keeping one axis pinned to `anchorPx`.
 *
 * @param srcW      Source video / image width  (any numeric unit)
 * @param srcH      Source video / image height (same unit)
 * @param anchorPx  Pin the shorter axis to this display pixel count
 * @returns         { w: DisplayPx, h: DisplayPx }
 */
export function fitToAnchor(srcW: number, srcH: number, anchorPx: number = 384): { w: DisplayPx; h: DisplayPx } {
  if (srcW <= 0 || srcH <= 0) return { w: anchorPx as DisplayPx, h: anchorPx as DisplayPx };
  const ar = srcW / srcH;
  if (ar >= 1) {
    // Wider than tall → anchor height, scale width
    const h = anchorPx as DisplayPx;
    const w = Math.round(anchorPx * ar) as DisplayPx;
    return { w, h };
  } else {
    // Taller than wide → anchor width, scale height
    const w = anchorPx as DisplayPx;
    const h = Math.round(anchorPx / ar) as DisplayPx;
    return { w, h };
  }
}
