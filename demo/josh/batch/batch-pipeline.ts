/**
 * Phase 2B: Batch Pipeline Orchestrator
 *
 * Processes a full video through the JOSH pipeline in 100-frame chunks.
 * Each chunk goes through phases A–I sequentially.  All model-dependent
 * phases (MASt3R, ROMP, JOSH optimiser) are stubs — they return null /
 * zeroes so the pipeline is fully runnable without ONNX models loaded.
 *
 * Phases per chunk
 * ─────────────────
 *  A  Extract frames at 5 FPS           (FrameExtractor)
 *  B  Foreground segmentation           (SegmentationNode — BodyPix / HSV fallback)
 *  C  MASt3R dense matching             (STUB — returns null pointmaps)
 *  D  Focal-length recovery             (one-time, first chunk only)
 *  E  ROMP SMPL initialisation          (STUB — returns null)
 *  F  MoveNet 2D pose                   (Pose2DNode)
 *  G  Contact detection                 (detectContacts)
 *  H  JOSH optimisation (700 iters)     (STUB — returns zero params)
 *  I  Keyframe interpolation            (for non-keyframe frames)
 */

import { JOSH_CONFIG } from '../config.ts';
import { FrameExtractor, type ExtractedFrame } from './frame-extractor.ts';
import { JoshResultCache, type PerFrameResult } from './result-cache.ts';
import { SegmentationNode, type SegmentationResult } from '../nodes/segmentation.node.ts';
import { Pose2DNode, type Keypoint2D } from '../nodes/pose-2d.node.ts';
import { detectContacts, type ContactResult } from '../utils/contact-detection.ts';

// ---------------------------------------------------------------------------
// Re-export the cache result type so callers can import it from one place
// ---------------------------------------------------------------------------

export type { PerFrameResult as JoshFrameResult } from './result-cache.ts';

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface BatchProgress {
  phase:
    | 'extract'
    | 'segment'
    | 'mast3r'
    | 'focal'
    | 'romp'
    | 'pose2d'
    | 'contact'
    | 'optimize'
    | 'interpolate';
  frameIndex: number;
  totalFrames: number;
  chunkIndex: number;
  totalChunks: number;
  /** Populated during the 'optimize' phase */
  iterIndex?: number;
  /** Populated during the 'optimize' phase */
  loss?: number;
  etaMs?: number;
}

export interface BatchResult {
  frameCount: number;
  cachedFrames: number;
  processingTimeMs: number;
}

// ---------------------------------------------------------------------------
// Internal per-frame intermediate types
// ---------------------------------------------------------------------------

interface FrameContext {
  frame: ExtractedFrame;
  segmentation: SegmentationResult | null;
  /** null when MASt3R stub is active */
  pointmap: Float32Array | null;
  /** null when ROMP stub is active */
  smplInit: Float32Array | null;
  pose2d: Keypoint2D[] | null;
  contact: ContactResult | null;
  /** 89-dim JOSH params: pose(72)+shape(10)+trans(3)+orient(3)+scale(1) */
  joshParams: Float32Array | null;
}

// ---------------------------------------------------------------------------
// Utility — simple video hash from URL (for cache keying)
// ---------------------------------------------------------------------------

async function hashVideoUrl(url: string): Promise<string> {
  try {
    const encoder = new TextEncoder();
    const data = encoder.encode(url);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray
      .slice(0, 8)
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('');
  } catch {
    // Fallback: simple string hash
    let h = 0;
    for (let i = 0; i < url.length; i++) {
      h = (Math.imul(31, h) + url.charCodeAt(i)) | 0;
    }
    return Math.abs(h).toString(16).padStart(8, '0');
  }
}

// ---------------------------------------------------------------------------
// STUB helpers — clearly marked for future replacement
// ---------------------------------------------------------------------------

/**
 * STUB: replace with real model
 * MASt3R dense point-map inference.
 * Returns null until a real model is wired in.
 */
function stubMast3r(_frame: ExtractedFrame): Float32Array | null {
  // STUB: replace with real model — run MASt3R ONNX session and return
  // a Float32Array of shape [H * W * 3] (XYZ per pixel).
  return null;
}

/**
 * STUB: replace with real model
 * Focal-length recovery from point-maps.
 * Returns the default values from config until a real implementation exists.
 */
function stubFocalRecovery(_pointmaps: (Float32Array | null)[]): {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
} {
  // STUB: replace with real model — solve for camera intrinsics from
  // the first chunk's MASt3R point-maps using DLT or RANSAC.
  return {
    fx: JOSH_CONFIG.defaultFx,
    fy: JOSH_CONFIG.defaultFy,
    cx: JOSH_CONFIG.defaultCx,
    cy: JOSH_CONFIG.defaultCy,
  };
}

/**
 * STUB: replace with real model
 * ROMP multi-person SMPL parameter initialisation.
 * Returns null until a real ONNX session is available.
 */
function stubRomp(_frame: ExtractedFrame): Float32Array | null {
  // STUB: replace with real model — run ROMP ONNX session (512×512 input)
  // and return initial SMPL pose/shape parameters as Float32Array[89].
  return null;
}

/**
 * STUB: replace with real model
 * JOSH gradient-based optimiser (700 iterations: 500 stage-1 + 200 stage-2).
 * Returns zero-filled parameter vector until the WASM solver is wired in.
 */
async function stubJoshOptimize(
  _frameCtx: FrameContext,
  _camera: { fx: number; fy: number; cx: number; cy: number },
  totalIters: number,
  onIter: (iterIndex: number, loss: number) => void,
  signal: AbortSignal | undefined,
): Promise<Float32Array> {
  // STUB: replace with real model — call the JOSH WASM solver with
  // depth map, 2D keypoints, contact flags, and SMPL initialisation.
  // For now we simulate iteration progress with a decaying fake loss.
  for (let i = 0; i < totalIters; i++) {
    if (signal?.aborted) break;
    const fakeLoss = 10.0 * Math.exp(-i / (totalIters * 0.4));
    onIter(i, fakeLoss);
    // Yield to the event loop every 50 iterations to keep UI responsive.
    if (i % 50 === 0) {
      await new Promise<void>((r) => setTimeout(r, 0));
    }
  }
  return new Float32Array(JOSH_CONFIG.paramDim); // zeros
}

/**
 * Linear interpolation of JOSH params between two keyframe results.
 * Used for non-keyframe slots produced by chunking.
 */
function interpolateParams(a: Float32Array, b: Float32Array, t: number): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) {
    out[i] = a[i]! * (1 - t) + b[i]! * t;
  }
  return out;
}

// ---------------------------------------------------------------------------
// BatchPipeline
// ---------------------------------------------------------------------------

export class BatchPipeline {
  private readonly _device: GPUDevice;
  private readonly _onProgress: ((p: BatchProgress) => void) | undefined;

  private readonly _extractor = new FrameExtractor();
  private readonly _segmentation = new SegmentationNode();
  private readonly _pose2d = new Pose2DNode();

  private _segInitialized = false;
  private _poseInitialized = false;

  constructor(device: GPUDevice, onProgress?: (p: BatchProgress) => void) {
    this._device = device;
    this._onProgress = onProgress;
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Process a video URL end-to-end through the JOSH batch pipeline.
   * Results are written to IndexedDB via JoshResultCache.
   *
   * @param videoUrl  HTTP or blob URL pointing to the source video.
   * @param signal    Optional AbortSignal for cancellation.
   */
  async process(videoUrl: string, signal?: AbortSignal): Promise<BatchResult> {
    const t0 = performance.now();

    // --- Open result cache ---------------------------------------------------
    const videoHash = await hashVideoUrl(videoUrl);
    const cache = new JoshResultCache(videoHash);
    await cache.open();

    // --- Phase A: extract all frames at targetFps ----------------------------
    let allFrames: ExtractedFrame[];
    try {
      allFrames = await this._extractor.extractFromUrl(videoUrl, {
        targetFps: JOSH_CONFIG.targetFps,
        onProgress: (extracted, total) => {
          this._emit({
            phase: 'extract',
            frameIndex: extracted,
            totalFrames: total,
            chunkIndex: 0,
            totalChunks: 1,
          });
        },
      });
    } catch (err) {
      await cache.close();
      throw err;
    }

    if (signal?.aborted) {
      await cache.close();
      throw new DOMException('Aborted', 'AbortError');
    }

    const totalFrames = allFrames.length;
    const chunkSize = JOSH_CONFIG.chunkSize;
    const totalChunks = Math.ceil(totalFrames / chunkSize);

    // --- Initialise shared nodes --------------------------------------------
    await this._ensureSegmentationInit(allFrames[0]?.width ?? 384, allFrames[0]?.height ?? 384);
    await this._ensurePoseInit();

    // --- Camera intrinsics (recovered once from first chunk) ----------------
    let camera = {
      fx: JOSH_CONFIG.defaultFx,
      fy: JOSH_CONFIG.defaultFy,
      cx: JOSH_CONFIG.defaultCx,
      cy: JOSH_CONFIG.defaultCy,
    };
    let focalRecovered = false;

    let cachedFrames = 0;

    // --- Chunk loop ----------------------------------------------------------
    for (let chunkIdx = 0; chunkIdx < totalChunks; chunkIdx++) {
      if (signal?.aborted) break;

      const chunkStart = chunkIdx * chunkSize;
      const chunkEnd = Math.min(chunkStart + chunkSize, totalFrames);
      const chunkFrames = allFrames.slice(chunkStart, chunkEnd);

      const frameContexts: FrameContext[] = chunkFrames.map((frame) => ({
        frame,
        segmentation: null,
        pointmap: null,
        smplInit: null,
        pose2d: null,
        contact: null,
        joshParams: null,
      }));

      // ---- Phase B: Segmentation ------------------------------------------
      for (let i = 0; i < frameContexts.length; i++) {
        if (signal?.aborted) break;
        const ctx = frameContexts[i]!;
        const globalIdx = chunkStart + i;

        // Skip if already cached
        if (await cache.hasFrame(globalIdx)) {
          cachedFrames++;
          this._emit({
            phase: 'segment',
            frameIndex: globalIdx,
            totalFrames,
            chunkIndex: chunkIdx,
            totalChunks,
          });
          continue;
        }

        ctx.segmentation = await this._segmentation.segment(ctx.frame.imageData);
        this._emit({
          phase: 'segment',
          frameIndex: globalIdx,
          totalFrames,
          chunkIndex: chunkIdx,
          totalChunks,
        });
      }

      if (signal?.aborted) break;

      // ---- Phase C: MASt3R (stub) -----------------------------------------
      for (let i = 0; i < frameContexts.length; i++) {
        if (signal?.aborted) break;
        const ctx = frameContexts[i]!;
        const globalIdx = chunkStart + i;

        if (ctx.segmentation === null) continue; // already cached

        ctx.pointmap = stubMast3r(ctx.frame); // STUB: replace with real model
        this._emit({
          phase: 'mast3r',
          frameIndex: globalIdx,
          totalFrames,
          chunkIndex: chunkIdx,
          totalChunks,
        });
      }

      if (signal?.aborted) break;

      // ---- Phase D: Focal-length recovery (first chunk only) ---------------
      if (!focalRecovered) {
        const pointmaps = frameContexts.map((c) => c.pointmap);
        camera = stubFocalRecovery(pointmaps); // STUB: replace with real model
        focalRecovered = true;
        // Emit once for the first frame of this chunk
        this._emit({
          phase: 'focal',
          frameIndex: chunkStart,
          totalFrames,
          chunkIndex: chunkIdx,
          totalChunks,
        });
      }

      // ---- Phase E: ROMP (stub) -------------------------------------------
      for (let i = 0; i < frameContexts.length; i++) {
        if (signal?.aborted) break;
        const ctx = frameContexts[i]!;
        const globalIdx = chunkStart + i;

        if (ctx.segmentation === null) continue; // already cached

        ctx.smplInit = stubRomp(ctx.frame); // STUB: replace with real model
        this._emit({
          phase: 'romp',
          frameIndex: globalIdx,
          totalFrames,
          chunkIndex: chunkIdx,
          totalChunks,
        });
      }

      if (signal?.aborted) break;

      // ---- Phase F: MoveNet 2D pose ----------------------------------------
      for (let i = 0; i < frameContexts.length; i++) {
        if (signal?.aborted) break;
        const ctx = frameContexts[i]!;
        const globalIdx = chunkStart + i;

        if (ctx.segmentation === null) continue; // already cached

        const preprocessed = this._pose2d.preprocessFrame(ctx.frame.imageData);
        ctx.pose2d = await this._pose2d.detect(preprocessed);
        this._emit({
          phase: 'pose2d',
          frameIndex: globalIdx,
          totalFrames,
          chunkIndex: chunkIdx,
          totalChunks,
        });
      }

      if (signal?.aborted) break;

      // ---- Phase G: Contact detection -------------------------------------
      for (let i = 0; i < frameContexts.length; i++) {
        if (signal?.aborted) break;
        const ctx = frameContexts[i]!;
        const globalIdx = chunkStart + i;

        if (ctx.segmentation === null || ctx.pose2d === null) continue;

        // We don't yet have real SMPL vertices (that requires phase H output),
        // so we run contact detection with a placeholder zero-vertex array.
        // The real pipeline would use posed vertices from the SMPL forward pass.
        const placeholderVertices = new Float32Array(JOSH_CONFIG.smplVertexCount * 3);
        // Foot-sole vertex indices (approximate from SMPL topology).
        const footSoleIndices = [3216, 3226, 3387, 6617, 6624, 6787];
        const depthWidth = JOSH_CONFIG.depthMapSize;
        const depthHeight = JOSH_CONFIG.depthMapSize;
        const dummyDepthMap = new Float32Array(depthWidth * depthHeight); // zeros

        ctx.contact = detectContacts(
          placeholderVertices,
          dummyDepthMap,
          depthWidth,
          depthHeight,
          footSoleIndices,
          {
            contactThresholdMeters: JOSH_CONFIG.contactThreshold,
            cameraFx: camera.fx,
            cameraFy: camera.fy,
            cameraCx: camera.cx,
            cameraCy: camera.cy,
          },
        );

        this._emit({
          phase: 'contact',
          frameIndex: globalIdx,
          totalFrames,
          chunkIndex: chunkIdx,
          totalChunks,
        });
      }

      if (signal?.aborted) break;

      // ---- Phase H: JOSH optimisation (stub) ------------------------------
      const totalIters = JOSH_CONFIG.stage1.iters + JOSH_CONFIG.stage2.iters; // 700

      for (let i = 0; i < frameContexts.length; i++) {
        if (signal?.aborted) break;
        const ctx = frameContexts[i]!;
        const globalIdx = chunkStart + i;

        if (ctx.segmentation === null) continue; // already cached

        const phaseStartTime = performance.now();

        ctx.joshParams = await stubJoshOptimize( // STUB: replace with real model
          ctx,
          camera,
          totalIters,
          (iterIndex, loss) => {
            const elapsed = performance.now() - phaseStartTime;
            const itersRemaining = totalIters - iterIndex;
            const msPerIter = elapsed / Math.max(iterIndex, 1);
            const etaMs = itersRemaining * msPerIter;
            this._emit({
              phase: 'optimize',
              frameIndex: globalIdx,
              totalFrames,
              chunkIndex: chunkIdx,
              totalChunks,
              iterIndex,
              loss,
              etaMs,
            });
          },
          signal,
        );

        // Persist result to IndexedDB -----------------------------------------
        const result: PerFrameResult = {
          frameIndex: globalIdx,
          timestamp: ctx.frame.timestamp,
          smplPose: ctx.joshParams.slice(0, 72),
          smplShape: ctx.joshParams.slice(72, 82),
          cameraPose: new Float32Array(16), // identity until real solver
          depthScale: 1.0,
          vertices: null,
          jointPositions: new Float32Array(JOSH_CONFIG.smplJointCount * 3),
          losses: new Float32Array(6),
        };
        await cache.storeFrame(result);
      }

      if (signal?.aborted) break;

      // ---- Phase I: Keyframe interpolation --------------------------------
      // At 5 FPS every extracted frame is a keyframe, but if the chunk
      // contains any gaps (e.g. from cache skips), linearly interpolate.
      // We walk the chunk looking for consecutive cached + newly-computed
      // frames and fill any nulls via interpolation.

      const keyframePairs: { aIdx: number; bIdx: number }[] = [];
      for (let i = 0; i < frameContexts.length - 1; i++) {
        const aCtx = frameContexts[i]!;
        const bCtx = frameContexts[i + 1]!;
        if (aCtx.joshParams !== null && bCtx.joshParams !== null) {
          keyframePairs.push({ aIdx: i, bIdx: i + 1 });
        }
      }

      for (const { aIdx, bIdx } of keyframePairs) {
        if (signal?.aborted) break;
        const aParams = frameContexts[aIdx]!.joshParams!;
        const bParams = frameContexts[bIdx]!.joshParams!;
        const globalIdx = chunkStart + aIdx;

        // At 5 FPS there are no sub-frames between adjacent keyframes,
        // so this loop body only runs if bIdx > aIdx + 1 (future use).
        for (let sub = aIdx + 1; sub < bIdx; sub++) {
          const t = (sub - aIdx) / (bIdx - aIdx);
          const interpolated = interpolateParams(aParams, bParams, t);
          frameContexts[sub]!.joshParams = interpolated;

          const subGlobalIdx = chunkStart + sub;
          const subFrame = frameContexts[sub]!.frame;
          const interpResult: PerFrameResult = {
            frameIndex: subGlobalIdx,
            timestamp: subFrame.timestamp,
            smplPose: interpolated.slice(0, 72),
            smplShape: interpolated.slice(72, 82),
            cameraPose: new Float32Array(16),
            depthScale: 1.0,
            vertices: null,
            jointPositions: new Float32Array(JOSH_CONFIG.smplJointCount * 3),
            losses: new Float32Array(6),
          };
          await cache.storeFrame(interpResult);

          this._emit({
            phase: 'interpolate',
            frameIndex: subGlobalIdx,
            totalFrames,
            chunkIndex: chunkIdx,
            totalChunks,
          });
        }

        this._emit({
          phase: 'interpolate',
          frameIndex: globalIdx,
          totalFrames,
          chunkIndex: chunkIdx,
          totalChunks,
        });
      }
    } // end chunk loop

    // --- Store video metadata -----------------------------------------------
    await cache.storeMetadata({
      videoHash,
      frameCount: totalFrames,
      fps: JOSH_CONFIG.targetFps,
      processedAt: Date.now(),
    });

    const processingTimeMs = performance.now() - t0;
    const finalCachedCount = await cache.frameCount();

    await cache.close();

    return {
      frameCount: totalFrames,
      cachedFrames: finalCachedCount,
      processingTimeMs,
    };
  }

  /**
   * Retrieve a single frame result by index.
   * Opens and closes the cache on each call — prefer batched access for
   * performance-sensitive consumers.
   */
  async getFrameResult(frameIndex: number): Promise<PerFrameResult | null> {
    // We need the video hash to open the right DB, but at this API layer
    // we don't have it.  Callers that need low-latency access should hold
    // a reference to the JoshResultCache directly.  This method is a
    // convenience wrapper that searches across known cache databases by
    // trying to load from the most recently created one.
    //
    // For now, surface a clear error so callers know to use the cache directly.
    void frameIndex;
    console.warn(
      '[BatchPipeline] getFrameResult() requires a video hash. ' +
        'Use JoshResultCache directly for production use.',
    );
    return null;
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  private _emit(p: BatchProgress): void {
    this._onProgress?.(p);
  }

  private async _ensureSegmentationInit(width: number, height: number): Promise<void> {
    if (!this._segInitialized) {
      await this._segmentation.initialize(width, height);
      this._segInitialized = true;
    }
  }

  private async _ensurePoseInit(): Promise<void> {
    if (!this._poseInitialized) {
      await this._pose2d.initialize();
      this._poseInitialized = true;
    }
  }

  /**
   * Release GPU/ONNX resources held by internal nodes.
   * Call after all processing is complete.
   */
  dispose(): void {
    this._segmentation.dispose();
    this._pose2d.dispose();
  }
}
