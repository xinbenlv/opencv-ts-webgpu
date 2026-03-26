/**
 * Phase 2A: Extract frames from a video file at a target FPS.
 *
 * Primary strategy: HTMLVideoElement + canvas.drawImage() — works everywhere.
 * This is simpler and more reliable than WebCodecs for the frame-extraction use-case.
 */

import { JOSH_CONFIG } from '../config.ts';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ExtractedFrame {
  frameIndex: number;
  timestamp: number; // seconds
  imageData: ImageData;
  width: number;
  height: number;
}

export interface FrameExtractionOptions {
  /** Target frames-per-second to sample.  Default: JOSH_CONFIG.targetFps (5). */
  targetFps?: number;
  /** Optional hard cap on the total number of frames extracted. */
  maxFrames?: number;
  /** Progress callback — called after each frame is extracted. */
  onProgress?: (extracted: number, total: number) => void;
}

export interface VideoInfo {
  duration: number; // seconds
  width: number;
  height: number;
  fps: number; // native fps (may be approximate)
}

// ---------------------------------------------------------------------------
// FrameExtractor
// ---------------------------------------------------------------------------

export class FrameExtractor {
  /**
   * Extract frames from a File object (e.g. from a drag-and-drop input).
   */
  async extractFromFile(file: File, options?: FrameExtractionOptions): Promise<ExtractedFrame[]> {
    const url = URL.createObjectURL(file);
    try {
      return await this._extractFromObjectUrl(url, options);
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  /**
   * Extract frames from a URL (HTTP or blob URL).
   */
  async extractFromUrl(url: string, options?: FrameExtractionOptions): Promise<ExtractedFrame[]> {
    return this._extractFromObjectUrl(url, options);
  }

  /**
   * Get video metadata without extracting frames.
   */
  async getVideoInfo(file: File): Promise<VideoInfo> {
    const url = URL.createObjectURL(file);
    try {
      const video = this._createVideoElement();
      video.src = url;
      await this._waitForEvent(video, 'loadedmetadata');

      // Attempt to infer native fps from duration + a heuristic.
      // True FPS is not exposed by the HTMLVideoElement API; we report a
      // sensible default and let the caller decide the sampling rate.
      const fps = await this._estimateFps(video);

      return {
        duration: video.duration,
        width: video.videoWidth,
        height: video.videoHeight,
        fps,
      };
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  private async _extractFromObjectUrl(
    url: string,
    options?: FrameExtractionOptions,
  ): Promise<ExtractedFrame[]> {
    const targetFps = options?.targetFps ?? JOSH_CONFIG.targetFps;
    const maxFrames = options?.maxFrames ?? Number.POSITIVE_INFINITY;
    const onProgress = options?.onProgress;

    const video = this._createVideoElement();
    video.src = url;

    // Wait for metadata so we know duration + dimensions.
    await this._waitForEvent(video, 'loadedmetadata');

    const { duration, videoWidth: width, videoHeight: height } = video;
    const frameInterval = 1 / targetFps;
    const estimatedTotal = Math.min(Math.floor(duration / frameInterval) + 1, maxFrames);

    // Set up an off-screen canvas for pixel readback.
    const canvas = this._createCanvas(width, height);
    const ctx = canvas.getContext('2d');
    if (ctx === null) {
      throw new Error('[FrameExtractor] Could not obtain 2D canvas context');
    }

    const frames: ExtractedFrame[] = [];
    let frameIndex = 0;

    for (let t = 0; t < duration && frameIndex < maxFrames; t += frameInterval) {
      video.currentTime = t;
      await this._waitForEvent(video, 'seeked');

      ctx.drawImage(video, 0, 0, width, height);
      const imageData = ctx.getImageData(0, 0, width, height);

      frames.push({
        frameIndex,
        timestamp: t,
        imageData,
        width,
        height,
      });

      frameIndex++;
      onProgress?.(frameIndex, estimatedTotal);
    }

    return frames;
  }

  /**
   * Create an HTMLVideoElement configured for seek-based extraction.
   * muted + playsInline are required to work on mobile browsers.
   */
  private _createVideoElement(): HTMLVideoElement {
    const video = document.createElement('video');
    video.muted = true;
    video.playsInline = true;
    video.preload = 'auto';
    return video;
  }

  private _createCanvas(width: number, height: number): HTMLCanvasElement {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    return canvas;
  }

  /**
   * Promisify a single HTMLVideoElement event.
   */
  private _waitForEvent(video: HTMLVideoElement, event: string): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const onEvent = () => {
        video.removeEventListener(event, onEvent);
        video.removeEventListener('error', onError);
        resolve();
      };
      const onError = () => {
        video.removeEventListener(event, onEvent);
        video.removeEventListener('error', onError);
        reject(new Error(`[FrameExtractor] Video error while waiting for "${event}"`));
      };
      video.addEventListener(event, onEvent, { once: true });
      video.addEventListener('error', onError, { once: true });
    });
  }

  /**
   * Estimate native video FPS using requestVideoFrameCallback if available,
   * otherwise fall back to a 30 fps default.
   */
  private async _estimateFps(video: HTMLVideoElement): Promise<number> {
    // requestVideoFrameCallback is a non-standard but widely supported API.
    if (typeof (video as HTMLVideoElement & { requestVideoFrameCallback?: unknown })
      .requestVideoFrameCallback === 'function') {
      return new Promise<number>((resolve) => {
        let frameCount = 0;
        const MAX_SAMPLES = 10;
        const timestamps: number[] = [];

        type VFC = (_now: number, meta: { mediaTime: number }) => void;
        const callback: VFC = (_now, meta) => {
          timestamps.push(meta.mediaTime);
          frameCount++;

          if (frameCount < MAX_SAMPLES) {
            (video as HTMLVideoElement & { requestVideoFrameCallback: (cb: VFC) => void })
              .requestVideoFrameCallback(callback);
          } else {
            const elapsed = timestamps[timestamps.length - 1]! - timestamps[0]!;
            const fps = elapsed > 0 ? (MAX_SAMPLES - 1) / elapsed : 30;
            resolve(Math.round(fps));
          }
        };

        (video as HTMLVideoElement & { requestVideoFrameCallback: (cb: VFC) => void })
          .requestVideoFrameCallback(callback);

        // Start playing briefly to collect frame timestamps.
        video.play().catch(() => resolve(30));
        setTimeout(() => {
          video.pause();
          if (frameCount < 2) resolve(30);
        }, 2000);

      });
    }

    return 30;
  }
}
