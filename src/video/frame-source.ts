import { CvError } from '../core/errors.ts';
import type { Disposable } from '../core/resource-tracker.ts';

/**
 * Video frame source using the WebCodecs API.
 *
 * Acquires video frames from camera or media elements and delivers them
 * as VideoFrame objects for GPU import via GPUExternalTexture.
 */
export class FrameSource implements Disposable {
  private _stream: MediaStream | null = null;
  private _videoTrack: MediaStreamTrack | null = null;
  private _reader: ReadableStreamDefaultReader<VideoFrame> | null = null;
  private _disposed = false;

  /**
   * Open a camera stream.
   */
  async openCamera(constraints?: MediaTrackConstraints): Promise<void> {
    this._stream = await navigator.mediaDevices.getUserMedia({
      video: constraints ?? { width: 640, height: 480, facingMode: 'user' },
      audio: false,
    });

    const tracks = this._stream.getVideoTracks();
    if (tracks.length === 0) {
      throw new CvError('No video track found in camera stream.');
    }
    this._videoTrack = tracks[0]!;

    // Use MediaStreamTrackProcessor for frame-by-frame access
    const processor = new MediaStreamTrackProcessor({ track: this._videoTrack });
    this._reader = processor.readable.getReader();
  }

  /**
   * Read the next video frame.
   * The caller is responsible for closing the returned VideoFrame.
   */
  async nextFrame(): Promise<VideoFrame | null> {
    if (!this._reader || this._disposed) return null;

    const { value, done } = await this._reader.read();
    if (done) return null;
    return value;
  }

  /**
   * Get the video track settings (resolution, frame rate).
   */
  get settings(): MediaTrackSettings | null {
    return this._videoTrack?.getSettings() ?? null;
  }

  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;
    this._reader?.cancel();
    this._videoTrack?.stop();
    this._stream = null;
  }
}
