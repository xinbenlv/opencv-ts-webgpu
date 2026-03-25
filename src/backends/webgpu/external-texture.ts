import { GpuError } from '../../core/errors.ts';

/**
 * Adapter for importing VideoFrame sources into WebGPU via GPUExternalTexture.
 *
 * WebCodecs VideoFrame → GPUExternalTexture allows zero-copy video frame
 * ingestion from camera or media sources directly into compute shaders.
 */
export class ExternalTextureAdapter {
  private readonly _device: GPUDevice;

  constructor(device: GPUDevice) {
    this._device = device;
  }

  /**
   * Import a VideoFrame as a GPUExternalTexture.
   *
   * Note: GPUExternalTexture is valid only for the current task (microtask).
   * It must be used immediately in a bind group and submitted before yielding.
   */
  importVideoFrame(frame: VideoFrame): GPUExternalTexture {
    if (!frame || frame.codedWidth === 0) {
      throw new GpuError('Invalid VideoFrame: width is 0 or frame is closed.');
    }

    return this._device.importExternalTexture({
      source: frame,
      label: `videoframe_${frame.timestamp}`,
    });
  }

  /**
   * Import an HTMLVideoElement as a GPUExternalTexture.
   */
  importVideo(video: HTMLVideoElement): GPUExternalTexture {
    if (video.readyState < 2) {
      throw new GpuError('Video element not ready (readyState < HAVE_CURRENT_DATA).');
    }

    return this._device.importExternalTexture({
      source: video,
      label: 'video_element',
    });
  }

  /**
   * Create a bind group layout entry for an external texture binding.
   */
  static bindGroupLayoutEntry(bindingIndex: number): GPUBindGroupLayoutEntry {
    return {
      binding: bindingIndex,
      visibility: GPUShaderStage.COMPUTE,
      externalTexture: {},
    };
  }
}
