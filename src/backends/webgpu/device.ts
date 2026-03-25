import { GpuError } from '../../core/errors.ts';

let _cachedDevice: GPUDevice | null = null;

/**
 * Request and cache a GPUDevice singleton.
 *
 * In browser contexts, there's typically one logical device per tab.
 * This function handles the adapter → device negotiation and caches the result.
 */
export async function getGpuDevice(
  options?: {
    powerPreference?: GPUPowerPreference;
    requiredFeatures?: GPUFeatureName[];
  },
): Promise<GPUDevice> {
  if (_cachedDevice) return _cachedDevice;

  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new GpuError(
      'WebGPU is not available. ' +
        `navigator.gpu=${typeof navigator !== 'undefined' ? String(!!navigator.gpu) : 'N/A'}, ` +
        `isSecureContext=${typeof window !== 'undefined' ? String(window.isSecureContext) : 'N/A'}, ` +
        `protocol=${typeof location !== 'undefined' ? location.protocol : 'N/A'}. ` +
        'WebGPU requires a secure context (HTTPS or localhost).',
    );
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: options?.powerPreference ?? 'high-performance',
  });

  if (!adapter) {
    throw new GpuError('Failed to obtain GPUAdapter. No compatible GPU found.');
  }

  const device = await adapter.requestDevice({
    requiredFeatures: options?.requiredFeatures ?? [],
  });

  device.lost.then((info) => {
    console.error(`WebGPU device lost: ${info.message} (reason: ${info.reason})`);
    _cachedDevice = null;
  });

  _cachedDevice = device;
  return device;
}

/**
 * Destroy the cached device and release GPU resources.
 */
export function releaseGpuDevice(): void {
  if (_cachedDevice) {
    _cachedDevice.destroy();
    _cachedDevice = null;
  }
}
