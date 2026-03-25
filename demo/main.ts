import { getGpuDevice } from '../src/backends/webgpu/device.ts';
import { BufferManager } from '../src/core/buffer-manager.ts';
import { ResourceTracker } from '../src/core/resource-tracker.ts';
import type { NodeContext } from '../src/graph/node.ts';

/**
 * OpenCV.ts 5.0 — JOSH Live Demo Entry Point
 *
 * This demo shows the complete JOSH pipeline running in the browser:
 * 1. Camera feed → WebCodecs VideoFrame
 * 2. Node A (Depth Estimation) + Node B (HMR) in parallel on WebGPU
 * 3. Node C (JOSH Solver) joint optimization via WASM + WebGPU
 * 4. 3D viewer renders the reconstructed scene + human mesh
 */

async function main() {
  const statusEl = document.getElementById('cameraStatus')!;
  const fpsEl = document.getElementById('fps')!;
  const frameEl = document.getElementById('frameCount')!;
  const gpuMemEl = document.getElementById('gpuMem')!;

  // ─── Step 1: Initialize WebGPU ───
  statusEl.textContent = 'Initializing WebGPU...';
  let device: GPUDevice;
  try {
    device = await getGpuDevice();
  } catch (e) {
    statusEl.textContent = `WebGPU not available: ${e instanceof Error ? e.message : String(e)}`;
    return;
  }

  // ─── Step 2: Create core infrastructure ───
  const resourceTracker = new ResourceTracker('demo');
  const bufferManager = new BufferManager(device);
  resourceTracker.track(bufferManager);

  const ctx: NodeContext = {
    device,
    bufferManager,
    resourceTracker,
  };

  // ─── Step 3: Open camera ───
  statusEl.textContent = 'Requesting camera access...';
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
      audio: false,
    });
    const videoEl = document.getElementById('videoInput') as HTMLVideoElement;
    videoEl.srcObject = stream;
    await videoEl.play();
    statusEl.style.display = 'none';
  } catch (e) {
    statusEl.textContent = `Camera access denied: ${e instanceof Error ? e.message : String(e)}`;
    return;
  }

  // ─── Step 4: Build JOSH pipeline ───
  statusEl.textContent = 'Compiling JOSH pipeline...';

  // Dynamic import to allow tree-shaking in library builds
  const { buildJoshPipeline } = await import('../src/graphs/josh/pipeline.ts');

  let pipeline;
  try {
    pipeline = await buildJoshPipeline(ctx);
  } catch (e) {
    statusEl.textContent = `Pipeline build failed: ${e instanceof Error ? e.message : String(e)}`;
    console.error(e);
    return;
  }

  // ─── Step 5: Render loop ───
  let frameCount = 0;
  let lastTime = performance.now();
  let fps = 0;

  async function renderFrame() {
    const now = performance.now();
    frameCount++;

    // FPS calculation (rolling average)
    if (now - lastTime >= 1000) {
      fps = frameCount;
      frameCount = 0;
      lastTime = now;
      fpsEl.textContent = String(fps);
      gpuMemEl.textContent = `${(bufferManager.allocatedBytes / 1024 / 1024).toFixed(1)} MB`;
    }

    frameEl.textContent = String(frameCount);

    try {
      await pipeline!.execute(frameCount);
    } catch (e) {
      console.error('Frame execution error:', e);
    }

    requestAnimationFrame(renderFrame);
  }

  requestAnimationFrame(renderFrame);
}

main().catch(console.error);
