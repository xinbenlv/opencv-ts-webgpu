/**
 * OpenCV.ts 5.0 — JOSH Live Demo Entry Point
 *
 * Wires together the realtime modules:
 *   LoadingStatus  — pipeline initialisation UI
 *   VideoSource    — camera / sample-file input
 *   DepthRenderer  — Node A visualisation
 *   renderSkeletonToCanvas — Node B visualisation
 *   SmplRenderer   — Node C 3D mesh viewer
 *
 * Render loop and WebGPU bootstrap live here; all substantial logic is in
 * the realtime/ sub-modules so they can be tested independently.
 */

import { getGpuDevice } from '../src/backends/webgpu/device.ts';
import { BufferManager } from '../src/core/buffer-manager.ts';
import { ResourceTracker } from '../src/core/resource-tracker.ts';
import type { NodeContext } from '../src/graph/node.ts';
import { PROC_W, PROC_H, type DisplayPx, fitToAnchor } from '../src/core/dimensions.ts';
import { SmplRenderer } from './josh/rendering/smpl-renderer.ts';
import { buildSyntheticSmplModel } from './josh/models/smpl-synthetic.ts';

import { showSecureContextWarning } from './secure-context.ts';
import { LoadingStatus } from './realtime/loading-status.ts';
import { VideoSource } from './realtime/video-source.ts';
import { DepthRenderer } from './realtime/depth-renderer.ts';
import { renderSkeletonToCanvas } from './realtime/skeleton-renderer.ts';

// ---------------------------------------------------------------------------
// Vertex normals helper (used once for SMPL mesh init)
// ---------------------------------------------------------------------------

function computeVertexNormals(vertices: Float32Array, faces: Uint32Array): Float32Array {
  const normals = new Float32Array(vertices.length);
  const faceCount = faces.length / 3;
  for (let f = 0; f < faceCount; f++) {
    const i0 = faces[f * 3]!, i1 = faces[f * 3 + 1]!, i2 = faces[f * 3 + 2]!;
    const ax = vertices[i1*3]!-vertices[i0*3]!, ay = vertices[i1*3+1]!-vertices[i0*3+1]!, az = vertices[i1*3+2]!-vertices[i0*3+2]!;
    const bx = vertices[i2*3]!-vertices[i0*3]!, by = vertices[i2*3+1]!-vertices[i0*3+1]!, bz = vertices[i2*3+2]!-vertices[i0*3+2]!;
    const nx = ay*bz - az*by, ny = az*bx - ax*bz, nz = ax*by - ay*bx;
    for (const vi of [i0, i1, i2]) {
      normals[vi*3]   = (normals[vi*3]   ?? 0) + nx;
      normals[vi*3+1] = (normals[vi*3+1] ?? 0) + ny;
      normals[vi*3+2] = (normals[vi*3+2] ?? 0) + nz;
    }
  }
  for (let v = 0; v < vertices.length / 3; v++) {
    const x = normals[v*3]!, y = normals[v*3+1]!, z = normals[v*3+2]!;
    const len = Math.sqrt(x*x + y*y + z*z);
    if (len > 0.0001) { normals[v*3] = x/len; normals[v*3+1] = y/len; normals[v*3+2] = z/len; }
    else { normals[v*3+1] = 1; }
  }
  return normals;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  if (typeof SharedArrayBuffer === 'undefined') {
    showSecureContextWarning();
    return;
  }

  // ── DOM refs ──────────────────────────────────────────────────────────────
  const statusEl       = document.getElementById('cameraStatus')!;
  const fpsEl          = document.getElementById('fps')!;
  const frameEl        = document.getElementById('frameCount')!;
  const gpuMemEl       = document.getElementById('gpuMem')!;
  const loadingOverlay = document.getElementById('loadingOverlay')!;
  const loadingSteps   = document.getElementById('loadingSteps')!;
  const statusModal    = document.getElementById('statusModal')!;
  const showStatusBtn  = document.getElementById('showStatusBtn')!;
  const closeStatusBtn = document.getElementById('closeStatusBtn')!;

  // ── Loading status ────────────────────────────────────────────────────────
  const loadingStatus = new LoadingStatus();
  loadingStatus.attachToDOM(loadingSteps, statusModal, showStatusBtn, closeStatusBtn);
  const setStep = loadingStatus.setStep.bind(loadingStatus);
  (window as any).__joshLoadingStatus = setStep;

  // ── Step 1: WebGPU ────────────────────────────────────────────────────────
  setStep('webgpu', 'active', 'Initializing WebGPU...');
  statusEl.textContent = 'Initializing WebGPU...';
  let device: GPUDevice;
  try {
    device = await getGpuDevice();
    setStep('webgpu', 'done', 'WebGPU initialized');
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    setStep('webgpu', 'error', `WebGPU not available: ${msg}`);
    statusEl.textContent = `WebGPU not available: ${msg}`;
    return;
  }

  // ── Step 2: Core infrastructure ───────────────────────────────────────────
  const resourceTracker = new ResourceTracker('demo');
  const bufferManager   = new BufferManager(device);
  resourceTracker.track(bufferManager);
  const ctx: NodeContext = { device, bufferManager, resourceTracker };

  // ── Step 3: Video source ──────────────────────────────────────────────────
  const videoEl    = document.getElementById('videoInput') as HTMLVideoElement;
  const inputSelect = document.getElementById('inputSource') as HTMLSelectElement;
  const videoSource = new VideoSource(videoEl, inputSelect, statusEl);
  await videoSource.init();

  // ── Step 4: JOSH pipeline ─────────────────────────────────────────────────
  setStep('video',     'done',    'Video source ready');
  setStep('pipeline',  'active',  'Compiling graph...');
  setStep('depthModel','pending', 'Node A: MiDAS depth model (64 MB)');
  setStep('hmrModel',  'pending', 'Node B: ROMP pose model (111 MB)');
  setStep('smplModel', 'pending', 'Node B: SMPL forward kinematics (CPU)');
  setStep('solver',    'pending', 'Node C: JOSH solver (L-BFGS + gradients)');
  statusEl.textContent = 'Compiling JOSH pipeline...';

  const { buildJoshPipeline } = await import('./josh/pipeline.ts');
  let pipelineResult;
  try {
    pipelineResult = await buildJoshPipeline(ctx);
    setStep('pipeline', 'done', 'Pipeline compiled');
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    setStep('pipeline', 'error', `Pipeline failed: ${msg}`);
    statusEl.textContent = `Pipeline build failed: ${msg}`;
    console.error(e);
    return;
  }
  const { pipeline, depthNodeId, hmrNodeId, solverNodeId } = pipelineResult;

  // Fade out loading overlay
  loadingOverlay.style.transition = 'opacity 0.5s';
  loadingOverlay.style.opacity = '0';
  setTimeout(() => { loadingOverlay.style.display = 'none'; }, 600);

  // ── Step 5: Canvas setup ──────────────────────────────────────────────────
  const FRAME_W = PROC_W as number;
  const FRAME_H = PROC_H as number;
  const frameCanvas = document.createElement('canvas');
  frameCanvas.width = FRAME_W;
  frameCanvas.height = FRAME_H;
  const frameCtx2d  = frameCanvas.getContext('2d', { willReadFrequently: true })!;
  const rgbFloat32  = new Float32Array(FRAME_H * FRAME_W * 3);

  const depthCanvas  = document.getElementById('depthCanvas')  as HTMLCanvasElement;
  const meshCanvas   = document.getElementById('meshCanvas')   as HTMLCanvasElement;
  const outputCanvas = document.getElementById('outputCanvas') as HTMLCanvasElement;

  let displayW: DisplayPx = FRAME_W as DisplayPx;
  let displayH: DisplayPx = FRAME_H as DisplayPx;

  function applyDisplaySize() {
    const { w, h } = fitToAnchor(videoEl.videoWidth || FRAME_W, videoEl.videoHeight || FRAME_H, FRAME_H);
    displayW = w; displayH = h;
    for (const c of [depthCanvas, meshCanvas, outputCanvas]) {
      c.width  = displayW as number;
      c.height = displayH as number;
    }
  }
  videoEl.addEventListener('loadedmetadata', applyDisplaySize);
  if (videoEl.readyState >= HTMLMediaElement.HAVE_METADATA) applyDisplaySize();

  const depthCtx = depthCanvas.getContext('2d')!;
  const meshCtx  = meshCanvas.getContext('2d')!;

  const depthRenderer = new DepthRenderer(FRAME_W, FRAME_H);

  // Node C: SmplRenderer — 3D WebGPU mesh viewer
  const smplModelData = buildSyntheticSmplModel();
  const smplNormals   = computeVertexNormals(smplModelData.meanTemplate, smplModelData.faces);
  const smplRenderer  = new SmplRenderer(device, outputCanvas);
  smplRenderer.setMesh(smplModelData.meanTemplate, smplNormals, smplModelData.faces);

  // ── Step 6: Render loop ───────────────────────────────────────────────────
  let frameCount = 0;
  let lastTime   = performance.now();
  let fps        = 0;

  async function renderFrame() {
    const now = performance.now();
    frameCount++;

    if (now - lastTime >= 1000) {
      fps = frameCount; frameCount = 0; lastTime = now;
      fpsEl.textContent    = String(fps);
      gpuMemEl.textContent = `${(bufferManager.allocatedBytes / 1024 / 1024).toFixed(1)} MB`;
    }
    frameEl.textContent = String(frameCount);

    // Upload current video frame to all graph input ports
    if (videoEl.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      frameCtx2d.drawImage(videoEl, 0, 0, FRAME_W, FRAME_H);
      const imageData = frameCtx2d.getImageData(0, 0, FRAME_W, FRAME_H);
      const px = imageData.data;
      for (let i = 0; i < FRAME_H * FRAME_W; i++) {
        rgbFloat32[i * 3]     = (px[i * 4]     ?? 0) / 255;
        rgbFloat32[i * 3 + 1] = (px[i * 4 + 1] ?? 0) / 255;
        rgbFloat32[i * 3 + 2] = (px[i * 4 + 2] ?? 0) / 255;
      }
      for (const { nodeId, portName } of pipeline.graphInputs) {
        pipeline.writeInput(nodeId, portName, rgbFloat32.buffer);
      }
    }

    try {
      await pipeline.execute(frameCount);

      if (frameCount % 3 === 0) {
        // Node A: depth map
        const depthArr = new Float32Array(await pipeline.readOutput(depthNodeId, 'depthMap'));
        depthRenderer.render(depthArr, depthCtx, displayW as number, displayH as number);

        if (frameCount <= 9) {
          let dMin = Infinity, dMax = -Infinity, dSum = 0;
          for (let i = 0; i < depthArr.length; i++) {
            const v = depthArr[i]!;
            if (v < dMin) dMin = v; if (v > dMax) dMax = v; dSum += v;
          }
          console.log(`[Depth] frame=${frameCount} min=${dMin.toFixed(4)} max=${dMax.toFixed(4)} mean=${(dSum/depthArr.length).toFixed(4)}`);
        }

        // Node B: skeleton overlay
        const jointsArr = new Float32Array(await pipeline.readOutput(hmrNodeId, 'jointPositions'));
        const camArr    = new Float32Array(await pipeline.readOutput(hmrNodeId, 'estimatedCamera'));
        renderSkeletonToCanvas(jointsArr, camArr, meshCtx, displayW as number, displayH as number, videoEl);

        if (frameCount <= 9) {
          const j0  = [jointsArr[0],  jointsArr[1],  jointsArr[2]];
          const j15 = [jointsArr[45], jointsArr[46], jointsArr[47]];
          console.log(`[Joints] frame=${frameCount} pelvis=[${j0.map(v => v?.toFixed(3))}] head=[${j15.map(v => v?.toFixed(3))}]`);
        }

        // Node C: 3D mesh
        smplRenderer.updateVertices(new Float32Array(await pipeline.readOutput(solverNodeId, 'refinedVertices')));
        smplRenderer.render({ eye: [0, 0.9, 3.5], target: [0, 0.9, 0], up: [0, 1, 0], fovY: Math.PI / 4, near: 0.1, far: 10.0 });
      }
    } catch (e) {
      console.error('Frame execution error:', e);
    }

    requestAnimationFrame(renderFrame);
  }

  requestAnimationFrame(renderFrame);
}

main().catch(console.error);
