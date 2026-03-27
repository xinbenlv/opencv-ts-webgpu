import { getGpuDevice } from '../src/backends/webgpu/device.ts';
import { BufferManager } from '../src/core/buffer-manager.ts';
import { ResourceTracker } from '../src/core/resource-tracker.ts';
import type { NodeContext } from '../src/graph/node.ts';
import { PROC_W, PROC_H, type DisplayPx, fitToAnchor } from '../src/core/dimensions.ts';
import { SmplRenderer } from './josh/rendering/smpl-renderer.ts';
import { buildSyntheticSmplModel } from './josh/models/smpl-synthetic.ts';

/**
 * OpenCV.ts 5.0 — JOSH Live Demo Entry Point
 *
 * This demo shows the complete JOSH pipeline running in the browser:
 * 1. Camera feed → WebCodecs VideoFrame
 * 2. Node A (Depth Estimation) + Node B (HMR) in parallel on WebGPU
 * 3. Node C (JOSH Solver) joint optimization via WASM + WebGPU
 * 4. 3D viewer renders the reconstructed scene + human mesh
 */

function showSecureContextWarning() {
  const modal = document.createElement('div');
  modal.style.cssText = `
    position: fixed; bottom: 1rem; right: 1rem; z-index: 9999;
    max-width: 420px; padding: 1rem 1.25rem;
    background: #1a0000; border: 1px solid #ff4444; border-radius: 8px;
    color: #ff8888; font-family: system-ui, sans-serif; font-size: 0.85rem;
    line-height: 1.5; box-shadow: 0 4px 24px rgba(255,0,0,0.2);
  `;
  modal.innerHTML = `
    <strong style="color:#ff4444;">Secure Context Required</strong><br>
    <code>SharedArrayBuffer</code> is not available. This app requires a
    <a href="https://w3c.github.io/webappsec-secure-contexts/"
       target="_blank" rel="noopener" style="color:#ff6666;">secure context (HTTPS)</a>
    to protect against
    <a href="https://www.w3.org/TR/post-spectre-webdev/#shared-array-buffer"
       target="_blank" rel="noopener" style="color:#ff6666;">Spectre-class attacks</a>.<br><br>
    Access via <code>https://localhost:5173</code> or use an SSH tunnel for remote access.
  `;
  document.body.appendChild(modal);
}

async function main() {
  // ─── Check secure context ───
  if (typeof SharedArrayBuffer === 'undefined') {
    showSecureContextWarning();
    return;
  }

  const statusEl = document.getElementById('cameraStatus')!;
  const fpsEl = document.getElementById('fps')!;
  const frameEl = document.getElementById('frameCount')!;
  const gpuMemEl = document.getElementById('gpuMem')!;
  const loadingOverlay = document.getElementById('loadingOverlay')!;
  const loadingSteps = document.getElementById('loadingSteps')!;
  const statusModal = document.getElementById('statusModal')!;
  const statusModalSteps = document.getElementById('statusModalSteps')!;
  const showStatusBtn = document.getElementById('showStatusBtn')!;
  const closeStatusBtn = document.getElementById('closeStatusBtn')!;

  // ─── Loading status tracker ───
  const stepStates = new Map<string, { status: 'pending' | 'active' | 'done' | 'warn' | 'error'; text: string }>();

  function renderSteps(): string {
    const icons = { pending: '\u2502', active: '\u25B6', done: '\u2714', warn: '\u26A0', error: '\u2718' };
    const colors = { pending: '#555', active: '#60a5fa', done: '#4ade80', warn: '#fbbf24', error: '#f87171' };
    return [...stepStates.entries()].map(([, { status, text }]) =>
      `<div style="color:${colors[status]}">${icons[status]} ${text}</div>`
    ).join('');
  }

  function updateLoadingUI() {
    const html = renderSteps();
    loadingSteps.innerHTML = html;
    statusModalSteps.innerHTML = html; // keep modal in sync
  }

  function setStep(id: string, status: 'pending' | 'active' | 'done' | 'warn' | 'error', text: string) {
    stepStates.set(id, { status, text });
    updateLoadingUI();
  }

  // Status modal toggle
  showStatusBtn.addEventListener('click', () => {
    statusModal.style.display = 'flex';
    statusModalSteps.innerHTML = renderSteps();
  });
  closeStatusBtn.addEventListener('click', () => { statusModal.style.display = 'none'; });
  statusModal.addEventListener('click', (e) => {
    if (e.target === statusModal) statusModal.style.display = 'none';
  });

  // Expose globally so nodes can report progress
  (window as any).__joshLoadingStatus = setStep;

  setStep('webgpu', 'active', 'Initializing WebGPU...');

  // ─── Step 1: Initialize WebGPU ───
  statusEl.textContent = 'Initializing WebGPU...';
  let device: GPUDevice;
  try {
    device = await getGpuDevice();
    setStep('webgpu', 'done', 'WebGPU initialized');
  } catch (e) {
    setStep('webgpu', 'error', `WebGPU not available: ${e instanceof Error ? e.message : String(e)}`);
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

  // ─── Step 3: Video input source ───
  const videoEl = document.getElementById('videoInput') as HTMLVideoElement;
  const inputSelect = document.getElementById('inputSource') as HTMLSelectElement;
  let cameraStream: MediaStream | null = null;

  async function switchToCamera() {
    try {
      statusEl.textContent = 'Requesting camera access...';
      statusEl.style.display = '';
      if (videoEl.src) { videoEl.removeAttribute('src'); videoEl.load(); }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: false,
      });
      cameraStream = stream;
      videoEl.srcObject = stream;
      await videoEl.play();
      statusEl.style.display = 'none';
    } catch {
      statusEl.textContent = 'Camera unavailable';
      inputSelect.value = './assets/josh-demo.mp4';
      await switchToSample();
    }
  }

  async function switchToSample() {
    if (cameraStream) {
      cameraStream.getTracks().forEach((t) => t.stop());
      cameraStream = null;
    }
    videoEl.srcObject = null;
    videoEl.src = inputSelect.value;
    videoEl.loop = true;
    videoEl.muted = true;
    await videoEl.play().catch(() => {});
    statusEl.style.display = 'none';
  }

  inputSelect.addEventListener('change', () => {
    if (inputSelect.value === 'camera') {
      switchToCamera();
    } else {
      switchToSample();
    }
  });

  // Try camera first, fall back to sample
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
      audio: false,
    });
    cameraStream = stream;
    videoEl.srcObject = stream;
    await videoEl.play();
    statusEl.style.display = 'none';
    inputSelect.value = 'camera';
  } catch {
    inputSelect.value = './assets/josh-demo.mp4';
    await switchToSample();
  }

  // ─── Step 4: Build JOSH pipeline ───
  setStep('video', 'done', 'Video source ready');
  setStep('pipeline', 'active', 'Compiling graph...');
  setStep('depthModel', 'pending', 'Node A: MiDAS depth model (64 MB)');
  setStep('hmrModel', 'pending', 'Node B: ROMP pose model (111 MB)');
  setStep('smplModel', 'pending', 'Node B: SMPL forward kinematics (CPU)');
  setStep('solver', 'pending', 'Node C: JOSH solver (L-BFGS + gradients)');
  statusEl.textContent = 'Compiling JOSH pipeline...';

  // Dynamic import to allow tree-shaking in library builds
  const { buildJoshPipeline } = await import('./josh/pipeline.ts');

  let pipelineResult;
  try {
    pipelineResult = await buildJoshPipeline(ctx);
    setStep('pipeline', 'done', 'Pipeline compiled');
  } catch (e) {
    setStep('pipeline', 'error', `Pipeline failed: ${e instanceof Error ? e.message : String(e)}`);
    statusEl.textContent = `Pipeline build failed: ${e instanceof Error ? e.message : String(e)}`;
    console.error(e);
    return;
  }

  const { pipeline, depthNodeId, hmrNodeId, solverNodeId } = pipelineResult;

  // Dismiss loading overlay with fade
  loadingOverlay.style.transition = 'opacity 0.5s';
  loadingOverlay.style.opacity = '0';
  setTimeout(() => { loadingOverlay.style.display = 'none'; }, 600);

  // ─── Step 5: Render loop ───
  // Internal processing resolution (GPU buffers are always PROC_W × PROC_H)
  const FRAME_W = PROC_W as number;
  const FRAME_H = PROC_H as number;
  const frameCanvas = document.createElement('canvas');
  frameCanvas.width = FRAME_W;
  frameCanvas.height = FRAME_H;
  const frameCtx2d = frameCanvas.getContext('2d', { willReadFrequently: true })!;
  const rgbFloat32 = new Float32Array(FRAME_H * FRAME_W * 3);

  // Visualization canvases — displayed at the video's native aspect ratio
  const depthCanvas = document.getElementById('depthCanvas') as HTMLCanvasElement;
  const meshCanvas = document.getElementById('meshCanvas') as HTMLCanvasElement;
  const outputCanvas = document.getElementById('outputCanvas') as HTMLCanvasElement;

  // Mutable display dimensions (updated when video metadata loads)
  let displayW: DisplayPx = FRAME_W as DisplayPx;
  let displayH: DisplayPx = FRAME_H as DisplayPx;

  function applyDisplaySize() {
    const { w, h } = fitToAnchor(videoEl.videoWidth || FRAME_W, videoEl.videoHeight || FRAME_H, FRAME_H);
    displayW = w;
    displayH = h;
    for (const c of [depthCanvas, meshCanvas, outputCanvas]) {
      c.width = displayW as number;
      c.height = displayH as number;
    }
  }

  videoEl.addEventListener('loadedmetadata', applyDisplaySize);
  // Apply immediately in case metadata is already available
  if (videoEl.readyState >= HTMLMediaElement.HAVE_METADATA) applyDisplaySize();

  const depthCtx = depthCanvas.getContext('2d')!;
  const meshCtx = meshCanvas.getContext('2d')!;

  // Node C: SmplRenderer — 3D WebGPU mesh viewer (replaces flat depth passthrough)
  function computeVertexNormals(vertices: Float32Array, faces: Uint32Array): Float32Array {
    const normals = new Float32Array(vertices.length);
    const faceCount = faces.length / 3;
    for (let f = 0; f < faceCount; f++) {
      const i0 = faces[f * 3]!, i1 = faces[f * 3 + 1]!, i2 = faces[f * 3 + 2]!;
      const ax = vertices[i1*3]!-vertices[i0*3]!, ay = vertices[i1*3+1]!-vertices[i0*3+1]!, az = vertices[i1*3+2]!-vertices[i0*3+2]!;
      const bx = vertices[i2*3]!-vertices[i0*3]!, by = vertices[i2*3+1]!-vertices[i0*3+1]!, bz = vertices[i2*3+2]!-vertices[i0*3+2]!;
      const nx = ay*bz - az*by, ny = az*bx - ax*bz, nz = ax*by - ay*bx;
      for (const vi of [i0, i1, i2]) {
        normals[vi*3] = (normals[vi*3] ?? 0) + nx;
        normals[vi*3+1] = (normals[vi*3+1] ?? 0) + ny;
        normals[vi*3+2] = (normals[vi*3+2] ?? 0) + nz;
      }
    }
    for (let v = 0; v < vertices.length / 3; v++) {
      const x = normals[v*3]!, y = normals[v*3+1]!, z = normals[v*3+2]!;
      const len = Math.sqrt(x*x + y*y + z*z);
      if (len > 0.0001) {
        normals[v*3] = x / len; normals[v*3+1] = y / len; normals[v*3+2] = z / len;
      } else { normals[v*3+1] = 1; }
    }
    return normals;
  }
  const smplModelData = buildSyntheticSmplModel();
  const smplNormals = computeVertexNormals(smplModelData.meanTemplate, smplModelData.faces);
  const smplRenderer = new SmplRenderer(device, outputCanvas);
  smplRenderer.setMesh(smplModelData.meanTemplate, smplNormals, smplModelData.faces);

  // SMPL skeleton definition (24 joints, parent indices)
  const SMPL_PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21];
  const SMPL_BONE_COLORS = [
    '#4ade80', // torso: green
    '#60a5fa', // left side: blue
    '#f87171', // right side: red
  ];
  function boneColor(i: number): string {
    const name = (['pelvis','l_hip','r_hip','spine1','l_knee','r_knee','spine2',
      'l_ankle','r_ankle','spine3','l_foot','r_foot','neck','l_collar','r_collar',
      'head','l_shoulder','r_shoulder','l_elbow','r_elbow','l_wrist','r_wrist',
      'l_hand','r_hand'] as const)[i] ?? '';
    if (name.startsWith('l_')) return SMPL_BONE_COLORS[1]!;
    if (name.startsWith('r_')) return SMPL_BONE_COLORS[2]!;
    return SMPL_BONE_COLORS[0]!;
  }

  // T-pose reference positions (normalized 0–1 in canvas space)
  const TPOSE: [number, number][] = [
    [0.50, 0.45], // 0 pelvis
    [0.44, 0.48], // 1 l_hip
    [0.56, 0.48], // 2 r_hip
    [0.50, 0.38], // 3 spine1
    [0.44, 0.62], // 4 l_knee
    [0.56, 0.62], // 5 r_knee
    [0.50, 0.32], // 6 spine2
    [0.44, 0.78], // 7 l_ankle
    [0.56, 0.78], // 8 r_ankle
    [0.50, 0.26], // 9 spine3
    [0.44, 0.84], // 10 l_foot
    [0.56, 0.84], // 11 r_foot
    [0.50, 0.20], // 12 neck
    [0.46, 0.22], // 13 l_collar
    [0.54, 0.22], // 14 r_collar
    [0.50, 0.12], // 15 head
    [0.36, 0.22], // 16 l_shoulder
    [0.64, 0.22], // 17 r_shoulder
    [0.28, 0.32], // 18 l_elbow
    [0.72, 0.32], // 19 r_elbow
    [0.22, 0.42], // 20 l_wrist
    [0.78, 0.42], // 21 r_wrist
    [0.20, 0.45], // 22 l_hand
    [0.80, 0.45], // 23 r_hand
  ];

  function renderSkeletonToCanvas(
    joints: Float32Array,
    cam: Float32Array,    // [scale, tx, ty] weak-perspective camera from HMR
    ctx2d: CanvasRenderingContext2D,
    w: number,
    h: number,
  ) {
    // Draw video frame as background — preserve video's native aspect ratio
    const vW = videoEl.videoWidth || w;
    const vH = videoEl.videoHeight || h;
    const srcAR = vW / vH;
    const dstAR = w / h;
    let sx = 0, sy = 0, sw = vW, sh = vH;
    if (Math.abs(srcAR - dstAR) > 0.01) {
      if (srcAR > dstAR) {
        sw = Math.round(vH * dstAR);
        sx = Math.round((vW - sw) / 2);
      } else {
        sh = Math.round(vW / dstAR);
        sy = Math.round((vH - sh) / 2);
      }
    }
    ctx2d.drawImage(videoEl, sx, sy, sw, sh, 0, 0, w, h);
    ctx2d.fillStyle = 'rgba(0,0,0,0.35)';
    ctx2d.fillRect(0, 0, w, h);

    // Check if joints are all zero (placeholder) → use T-pose reference
    let sumAbs = 0;
    for (let i = 0; i < Math.min(joints.length, 72); i++) sumAbs += Math.abs(joints[i]!);
    const allZero = sumAbs < 0.01;

    // ── Project 3D metric joints → 2D canvas pixels ──────────────────────────
    // Weak-perspective camera: cam = [s, tx, ty]
    //   s   — scale factor (maps metres to canvas fraction)
    //   tx  — horizontal offset in [-1, 1] → half at 0 means centre
    //   ty  — vertical offset in [-1, 1]
    //
    // Projection: px = (s * Jx + tx) * w/2 + w/2
    //             py = (-s * Jy + ty) * h/2 + h/2   (world Y is up, canvas Y is down)
    //
    // When cam is all-zeros (no detection), fall back to T-pose reference layout.
    const camScale = cam[0] ?? 0;
    const camTx    = cam[1] ?? 0;
    const camTy    = cam[2] ?? 0;
    const haveCamera = camScale > 0.05;

    const positions: [number, number][] = [];

    if (allZero) {
      // T-pose reference (normalised canvas layout)
      for (let j = 0; j < 24; j++) {
        positions.push([TPOSE[j]![0] * w, TPOSE[j]![1] * h]);
      }
    } else if (haveCamera) {
      // Weak-perspective projection: place skeleton on the detected person.
      // FK joint positions are in absolute SMPL world space (pelvis ≈ Y=0.9 m),
      // so subtract the FK pelvis before scaling so the projection is pelvis-relative.
      const pelvisXWorld = joints[0]!;
      const pelvisYWorld = joints[1]!;
      for (let j = 0; j < 24; j++) {
        const Jx = joints[j * 3]! - pelvisXWorld;
        const Jy = joints[j * 3 + 1]! - pelvisYWorld;
        const px = (camScale * Jx + camTx) * (w / 2) + w / 2;
        const py = (-camScale * Jy + camTy) * (h / 2) + h / 2;
        positions.push([px, py]);
      }
    } else {
      // No camera but joints available — auto-scale & centre (abstract view)
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (let j = 0; j < 24; j++) {
        const jx = joints[j * 3]!;
        const jy = joints[j * 3 + 1]!;
        if (jx < minX) minX = jx; if (jx > maxX) maxX = jx;
        if (jy < minY) minY = jy; if (jy > maxY) maxY = jy;
      }
      const rangeX = maxX - minX || 1;
      const rangeY = maxY - minY || 1;
      const cx = (minX + maxX) / 2;
      const cy = (minY + maxY) / 2;
      const scale = Math.min(w, h) * 0.7 / Math.max(rangeX, rangeY);
      for (let j = 0; j < 24; j++) {
        const x = (joints[j * 3]! - cx) * scale + w * 0.5;
        const y = -(joints[j * 3 + 1]! - cy) * scale + h * 0.5;
        positions.push([x, y]);
      }
    }

    // Draw bones
    ctx2d.lineWidth = 3;
    for (let j = 1; j < 24; j++) {
      const parent = SMPL_PARENTS[j]!;
      ctx2d.strokeStyle = boneColor(j);
      ctx2d.beginPath();
      ctx2d.moveTo(positions[parent]![0], positions[parent]![1]);
      ctx2d.lineTo(positions[j]![0], positions[j]![1]);
      ctx2d.stroke();
    }

    // Draw joints
    for (let j = 0; j < 24; j++) {
      const [x, y] = positions[j]!;
      ctx2d.fillStyle = j === 15 ? '#fbbf24' : boneColor(j); // head = yellow
      ctx2d.beginPath();
      ctx2d.arc(x, y, j === 15 ? 8 : 4, 0, Math.PI * 2);
      ctx2d.fill();
      ctx2d.strokeStyle = '#000';
      ctx2d.lineWidth = 1;
      ctx2d.stroke();
    }

    // Label
    const label = allZero ? 'SMPL T-pose (no detection)' : haveCamera ? 'SMPL Joints — tracked' : 'SMPL Joints (abstract)';
    ctx2d.fillStyle = allZero ? '#666' : haveCamera ? '#4ade80' : '#fbbf24';
    ctx2d.font = '11px system-ui';
    ctx2d.textAlign = 'left';
    ctx2d.fillText(label, 8, h - 8);
  }

  let frameCount = 0;
  let lastTime = performance.now();
  let fps = 0;

  // Reusable off-screen canvas for the 384×384 depth data (avoids allocation per frame)
  const depthOffscreen = document.createElement('canvas');
  depthOffscreen.width = FRAME_W;
  depthOffscreen.height = FRAME_H;
  const depthOffCtx = depthOffscreen.getContext('2d')!;

  function renderDepthToCanvas(
    depthData: Float32Array,
    ctx2d: CanvasRenderingContext2D,
    dstW: number,
    dstH: number,
  ) {
    // depthData is always FRAME_W × FRAME_H (processing resolution)
    // Render to offscreen 384×384 first, then scale-blit to the display canvas

    // Find min/max for normalization
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < depthData.length; i++) {
      const v = depthData[i]!;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = max - min || 1;

    const imgData = depthOffCtx.createImageData(FRAME_W, FRAME_H);
    const px = imgData.data;
    for (let i = 0; i < depthData.length; i++) {
      const norm = (depthData[i]! - min) / range;
      // Viridis-inspired colormap: blue → green → yellow
      const r = Math.min(255, Math.max(0, (norm * 3 - 1) * 255));
      const g = Math.min(255, Math.max(0, Math.sin(norm * Math.PI) * 255));
      const b = Math.min(255, Math.max(0, (1 - norm * 2) * 255));
      px[i * 4] = r;
      px[i * 4 + 1] = g;
      px[i * 4 + 2] = b;
      px[i * 4 + 3] = 255;
    }
    depthOffCtx.putImageData(imgData, 0, 0);

    // Scale to display canvas (letterbox if aspect ratios differ)
    const srcAR = FRAME_W / FRAME_H;
    const dstAR = dstW / dstH;
    let sx = 0, sy = 0, sw = FRAME_W, sh = FRAME_H;
    if (Math.abs(srcAR - dstAR) > 0.01) {
      // Letterbox: crop source to match destination AR
      if (srcAR > dstAR) {
        sw = Math.round(FRAME_H * dstAR);
        sx = Math.round((FRAME_W - sw) / 2);
      } else {
        sh = Math.round(FRAME_W / dstAR);
        sy = Math.round((FRAME_H - sh) / 2);
      }
    }
    ctx2d.drawImage(depthOffscreen, sx, sy, sw, sh, 0, 0, dstW, dstH);
  }

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

      // Visualize outputs every 3 frames (throttle GPU readback)
      if (frameCount % 3 === 0) {
        // Node A: depth estimation output
        const depthBuf = await pipeline.readOutput(depthNodeId, 'depthMap');
        const depthArr = new Float32Array(depthBuf);
        renderDepthToCanvas(depthArr, depthCtx, displayW as number, displayH as number);

        // Debug depth range (first few frames only)
        if (frameCount <= 9) {
          let dMin = Infinity, dMax = -Infinity, dSum = 0;
          for (let i = 0; i < depthArr.length; i++) {
            const v = depthArr[i]!;
            if (v < dMin) dMin = v;
            if (v > dMax) dMax = v;
            dSum += v;
          }
          console.log(`[Depth] frame=${frameCount} min=${dMin.toFixed(4)} max=${dMax.toFixed(4)} mean=${(dSum/depthArr.length).toFixed(4)}`);
        }

        // Node B: HMR skeleton overlay
        const jointsBuf = await pipeline.readOutput(hmrNodeId, 'jointPositions');
        const jointsArr = new Float32Array(jointsBuf);
        const camBuf = await pipeline.readOutput(hmrNodeId, 'estimatedCamera');
        const camArr = new Float32Array(camBuf);
        renderSkeletonToCanvas(jointsArr, camArr, meshCtx, displayW as number, displayH as number);

        // Debug joints (first few frames)
        if (frameCount <= 9) {
          const j0 = [jointsArr[0], jointsArr[1], jointsArr[2]];
          const j15 = [jointsArr[45], jointsArr[46], jointsArr[47]];
          console.log(`[Joints] frame=${frameCount} pelvis=[${j0.map(v => v?.toFixed(3))}] head=[${j15.map(v => v?.toFixed(3))}]`);
        }

        // Node C: render refined SMPL mesh via SmplRenderer
        const vertBuf = await pipeline.readOutput(solverNodeId, 'refinedVertices');
        smplRenderer.updateVertices(new Float32Array(vertBuf));
        smplRenderer.render({
          eye: [0, 0.9, 3.5],
          target: [0, 0.9, 0],
          up: [0, 1, 0],
          fovY: Math.PI / 4,
          near: 0.1,
          far: 10.0,
        });
      }
    } catch (e) {
      console.error('Frame execution error:', e);
    }

    requestAnimationFrame(renderFrame);
  }

  requestAnimationFrame(renderFrame);
}

main().catch(console.error);
