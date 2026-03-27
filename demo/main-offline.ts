/**
 * main-offline.ts — Entry point for offline JOSH processing mode.
 *
 * Flow: Drag-drop video → batch process → 3D viewer
 *
 * This module bootstraps the offline tab UI when the page loads in offline
 * mode (i.e. index.html with ?mode=offline, or when index.html switches to
 * the Offline tab).
 */

import { SMPLLoaderUI } from './josh/models/smpl-loader-ui.ts';
import type { SMPLModelData } from './josh/models/smpl-loader-ui.ts';
import { BatchPipeline } from './josh/batch/batch-pipeline.ts';
import type { BatchProgress } from './josh/batch/batch-pipeline.ts';
import { getGpuDevice } from '../src/backends/webgpu/device.ts';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface OfflineUIElements {
  smplContainer: HTMLElement;
  videoDropZone: HTMLElement;
  videoFileInput: HTMLInputElement;
  processBtn: HTMLButtonElement;
  progressSection: HTMLElement;
  phase1Bar: HTMLProgressElement;
  phase2Bar: HTMLProgressElement;
  phase3Bar: HTMLProgressElement;
  frameCounter: HTMLElement;
  lossCurveCanvas: HTMLCanvasElement;
  viewerCanvas: HTMLCanvasElement;
  timelineScrubber: HTMLInputElement;
}

// ---------------------------------------------------------------------------
// Offline UI state
// ---------------------------------------------------------------------------

let smplLoader: SMPLLoaderUI | null = null;
let smplModel: SMPLModelData | null = null;
let videoFile: File | null = null;
let isProcessing = false;

// Loss history for the curve canvas
const lossHistory: number[] = [];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function updateProcessButton(btn: HTMLButtonElement) {
  const ready = smplModel !== null && videoFile !== null && !isProcessing;
  btn.disabled = !ready;
  btn.style.opacity = ready ? '1' : '0.4';
  btn.style.cursor = ready ? 'pointer' : 'not-allowed';
}

function drawLossCurve(canvas: HTMLCanvasElement, losses: number[]) {
  const ctx = canvas.getContext('2d');
  if (!ctx || losses.length < 2) return;

  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const min = Math.min(...losses);
  const max = Math.max(...losses);
  const range = max - min || 1;

  ctx.strokeStyle = '#4ade80';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < losses.length; i++) {
    const x = (i / (losses.length - 1)) * w;
    const y = h - ((losses[i]! - min) / range) * (h - 8) - 4;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = '#405060';
  ctx.font = '10px monospace';
  ctx.textAlign = 'left';
  ctx.fillText(`${min.toFixed(4)}`, 2, h - 2);
  ctx.textAlign = 'right';
  ctx.fillText(`${max.toFixed(4)}`, w - 2, 12);
}

// ---------------------------------------------------------------------------
// Video drop zone setup
// ---------------------------------------------------------------------------

function setupVideoDropZone(
  zone: HTMLElement,
  input: HTMLInputElement,
  processBtn: HTMLButtonElement,
  labelEl: HTMLElement,
) {
  function onFile(file: File) {
    videoFile = file;
    labelEl.textContent = `Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`;
    labelEl.style.color = '#7ab8e0';
    zone.style.borderColor = '#4a9fc2';
    updateProcessButton(processBtn);
  }

  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.style.borderColor = '#4a9fc2';
    zone.style.background = '#0d1a24';
  });
  zone.addEventListener('dragleave', () => {
    if (!videoFile) {
      zone.style.borderColor = '#333';
      zone.style.background = '#0d0d14';
    }
  });
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files[0];
    if (file && (file.type.startsWith('video/') || file.name.endsWith('.mp4') || file.name.endsWith('.webm') || file.name.endsWith('.mov'))) {
      onFile(file);
    } else if (file) {
      labelEl.textContent = 'Not a video file — try .mp4, .webm, .mov';
      labelEl.style.color = '#f87171';
    }
  });
  zone.addEventListener('click', () => input.click());
  input.addEventListener('change', () => {
    const file = input.files?.[0];
    if (file) onFile(file);
  });
}

// ---------------------------------------------------------------------------
// Real batch pipeline
// ---------------------------------------------------------------------------

let _abortController: AbortController | null = null;

async function runOfflinePipeline(els: OfflineUIElements) {
  if (!smplModel || !videoFile) return;
  isProcessing = true;
  lossHistory.length = 0;
  _abortController = new AbortController();

  updateProcessButton(els.processBtn);
  els.progressSection.style.display = 'block';
  els.phase1Bar.value = 0;
  els.phase2Bar.value = 0;
  els.phase3Bar.value = 0;
  els.frameCounter.textContent = 'Initializing WebGPU…';

  // Get (or reuse) WebGPU device
  let device: GPUDevice;
  try {
    device = await getGpuDevice();
  } catch (e) {
    els.frameCounter.textContent = `WebGPU unavailable: ${e instanceof Error ? e.message : String(e)}`;
    isProcessing = false;
    updateProcessButton(els.processBtn);
    return;
  }

  // Create object URL from the File so BatchPipeline can fetch it
  const videoUrl = URL.createObjectURL(videoFile);

  function onProgress(p: BatchProgress) {
    const frac = p.totalFrames > 0 ? p.frameIndex / p.totalFrames : 0;
    const pct = Math.round(frac * 100);

    if (p.phase === 'extract') {
      els.phase1Bar.value = pct;
      els.frameCounter.textContent = `Extracting frames — ${p.frameIndex} / ${p.totalFrames}`;
    } else if (p.phase === 'optimize') {
      els.phase2Bar.value = 100; // preprocessing done
      const iterFrac = p.iterIndex != null && p.iterIndex > 0
        ? (p.iterIndex / 700) * 100 : 0;
      els.phase3Bar.value = Math.round(frac * 100 * 0.9 + iterFrac * 0.1);
      if (p.loss != null) {
        lossHistory.push(p.loss);
        drawLossCurve(els.lossCurveCanvas, lossHistory);
        els.frameCounter.textContent =
          `Optimizing frame ${p.frameIndex + 1}/${p.totalFrames}` +
          ` — iter ${p.iterIndex ?? 0}/700  loss=${p.loss.toFixed(4)}`;
      }
    } else {
      // segment / mast3r / focal / romp / pose2d / contact / interpolate
      els.phase2Bar.value = pct;
      els.frameCounter.textContent =
        `${p.phase.charAt(0).toUpperCase() + p.phase.slice(1)}` +
        ` — frame ${p.frameIndex + 1} / ${p.totalFrames}`;
    }
  }

  try {
    const pipeline = new BatchPipeline(device, onProgress);
    pipeline.setSmplModel(smplModel);
    const result = await pipeline.process(videoUrl, _abortController.signal);

    els.phase1Bar.value = 100;
    els.phase2Bar.value = 100;
    els.phase3Bar.value = 100;
    els.timelineScrubber.max = String(result.frameCount - 1);
    els.timelineScrubber.value = '0';
    els.timelineScrubber.disabled = false;
    els.frameCounter.textContent =
      `Done — ${result.frameCount} frames in ${(result.processingTimeMs / 1000).toFixed(1)}s`;
  } catch (e) {
    if ((e as Error).name === 'AbortError') {
      els.frameCounter.textContent = 'Cancelled.';
    } else {
      console.error('[offline]', e);
      els.frameCounter.textContent = `Error: ${e instanceof Error ? e.message : String(e)}`;
    }
  } finally {
    URL.revokeObjectURL(videoUrl);
    isProcessing = false;
    updateProcessButton(els.processBtn);
  }
}

// ---------------------------------------------------------------------------
// Mount function — called by the tab switcher in index.html
// ---------------------------------------------------------------------------

export function mountOfflineUI(container: HTMLElement) {
  container.innerHTML = `
    <div style="display:flex;flex-direction:column;gap:16px;padding:16px;height:100%;overflow-y:auto;
                font-family:system-ui,sans-serif;color:#e0e6f0;box-sizing:border-box;">

      <!-- Row 1: SMPL loader + Video drop -->
      <div style="display:flex;gap:16px;flex-wrap:wrap;">

        <!-- SMPL upload -->
        <div style="flex:1;min-width:280px;">
          <div style="font-size:0.8rem;font-weight:600;color:#607080;
                      text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">
            SMPL Model
          </div>
          <div id="offline-smpl-container" style="
            background:#111;border:1px solid #222;border-radius:8px;padding:12px;
            min-height:100px;display:flex;align-items:center;justify-content:center;
          "></div>
        </div>

        <!-- Video drop zone -->
        <div style="flex:1;min-width:280px;">
          <div style="font-size:0.8rem;font-weight:600;color:#607080;
                      text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">
            Input Video
          </div>
          <div id="offline-video-zone" style="
            border:2px dashed #333;border-radius:8px;padding:28px 20px;
            text-align:center;cursor:pointer;background:#0d0d14;
            transition:border-color 0.2s,background 0.2s;
          ">
            <div style="font-size:2rem;opacity:0.5;margin-bottom:6px;">&#127910;</div>
            <div style="font-size:0.85rem;color:#c0d0e0;">Drop video file here or click to browse</div>
            <div id="offline-video-label" style="font-size:0.75rem;color:#405060;margin-top:6px;">
              Supported: .mp4, .webm, .mov
            </div>
            <input type="file" id="offline-video-input" accept="video/*" style="display:none" />
          </div>
        </div>
      </div>

      <!-- Process button -->
      <div>
        <button id="offline-process-btn" disabled style="
          padding:10px 28px;background:#1a3a1a;color:#4ade80;
          border:1px solid #2a5a2a;border-radius:6px;font-size:0.9rem;
          font-weight:600;cursor:not-allowed;opacity:0.4;transition:opacity 0.2s;
        ">
          &#9654; Process Video
        </button>
      </div>

      <!-- Progress section (hidden until processing) -->
      <div id="offline-progress" style="display:none;background:#0d1422;border:1px solid #1a2a3a;
           border-radius:8px;padding:16px;font-size:0.8rem;">
        <div style="color:#607080;margin-bottom:8px;font-weight:600;text-transform:uppercase;
                    letter-spacing:0.05em;">Progress</div>
        <div style="display:flex;flex-direction:column;gap:8px;">
          <div>
            <div style="color:#90a0b0;margin-bottom:3px;">Phase 1 — Frame Extraction</div>
            <progress id="offline-phase1" value="0" max="100"
              style="width:100%;height:6px;accent-color:#4a9fc2;"></progress>
          </div>
          <div>
            <div style="color:#90a0b0;margin-bottom:3px;">Phase 2 — Depth + HMR Inference</div>
            <progress id="offline-phase2" value="0" max="100"
              style="width:100%;height:6px;accent-color:#a78bfa;"></progress>
          </div>
          <div>
            <div style="color:#90a0b0;margin-bottom:3px;">Phase 3 — JOSH Solver</div>
            <progress id="offline-phase3" value="0" max="100"
              style="width:100%;height:6px;accent-color:#4ade80;"></progress>
          </div>
        </div>
        <div id="offline-frame-counter" style="margin-top:8px;color:#507090;font-family:monospace;"></div>
        <div style="margin-top:8px;">
          <div style="color:#406050;font-size:0.75rem;margin-bottom:4px;">Loss curve</div>
          <canvas id="offline-loss-canvas" width="400" height="80"
            style="width:100%;height:80px;background:#0a0a10;border-radius:4px;"></canvas>
        </div>
      </div>

      <!-- 3D viewer -->
      <div style="flex:1;min-height:300px;background:#0a0a10;border:1px solid #1a2030;
           border-radius:8px;overflow:hidden;position:relative;">
        <div style="position:absolute;top:8px;left:12px;font-size:0.75rem;color:#304050;
                    font-weight:600;text-transform:uppercase;letter-spacing:0.05em;z-index:1;">
          3D Viewer
        </div>
        <canvas id="offline-viewer-canvas"
          style="width:100%;height:100%;display:block;"></canvas>
        <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
                    pointer-events:none;" id="offline-viewer-placeholder">
          <div style="text-align:center;color:#2a3a4a;">
            <div style="font-size:2rem;opacity:0.3;margin-bottom:6px;">&#9651;</div>
            <div style="font-size:0.8rem;">Results appear here after processing</div>
          </div>
        </div>
      </div>

      <!-- Timeline scrubber -->
      <div style="padding:0 4px;">
        <input type="range" id="offline-timeline" min="0" max="100" value="0" disabled
          style="width:100%;accent-color:#4a9fc2;cursor:not-allowed;" />
        <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#304050;">
          <span>0:00</span>
          <span id="offline-timeline-label">Timeline</span>
          <span>End</span>
        </div>
      </div>
    </div>
  `;

  // Wire up elements
  const smplContainer = container.querySelector('#offline-smpl-container') as HTMLElement;
  const videoZone = container.querySelector('#offline-video-zone') as HTMLElement;
  const videoInput = container.querySelector('#offline-video-input') as HTMLInputElement;
  const processBtn = container.querySelector('#offline-process-btn') as HTMLButtonElement;
  const progressSection = container.querySelector('#offline-progress') as HTMLElement;
  const phase1Bar = container.querySelector('#offline-phase1') as HTMLProgressElement;
  const phase2Bar = container.querySelector('#offline-phase2') as HTMLProgressElement;
  const phase3Bar = container.querySelector('#offline-phase3') as HTMLProgressElement;
  const frameCounter = container.querySelector('#offline-frame-counter') as HTMLElement;
  const lossCurveCanvas = container.querySelector('#offline-loss-canvas') as HTMLCanvasElement;
  const viewerCanvas = container.querySelector('#offline-viewer-canvas') as HTMLCanvasElement;
  const timelineScrubber = container.querySelector('#offline-timeline') as HTMLInputElement;
  const videoLabel = container.querySelector('#offline-video-label') as HTMLElement;

  const els: OfflineUIElements = {
    smplContainer,
    videoDropZone: videoZone,
    videoFileInput: videoInput,
    processBtn,
    progressSection,
    phase1Bar,
    phase2Bar,
    phase3Bar,
    frameCounter,
    lossCurveCanvas,
    viewerCanvas,
    timelineScrubber,
  };

  // SMPL loader
  smplLoader?.dispose();
  smplLoader = new SMPLLoaderUI({
    container: smplContainer,
    onLoad: (data) => {
      smplModel = data;
      updateProcessButton(processBtn);
    },
    onError: () => {
      smplModel = null;
      updateProcessButton(processBtn);
    },
  });

  // Video drop zone
  setupVideoDropZone(videoZone, videoInput, processBtn, videoLabel);

  // Process button
  processBtn.addEventListener('click', () => {
    if (!isProcessing && smplModel && videoFile) {
      runOfflinePipeline(els);
    }
  });

  // Timeline scrubber label
  timelineScrubber.addEventListener('input', () => {
    const f = parseInt(timelineScrubber.value, 10);
    const label = container.querySelector('#offline-timeline-label') as HTMLElement;
    label.textContent = `Frame ${f}`;
  });
}
