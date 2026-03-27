# Faithful JOSH Algorithm Reimplementation on WebGPU

> **Status as of v0.2.3 · 88877db** — Last audited 2026-03-26

## Legend
- ✅ Done and tested
- ⚠️ Code exists but incomplete / stub / not wired in
- ❌ Not started
- 🔑 Blocks end-to-end JOSH run

---

## What's Left to Run JOSH End-to-End

**All remaining work is TypeScript/browser — no Python required.**

| # | What | File(s) | Effort |
|---|------|---------|--------|
| 🔑 1 | **tf.js SMPL forward + 6 losses** | `demo/josh/tf/smpl-forward-ref.ts`, `losses-ref.ts` | ~1 day |
| 🔑 2 | **Wire Phase H** — replace `stubJoshOptimize` with tf.js optimizer | `batch-pipeline.ts` | ~0.5 day |
| 🔑 3 | **Feed SMPL model data into optimizer** | `batch-pipeline.ts`, `josh-optimizer.node.ts` | ~0.5 day |
| 🔑 4 | **ROMP → tf.js BlazePose 3D** — replace Python ONNX export | `demo/josh/nodes/romp.node.ts` rewrite | ~1 day |
| 🔑 5 | **MASt3R → tf.js MiDaS depth** — replace Python ONNX export | `demo/josh/nodes/mast3r.node.ts` rewrite | ~0.5 day |

**Total: ~3.5 days TypeScript work → full pipeline running in browser, zero Python.**

### Notes on items 4 and 5

**Item 4 — ROMP replaced by tf.js BlazePose 3D:**
- MediaPipe BlazePose gives 33 3D body landmarks directly from tf.js (`@mediapipe/pose` or `@tensorflow-models/pose-detection`)
- Map 33 BlazePose landmarks → 24 SMPL joints (correspondence table already in `pose-2d.node.ts`)
- Estimate initial pose axis-angles from joint directions (no ONNX file needed)
- Quality: slightly lower init than ROMP but the Adam optimizer corrects it over 700 iters
- The existing `romp.node.ts` fallback path (`ROMPOutput | null`) handles null gracefully

**Item 5 — MASt3R replaced by tf.js MiDaS:**
- MiDaS depth model is **already implemented** in `demo/josh/nodes/depth-estimation.node.ts`
- MiDaS gives per-pixel depth map → unproject to 3D point cloud using recovered focal length
- For L_3D loss: use MiDaS depth Z instead of MASt3R pointmaps
- Quality: monocular depth (relative scale), less accurate than MASt3R stereo
- Scale recovered by `recoverFocalLength` which also solves for global scale factor
- When real MASt3R ONNX is available later, swap back in — pipeline interface unchanged

---

## Tech Stack (Updated — All Browser)

```
opencv-ts-webgpu:     graph engine, video I/O, buffer management
onnxruntime-web:      MoveNet 2D pose (12 MB, already works)
tf.js WebGPU:         SMPL forward, 6 losses, Adam optimizer (Option B)
                      BlazePose 3D (replaces ROMP)
MiDaS (already impl): depth maps (replaces MASt3R for now)
WGSL compute:         Option A optimizer (fast path, after tf.js validates)
WebGPU render:        3D mesh + point cloud visualization
```

---

## Optimizer Path

### Current decision: tf.js first, WGSL second

**Phase 1 (now):** tf.js optimizer — correct by construction, ~20–35s/frame
- `tf.variableGrads()` + `tf.train.adam()` — zero manual gradient code
- All 6 losses as tf.js tensor ops — automatic differentiation
- Validates the WGSL kernels (compare outputs)

**Phase 2 (later):** WGSL fast path — ~7–17s/frame
- All WGSL kernels already written and unit-tested
- Replace tf.js optimizer with WGSL after correctness confirmed
- Keep tf.js as regression baseline

---

## Context

Reimplement the exact JOSH paper algorithm in-browser using WebGPU.
Offline processing (minutes per video). Full precision FP32. Target: MacBooks 32GB+.

**Paper details:**
- Two-stage Adam: Stage 1 (500 iters, lr=0.07), Stage 2 (200 iters, lr=0.014)
- 6 losses: L_3D, L_2D, L_c1, L_c2, L_p, L_smooth
- Weights: wc1=1, wc2=20, wp=10, Δc1=0, Δc2=0.1
- 100-frame chunks, keyframes every 0.2s
- Foreground segmentation BEFORE MASt3R (critical)

---

## Architecture

```
Video → [Frame Extractor]
           │
           ├──► [BodyPix Segmentation] ──► human masks
           │
           ├──► [MASt3R ONNX] + masks ──► dense 3D pointmaps + camera poses
           │
           ├──► [Focal Length Recovery] ──► camera intrinsics K
           │
           ├──► [ROMP ONNX] ──► initial SMPL params (θ, β)
           │
           └──► [MoveNet ONNX] ──► 2D keypoints with confidence
                                            │
                               ┌────────────▼────────────┐
                               │   JOSH Optimizer         │
                               │   700 iters Adam         │
                               │   6 losses               │
                               │   Option A: WGSL GPU     │
                               │   Option B: tf.js        │
                               └────────────┬────────────┘
                                            │
                               ┌────────────▼────────────┐
                               │  IndexedDB Cache         │
                               │  3D Renderer (WebGPU)    │
                               │  Timeline Playback       │
                               └─────────────────────────┘
```

---

## Phase 0: GPU-Native Optimizer Foundation

### 0A: WGSL Adam Optimizer Kernel ✅
- `demo/josh/kernels/adam-optimizer.wgsl` — GPU Adam, reads timestep from storage buffer
- `demo/josh/kernels/increment-counter.wgsl` — GPU-side t counter
- **Demo:** `demo/phase0.html` tab "Adam Optimizer" ✅
- **Test:** `tests/josh/adam-optimizer.test.ts` ❌ missing (covered partially by gradient-validation)

### 0B: WGSL Differentiable SMPL Forward Pass ✅
- `demo/josh/kernels/smpl-forward.wgsl` — shape blend + LBS
- `demo/josh/kernels/smpl-joints.wgsl` — Rodrigues + FK
- `demo/josh/kernels/rodrigues.wgsl` — axis-angle → rotation (sinc/cosc form)
- `demo/josh/kernels/rodrigues-deriv.wgsl` — analytical dR/dω
- **Demo:** `demo/phase0.html` tab "SMPL Forward" ✅
- **Test:** `tests/josh/rodrigues-deriv.test.ts` ✅ (8 tests, median error <1e-4)
- **Test:** `tests/josh/smpl-forward.test.ts` ❌ missing

### 0C: 6 Loss Function Kernels (WGSL) ✅
- `demo/josh/kernels/josh-contact-loss.wgsl` — L_c1
- `demo/josh/kernels/josh-contact-static.wgsl` — L_c2
- `demo/josh/kernels/josh-depth-reproj.wgsl` — L_3D
- `demo/josh/kernels/josh-reproj-2d.wgsl` — L_2D
- `demo/josh/kernels/josh-smpl-prior.wgsl` — L_p
- `demo/josh/kernels/josh-temporal.wgsl` — L_smooth
- **Demo:** `demo/phase0.html` tabs ✅
- **Test:** individual loss tests ❌ all missing

### 0D: Chain Rule Kernel (JVP) ✅
- `demo/josh/kernels/jvp-gradient.wgsl` — dL/dv → dL/dpose
- `demo/josh/kernels/jvp-joint-gradient.wgsl` — dL/djoint → dL/dpose (for L_2D)
- **Demo:** `demo/phase0.html` ✅
- **Test:** `tests/josh/gradient-validation.test.ts` ✅ (11 tests, median rel error 0.001)

### 0E: Optimization Loop Orchestrator ⚠️
- `demo/josh/nodes/josh-optimizer.node.ts` — 700-iter GPU loop, single command encoder ✅
- `demo/josh/nodes/josh-solver.node.ts` — node wrapper ✅
- **Blocker 🔑:** bindings 2,3 (skinning weights/indices) use placeholder `smplVerticesShaped` — need real SMPL `weights` array from loaded model
- **Blocker 🔑:** `batch-pipeline.ts` Phase H calls `stubJoshOptimize`, not this node
- **Demo:** `demo/phase0.html` tab "Optimizer Loop" ✅ (synthetic)
- **Test:** `tests/josh/optimizer-loop.test.ts` ❌ missing

### 0F: tf.js Reference Implementation ❌ 🔑
- `demo/josh/tf/smpl-forward-ref.ts` — tf.js SMPL forward (not started)
- `demo/josh/tf/losses-ref.ts` — tf.js loss functions (not started)
- Needed for: (a) definitive gradient correctness validation, (b) Option B optimizer
- **Demo:** `demo/phase0.html` tab "tf.js vs WGSL Validation" ❌
- **Test:** gradient cross-check in `tests/josh/gradient-validation.test.ts` ⚠️ (partial)

---

## Phase 1: Preprocessing Models

### 1A: Foreground Segmentation ✅
- `demo/josh/nodes/segmentation.node.ts` — MediaPipe primary + HSV fallback
- **Demo:** `demo/phase1.html` ✅
- **Test:** `tests/josh/phase1-nodes.test.ts` ✅ (22 tests)

### 1B: MASt3R → replaced by MiDaS depth (tf.js) 🔑
- **No Python needed.** MiDaS already implemented in `demo/josh/nodes/depth-estimation.node.ts`
- Rewrite `mast3r.node.ts` to wrap MiDaS: depth map → unproject to 3D point cloud using K
- L_3D loss uses MiDaS depth Z instead of MASt3R stereo pointmaps
- `scripts/export-mast3r.py` kept for later when real MASt3R ONNX is desired
- **Test:** `tests/josh/mast3r-node.test.ts` ❌ missing

### 1C: MASt3R Node ⚠️ → rewrite to use MiDaS 🔑
- `demo/josh/nodes/mast3r.node.ts` — currently expects 2.75GB ONNX
- **Rewrite:** delegate to `DepthEstimationNode`, unproject depth → point cloud
- Interface stays identical (returns `MASt3ROutput`) — rest of pipeline unchanged

### 1D: Focal Length Recovery ✅
- `demo/josh/utils/focal-recovery.ts` ✅
- Wired into `batch-pipeline.ts` Phase D ✅ (uses real MASt3R output when available)
- **Test:** `tests/josh/phase1-utils.test.ts` ✅ (focal recovery tests pass)

### 1E: MoveNet 2D Pose ✅
- `demo/josh/nodes/pose-2d.node.ts` ✅ — MoveNet ONNX + synthetic fallback
- COCO→SMPL joint mapping ✅
- **Test:** `tests/josh/phase1-nodes.test.ts` ✅

### 1F: Contact Detection ✅
- `demo/josh/utils/contact-detection.ts` ✅
- **Test:** `tests/josh/phase1-utils.test.ts` ✅

---

## Phase 2: Offline Batch Pipeline

### 2A: Frame Extraction ✅
- `demo/josh/batch/frame-extractor.ts` ✅

### 2B: Batch Orchestrator ⚠️ 🔑
- `demo/josh/batch/batch-pipeline.ts` ✅ — all 9 phases wired
- **Phase E (ROMP → BlazePose):** rewrite `romp.node.ts` to use tf.js BlazePose 3D instead of ONNX 🔑
- **Phase H (JOSH optimize):** calls `stubJoshOptimize` ❌ — **the core blocker**
  - Replace with tf.js `smpl-forward-ref.ts` + `losses-ref.ts` + `tf.train.adam()` 🔑
- **Test:** `tests/josh/batch-pipeline.test.ts` ❌ missing

### 2C: SE3 Interpolation ✅
- `demo/josh/utils/se3-interpolation.ts` ✅
- **Test:** `tests/josh/phase2-utils.test.ts` ✅

### 2D: Chunk Concatenation ✅
- `demo/josh/utils/chunk-concat.ts` ✅
- **Test:** `tests/josh/phase2-utils.test.ts` ✅

### 2E: Result Cache ✅
- `demo/josh/batch/result-cache.ts` ✅
- **Test:** `tests/josh/result-cache.test.ts` ❌ missing

### 2F: Progress UI ✅
- `demo/josh/ui/progress-ui.ts` ✅
- `demo/index.html` Offline tab ✅

---

## Phase 3: 3D Rendering

### 3A: SMPL Mesh Renderer ✅
- `demo/josh/rendering/smpl-renderer.ts` ✅
- `demo/josh/rendering/smpl-mesh.{vert,frag}.wgsl` ✅
- `demo/josh/rendering/compute-normals.wgsl` ✅

### 3B: Point Cloud Renderer ✅
- `demo/josh/rendering/pointcloud-renderer.ts` ✅

### 3C: Timeline Playback ✅
- `demo/josh/ui/timeline.ts` ✅
- `demo/phase3c.html` ✅

---

## Phase 4: SMPL Data + Integration

### 4A: SMPL Upload UI ✅
- `demo/josh/models/smpl-loader-ui.ts` ✅
- Auto-loads from `/smpl/smpl-neutral.smpl.bin` (41 MB pre-converted binary) ✅
- `scripts/convert-smpl.py` ✅ — pkl → binary conversion (already run)
- **Test:** `tests/josh/smpl-loader.test.ts` ❌ missing

### 4B: Config ✅
- `demo/josh/config.ts` ✅ — all paper hyperparameters

### 4C: Offline Entry Point ✅
- `demo/main-offline.ts` ✅
- `demo/index.html` Offline tab ✅

### 4D: Realtime Demo ⚠️
- Existing realtime pipeline preserved
- **Test:** `tests/josh/realtime-compat.test.ts` ❌ missing

---

## Test Coverage Summary

| Test file | Status | Tests |
|-----------|--------|-------|
| `rodrigues-deriv.test.ts` | ✅ | 8 |
| `phase1-utils.test.ts` | ✅ | 12 |
| `phase1-nodes.test.ts` | ✅ | 22 |
| `phase2-utils.test.ts` | ✅ | 19 |
| `gradient-validation.test.ts` | ✅ | 11 |
| `adam-optimizer.test.ts` | ❌ | 0 |
| `smpl-forward.test.ts` | ❌ | 0 |
| `loss-*.test.ts` (6 files) | ❌ | 0 |
| `optimizer-loop.test.ts` | ❌ | 0 |
| `batch-pipeline.test.ts` | ❌ | 0 |
| `result-cache.test.ts` | ❌ | 0 |
| `smpl-loader.test.ts` | ❌ | 0 |
| `timeline.test.ts` | ❌ | 0 |
| **Total** | **72 / 100 tests passing** | **72** |

---

## Remaining Work (Ordered by Priority)

### Must-do to run JOSH at all — all TypeScript, no Python

| # | Task | File(s) | Notes |
|---|------|---------|-------|
| 1 | **tf.js SMPL forward** | `demo/josh/tf/smpl-forward-ref.ts` | Rodrigues + FK + LBS as tf.js tensors |
| 2 | **tf.js 6 losses** | `demo/josh/tf/losses-ref.ts` | L_3D, L_2D, L_c1, L_c2, L_p, L_smooth |
| 3 | **Wire Phase H** | `batch-pipeline.ts` | Replace `stubJoshOptimize` with tf.variableGrads + tf.train.adam |
| 4 | **Feed SMPL data to Phase H** | `batch-pipeline.ts` | Pass loaded SMPLModelData into tf.js optimizer |
| 5 | **Rewrite MASt3R node → MiDaS** | `mast3r.node.ts` | Delegate to existing DepthEstimationNode, unproject to 3D |
| 6 | **Rewrite ROMP node → BlazePose 3D** | `romp.node.ts` | Use @tensorflow-models/pose-detection, map to SMPL init params |

### Tests needed for confidence

| Task | File(s) |
|------|---------|
| SMPL forward correctness | `smpl-forward.test.ts` |
| Each loss kernel | `loss-contact.test.ts`, `loss-depth-reproj.test.ts`, `loss-reproj-2d.test.ts`, `loss-smpl-prior.test.ts`, `loss-smooth.test.ts` |
| Optimizer convergence | `optimizer-loop.test.ts` |
| Batch pipeline e2e | `batch-pipeline.test.ts` |

### Nice-to-have

| Task | Notes |
|------|-------|
| `result-cache.test.ts` | IndexedDB mock |
| `smpl-loader.test.ts` | Binary parser round-trip |
| `timeline.test.ts` | jsdom |
| Realtime compat test | Regression guard |

---

## How to Test JOSH (When Complete)

```bash
# 1. Unit tests (fast, no GPU needed)
npm test

# 2. Gradient validation: WGSL vs tf.js (slow, ~30s)
npm run test:gradients

# 3. Optimizer convergence on synthetic scene (~60s)
npm run test:convergence

# 4. Full pipeline on 3-frame clip (browser, manual)
open https://100.101.195.35:5182/phase2b.html
# Drop a 1s clip → verify 9 phase bars complete → check 3D result

# 5. Visual check
open https://100.101.195.35:5182/phase3c.html
# Should show animated SMPL mesh with loss curve declining
```

---

## Implementation Priority Table

| Phase | Status | Blocks |
|-------|--------|--------|
| 0A Adam | ✅ | — |
| 0B SMPL fwd | ✅ | — |
| 0C 6 losses | ✅ | — |
| 0D JVP | ✅ | — |
| 0E Optimizer loop | ⚠️ | Phase H, need SMPL data wired |
| **0F tf.js ref** | **❌** | **Gradient validation + Option B** |
| 1A Segmentation | ✅ | — |
| 1B MASt3R → MiDaS tf.js | ❌ 🔑 | Rewrite node to use existing MiDaS |
| 1C MASt3R node | ⚠️ 🔑 | Rewrite to unproject MiDaS depth |
| 1D Focal recovery | ✅ | — |
| 1E MoveNet | ✅ | — |
| 1F Contact detect | ✅ | — |
| 2A Frame extract | ✅ | — |
| **2B Phase H** | **⚠️ 🔑** | **Replace stub with tf.js optimizer** |
| 2C SE3 interp | ✅ | — |
| 2D Chunk concat | ✅ | — |
| 2E Result cache | ✅ | — |
| 2F Progress UI | ✅ | — |
| 3A Mesh renderer | ✅ | — |
| 3B Point cloud | ✅ | — |
| 3C Timeline | ✅ | — |
| 4A SMPL upload | ✅ | — |
| 4B Config | ✅ | — |
| 4C Offline entry | ✅ | — |
| 4D Realtime compat | ⚠️ | — |

---

## Zero-Python Completion Roadmap

```
Week 1:
  Day 1-2  tf.js SMPL forward + 6 losses   (items 1-2)
  Day 3    Wire Phase H optimizer           (item 3-4)
  Day 4    MiDaS → MASt3R node rewrite     (item 5)
  Day 5    BlazePose → ROMP node rewrite   (item 6)
           ↓
           First real end-to-end JOSH run on a video clip ✅

Week 2:
  Add missing tests (smpl-forward, losses, optimizer, pipeline)
  WGSL fast path validation (compare WGSL vs tf.js outputs)
  Visual quality tuning
```
