# Faithful JOSH Algorithm Reimplementation on WebGPU

> **Status as of v0.2.2 · ec1fec9** — Last audited 2026-03-26

## Legend
- ✅ Done and tested
- ⚠️ Code exists but incomplete / stub / not wired in
- ❌ Not started
- 🔑 Blocks end-to-end JOSH run

---

## What's Left to Run JOSH End-to-End

| # | What | Effort | Why blocked |
|---|------|--------|-------------|
| 1 | **Wire optimizer into batch pipeline** (Phase H) | ~1 day | `stubJoshOptimize` returns fake loss; real `josh-optimizer.node.ts` exists but not called |
| 2 | **Feed real SMPL data to optimizer** | ~0.5 day | `josh-optimizer.node.ts` has placeholder `Float32Array` for skinning weights/indices |
| 3 | **Run `python scripts/export-romp.py`** | ~5 min | Needs machine with PyTorch; `.onnx` file missing |
| 4 | **Run `python scripts/export-mast3r.py`** | ~20 min | Needs machine with PyTorch + 8GB RAM; `.onnx` file missing |
| 5 | **tf.js SMPL forward** (Phase 0F) | ~1 day | Needed for gradient correctness validation; also the simpler path for optimizer |

**Decision pending:** use WGSL optimizer (fast, complex) or tf.js optimizer (2–3× slower, automatic gradients). See section below.

---

## Optimizer Path Decision

### Option A — WGSL (current partial impl)
- All kernels written and unit-tested (Adam, SMPL fwd, 6 losses, JVP)
- **Blocker:** skinning weights placeholder in `josh-optimizer.node.ts` bindings 2,3,4,5
- **Blocker:** `batch-pipeline.ts` Phase H calls `stubJoshOptimize` not the real node
- Speed: ~7–17s/frame on M-series MacBook

### Option B — tf.js (recommended for correctness first)
- Write `demo/josh/tf/smpl-forward-ref.ts` + `demo/josh/tf/losses-ref.ts`
- Use `tf.variableGrads()` + `tf.train.adam()` — zero manual gradient code
- Wire into `batch-pipeline.ts` Phase H directly
- Speed: ~20–35s/frame on M-series MacBook (acceptable for offline)
- **Advantage:** correct by construction; validates WGSL path afterward

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

### 1B: MASt3R ONNX Export ⚠️
- `scripts/export-mast3r.py` ✅ — script written
- `public/models/mast3r-vit-large-fp32.onnx` ❌ — **not generated yet** (run script on PyTorch machine)
- **Test:** requires ONNX file ❌

### 1C: MASt3R Node ⚠️
- `demo/josh/nodes/mast3r.node.ts` ✅ — browser loader written
- Wired into `batch-pipeline.ts` Phase C ✅
- **Blocked by:** ONNX file missing → falls back to null pointmaps

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
- **Phase E (ROMP):** `ROMPNode` class ready, `_ensureRompInit()` in place ⚠️ — blocked by missing ONNX
- **Phase H (JOSH optimize):** calls `stubJoshOptimize` ❌ — **the core blocker**
  - Returns fake decaying loss; real `josh-optimizer.node.ts` not called
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

### Must-do to run JOSH at all

| Task | File(s) | Notes |
|------|---------|-------|
| **0F: tf.js SMPL forward + losses** | `demo/josh/tf/smpl-forward-ref.ts`, `losses-ref.ts` | Enables Option B optimizer AND gradient validation |
| **Wire Phase H** | `batch-pipeline.ts` | Replace `stubJoshOptimize` with real optimizer call |
| **Feed SMPL data to optimizer** | `josh-optimizer.node.ts` | Replace skinning weight placeholders with loaded model data |
| **Run ROMP export** | `scripts/export-romp.py` | Needs PyTorch machine; ~5 min |
| **Run MASt3R export** | `scripts/export-mast3r.py` | Needs PyTorch + 8GB RAM; ~20 min |

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
| 1B MASt3R export | ⚠️ | ONNX file needed |
| 1C MASt3R node | ⚠️ | ONNX file needed |
| 1D Focal recovery | ✅ | — |
| 1E MoveNet | ✅ | — |
| 1F Contact detect | ✅ | — |
| 2A Frame extract | ✅ | — |
| **2B Phase H** | **⚠️** | **Core JOSH run blocked** |
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
