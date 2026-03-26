# Faithful JOSH Algorithm Reimplementation on WebGPU

## Context

Reimplement the exact JOSH paper algorithm in-browser using WebGPU. Offline processing (minutes per video). Full precision FP32. Target: MacBooks 32GB+.

**Tech stack:**
```
opencv-ts-webgpu:   graph engine, video I/O, buffer management
onnxruntime-web:    model inference (MASt3R, ROMP, MoveNet, BodyPix) via WebGPU EP
WGSL compute:       SMPL forward pass, losses, Adam optimizer (all on GPU)
tf.js:              gradient validation only (NOT in hot loop — too slow)
WebGPU render:      3D mesh + point cloud visualization
```

**Critical performance insight:** tf.js autograd in the 700-iteration inner loop adds ~5s overhead per frame from JS tensor allocation/GC. Instead: hand-write WGSL kernels for SMPL forward + losses + Adam, keep everything on GPU. Use tf.js only for offline gradient correctness validation.

**Paper details:**
- Two-stage Adam: Stage 1 (500 iters, lr=0.07), Stage 2 (200 iters, lr=0.014)
- 6 losses: L_3D, L_2D, L_c1, L_c2, L_p, L_smooth
- Weights: wc1=1, wc2=20, wp=10, Δc1=0, Δc2=0.1
- 100-frame chunks, keyframes every 0.2s
- Per-frame scale σ^t, depth map Z^t optimization in Stage 2
- Camera intrinsics fixed, extrinsics optimized per frame
- Foreground segmentation BEFORE MASt3R (critical)

**Performance estimate:**
- CUDA (RTX 3090): ~1-2s per frame → 2 min for 10s video
- WebGPU (Apple M2+): ~7-17s per frame → 10 min for 10s video (7-10x slower)
- Main bottleneck: no async GPU readback (mapAsync drains pipeline)
- Solution: keep optimizer on GPU (WGSL Adam), only read back final result

---

## Architecture

```
Video → [Frame Extractor (WebCodecs, 5 FPS)]
           │
           ├──► [BodyPix Segmentation] ──► human masks
           │
           ├──► [MASt3R FP32 (onnxruntime-web)] + masks
           │      └─► dense 3D pointmaps + camera poses
           │
           ├──► [Focal Length Recovery (WASM)]
           │      └─► camera intrinsics K (fixed)
           │
           ├──► [ROMP (onnxruntime-web)]
           │      └─► initial SMPL params (pose θ, shape β)
           │
           └──► [MoveNet 2D (onnxruntime-web)]
                  └─► 2D keypoints with confidence
                                    │
                         ALL ON GPU (no mapAsync in loop)
                    ┌─────────────────────────────────┐
                    │  JOSH Optimizer (WGSL compute)    │
                    │                                   │
                    │  Per iteration (700 total):        │
                    │  1. SMPL forward (WGSL)            │
                    │  2. 6 loss kernels (WGSL)          │
                    │  3. Chain rule gradient (WGSL)     │
                    │  4. Adam update (WGSL)             │
                    │                                   │
                    │  All in one command encoder        │
                    │  Single submit at end              │
                    └─────────────────────────────────┘
                                    │
                              mapAsync once
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
              [IndexedDB Cache]          [3D Renderer (WebGPU)]
              instant replay             mesh + pointcloud + timeline
```

---

## Phase 0: GPU-Native Optimizer Foundation

### 0A: WGSL Adam Optimizer Kernel
- **New file: `demo/josh/kernels/adam-optimizer.wgsl`**
- Per-parameter Adam update entirely on GPU: m, v, bias-corrected step
- Buffers: params[N], gradients[N], m[N], v[N], config (lr, beta1, beta2, eps, t)
- One dispatch of ceil(N/256) workgroups — no CPU involvement
- **Test: `test/josh/adam-optimizer.test.ts`**
  - Compare against known Adam step results (hand-computed for 5 params, 3 iterations)
  - Compare against tf.js `tf.train.adam()` on same inputs — values should match to 1e-5

### 0B: WGSL Differentiable SMPL Forward Pass
- **Rewrite: `demo/josh/kernels/smpl-forward.wgsl`** — shape blend + LBS
- **Rewrite: `demo/josh/kernels/smpl-joints.wgsl`** — Rodrigues + FK
- Must compute BOTH vertices AND per-vertex Jacobians d(v)/d(params)
- Analytical Jacobian of Rodrigues: dR/dw_i (known closed-form formula)
- FK chain rule: d(global_T)/d(θ_j) = parent_T × d(local_T)/d(θ_j)
- Output: vertices[6890,3] + jacobian[6890*3, N_params] (~7.4MB)
- **New file: `demo/josh/kernels/rodrigues-deriv.wgsl`** — dR(w)/dw_i
- **Test: `test/josh/smpl-forward.test.ts`**
  - Load known SMPL test case (T-pose, zero shape → expected vertex positions)
  - Perturb single joint by 0.1 rad → verify FK output changes correctly
  - **Numerical gradient check:** finite diff vs analytical Jacobian, relative error < 1e-4
  - Compare tf.js SMPL forward (Phase 0F) against WGSL output — should match to 1e-5

### 0C: 6 Loss Function Kernels (WGSL)
Each kernel outputs: scalar loss value + per-vertex gradient dL/dv[6890*3]

- **Rewrite: `demo/josh/kernels/josh-contact-loss.wgsl`** — L_c1 contact scale
  - L_c1 = wc1 × Σ max(0, ‖v_contact - scene_nearest‖ - Δc1)²
  - **Test:** known contact vertex at (0,0,1), scene surface at z=1.05
    - Expected loss = wc1 × (0.05 - 0)² = 0.0025 (if Δc1=0)
    - Expected gradient: d/dz = 2 × wc1 × 0.05 = 0.1

- **New file: `demo/josh/kernels/josh-contact-static.wgsl`** — L_c2 contact static
  - L_c2 = wc2 × Σ max(0, ‖v_contact(t) - v_contact(t-1)‖ - Δc2)²
  - **Test:** contact vertex moved 0.2m between frames, Δc2=0.1
    - Expected loss = wc2 × (0.2 - 0.1)² = 20 × 0.01 = 0.2

- **Rewrite: `demo/josh/kernels/josh-depth-reproj.wgsl`** — L_3D correspondence
  - L_3D = Σ ‖π⁻¹(Z^t, K, P^t) - X_matched‖² (3D point matching)
  - **Test:** project known 3D point to depth map, reconstruct → should match

- **New file: `demo/josh/kernels/josh-reproj-2d.wgsl`** — L_2D reprojection
  - L_2D = Σ ‖π(J_3D, K) - j_2D‖² × confidence
  - **Test:** SMPL joint at (0,0,2) with f=300, cx=192 → expected pixel (192,192)
    - Move joint to (0.1,0,2) → expected pixel (207,192), error vs detected

- **New file: `demo/josh/kernels/josh-smpl-prior.wgsl`** — L_p SMPL prior
  - L_p = wp × (‖θ - θ₀‖² + ‖β - β₀‖²)
  - **Test:** θ = θ₀ + 0.1 for all 72 → loss = 10 × 72 × 0.01 = 7.2

- **New file: `demo/josh/kernels/josh-smooth.wgsl`** — L_smooth temporal
  - L_smooth = w_s × (‖P^t - P^{t-1}‖² + ‖θ^t - θ^{t-1}‖²)
  - **Test:** camera moved 0.5m between frames → expected loss contribution

### 0D: Chain Rule Kernel (Jacobian-Vector Product)
- **New file: `demo/josh/kernels/jvp-gradient.wgsl`**
- gradient[j] = Σ_i J[i,j] × dL_dv[i] for all params
- Converts per-vertex loss gradients → per-parameter gradients
- 89 threads (or more for depth map params in Stage 2)
- **Test: `test/josh/chain-rule.test.ts`**
  - Create synthetic Jacobian (identity-like) + known dL/dv
  - Verify output matches matrix multiplication result
  - **Cross-check:** tf.js `tf.grad(loss_through_smpl)` vs WGSL chain rule — should match

### 0E: Optimization Loop Orchestrator
- **Rewrite: `demo/josh/nodes/josh-solver.node.ts`**
- Records all 700 iterations into ONE command encoder (no mapAsync in loop!)
- Stage 1: 500 iters, lr=0.07, w2D=0, optimize σ^t + P^t + θ^t
- Stage 2: 200 iters, lr=0.014, w2D enabled, also optimize Z^t
- Single submit + single mapAsync at end
- Progress: read loss buffer every 50 iterations (separate small readback)
- **Test: `test/josh/optimizer-loop.test.ts`**
  - Synthetic scene: person standing on flat ground at known position
  - Initialize with 10% error in all params
  - After 700 iterations: params should converge within 5% of ground truth
  - Loss should decrease monotonically (plot loss curve)

### 0F: tf.js Reference Implementation (for validation only)
- **New file: `demo/josh/tf/smpl-forward-ref.ts`** — tf.js SMPL forward
- **New file: `demo/josh/tf/losses-ref.ts`** — tf.js loss functions
- NOT used in production pipeline — only for `npm run test` gradient validation
- **Test: `test/josh/gradient-validation.test.ts`**
  - For random params: compute gradient via WGSL chain rule vs tf.js tf.grad()
  - Relative error should be < 1e-4 for all 89 parameters
  - This is the DEFINITIVE correctness test

---

## Phase 1: Preprocessing Models

### 1A: Foreground Segmentation
- **New file: `demo/josh/nodes/segmentation.node.ts`**
- Use BodyPix (tf.js built-in) or MediaPipe Selfie Segmentation ONNX
- Output: binary mask per frame (person = 0, background = 1)
- Applied to MASt3R input to exclude human region
- **Test: `test/josh/segmentation.test.ts`**
  - Run on JOSH demo video frame → mask should cover the person
  - Verify mask has reasonable area (10-60% of frame)
  - Verify MASt3R pointmap filtered by mask excludes human body region

### 1B: MASt3R ONNX Export
- Clone https://github.com/naver/mast3r
- `torch.onnx.export()` with ViT-Large checkpoint
- Input: two images [1,3,H,W] (max dim 512)
- Output: pointmaps [1,H,W,3] × 2, confidence [1,H,W] × 2
- **Output: `mast3r-vit-large-fp32.onnx` (~2.75GB)** in Git LFS
- **Test: `test/josh/mast3r-export.test.ts`**
  - Run ONNX model on two known frames → compare pointmap with PyTorch output
  - Max absolute difference < 1e-3 (FP32 rounding)

### 1C: MASt3R Node
- **New file: `demo/josh/nodes/mast3r.node.ts`**
- Load via cachedFetchModel() (Cache API for 2.75GB)
- Process frame pairs with sliding window-10 graph
- Output: per-frame dense point clouds in world coordinates
- **Test: `test/josh/mast3r-node.test.ts`**
  - Two frames of a static scene → pointmaps should be consistent
  - Camera baseline between frames should match known motion

### 1D: Focal Length Recovery
- **New file: `demo/josh/utils/focal-recovery.ts`**
- Port `recover_focal_shift()` from JOSH Python code
- Least-squares: minimize |f × xy/z - uv| for focal length f and depth shift
- Output: camera intrinsics K (fixed throughout optimization)
- **Test: `test/josh/focal-recovery.test.ts`**
  - Synthetic data: known f=300, generate projected points, recover f
  - Recovered f should match within 2%

### 1E: MoveNet 2D Pose
- **New file: `demo/josh/nodes/pose-2d.node.ts`**
- MoveNet SinglePose Lightning ONNX (~12MB)
- 17 COCO keypoints → map to SMPL joints via correspondence table
- **Test: `test/josh/pose-2d.test.ts`**
  - Run on JOSH demo frame → should detect person with confidence > 0.5
  - Key joints (hips, shoulders, head) should be within 20px of expected

### 1F: Contact Detection (Geometric)
- **New file: `demo/josh/utils/contact-detection.ts`**
- For candidate vertices (feet soles, hands), project to depth map
- If |vertex_z - depth_z| < 5cm → mark as contact
- **Test: `test/josh/contact-detection.test.ts`**
  - Person standing on flat ground → foot vertices should be contacts
  - Raised hand → hand vertices should NOT be contacts

---

## Phase 2: Offline Batch Pipeline

### 2A: Frame Extraction
- **New file: `demo/josh/batch/frame-extractor.ts`**
- WebCodecs VideoDecoder, sample at 5 FPS
- **Test:** extract 10s video → should get ~50 frames, correct resolution

### 2B: Batch Orchestrator
- **New file: `demo/josh/batch/batch-pipeline.ts`**
- Phased processing in 100-frame chunks:
  1. Extract frames → progress
  2. Segment foreground (per frame) → progress
  3. MASt3R on masked frame pairs (window-10) → progress
  4. Focal length recovery → one-time
  5. ROMP on all frames → progress
  6. MoveNet 2D on all frames → progress
  7. Contact detection per frame → progress
  8. JOSH optimization per frame (700 iters) → progress + loss curve
  9. Keyframe interpolation for non-optimized frames → progress
- **Test: `test/josh/batch-pipeline.test.ts`**
  - Process 5-frame synthetic sequence → all phases complete
  - Result cache populated → second run skips computation

### 2C: Keyframe Interpolation
- **New file: `demo/josh/utils/se3-interpolation.ts`**
- SLERP for rotations, linear for translations
- Interpolate camera poses + SMPL params between keyframes
- **Test: `test/josh/se3-interpolation.test.ts`**
  - Two SE3 poses at t=0 and t=1 → interpolated at t=0.5 should be midpoint
  - SLERP of 90° rotation → halfway should be 45°

### 2D: Chunk Concatenation
- **New file: `demo/josh/utils/chunk-concat.ts`**
- Chain coordinate frames: P_chunk2 = P_last_of_chunk1 × P_chunk2_local
- **Test:** two chunks with known transforms → concatenated trajectory should be continuous

### 2E: Result Cache (IndexedDB)
- **New file: `demo/josh/batch/result-cache.ts`**
- Per-frame: params, vertices, joints, camera, losses, depth
- Video content hash as key
- **Test:** store → load → compare → values match exactly

### 2F: Progress UI
- **Update: `demo/index.html`** — offline mode tab
- Phase bars, frame progress, iteration counter, loss curve (canvas chart)
- ETA computation based on rolling average

---

## Phase 3: 3D Rendering

### 3A: SMPL Mesh Renderer
- **New file: `demo/josh/rendering/smpl-renderer.ts`**
- WebGPU render pipeline (vertex + fragment shaders)
- **New file: `demo/josh/rendering/smpl-mesh.vert.wgsl`** — MVP projection
- **New file: `demo/josh/rendering/smpl-mesh.frag.wgsl`** — Phong lighting
- **New file: `demo/josh/rendering/compute-normals.wgsl`** — per-vertex normals
- **Test:** render T-pose → screenshot should show humanoid silhouette

### 3B: Scene Point Cloud Renderer
- **New file: `demo/josh/rendering/pointcloud-renderer.ts`**
- Render MASt3R pointmaps as colored points
- **Test:** render known point cloud → points should project correctly

### 3C: Timeline Playback
- **New file: `demo/josh/ui/timeline.ts`**
- Scrubber, play/pause/step, view toggles
- Loss curve per frame (from cached results)

---

## Phase 4: SMPL Data + Integration

### 4A: SMPL Upload UI
- **New file: `demo/josh/models/smpl-loader-ui.ts`**
- Drag-drop .pkl file, parse pickle format in JS, store in IndexedDB
- Link to smpl.is.tue.mpg.de for user to register (free academic license)
- **Test:** upload known .pkl → verify meanTemplate shape = [6890,3]

### 4B: Config
- **New file: `demo/josh/config.ts`**
- All paper hyperparameters as constants
  ```
  stage1: { iters: 500, lr: 0.07, w3D: 1, w2D: 0, wc1: 1, wc2: 20, wp: 10, ws: 1 }
  stage2: { iters: 200, lr: 0.014, w3D: 1, w2D: 1, wc1: 1, wc2: 20, wp: 10, ws: 1 }
  chunkSize: 100, keyframeInterval: 0.2, contactThreshold: 0.05
  Δc1: 0, Δc2: 0.1
  ```

### 4C: Offline Entry Point
- **New file: `demo/main-offline.ts`** — drag-drop video, batch process, 3D viewer
- **Update: `demo/index.html`** — tab: Realtime | Offline

### 4D: Keep Existing Realtime Demo
- Current realtime mode stays (MiDAS + ROMP + simple solver)
- Offline mode = faithful JOSH

---

## Verification Summary

Each phase has component-level tests. The **end-to-end correctness chain** is:

```
1. Unit tests per kernel (known inputs → expected outputs)
        ↓
2. Numerical gradient check (finite diff vs analytical Jacobian, <1e-4)
        ↓
3. tf.js cross-validation (WGSL gradient vs tf.grad(), <1e-4)
        ↓
4. Optimization convergence (synthetic scene: loss decreases, params converge)
        ↓
5. Visual validation (rendered mesh aligns with person in video)
        ↓
6. Paper comparison (if evaluation data available: W-MPJPE metric)
```

### Test Commands
```bash
npm run test                    # all unit tests (vitest)
npm run test:gradients          # gradient validation vs tf.js (slow, ~30s)
npm run test:convergence        # synthetic optimization test (slow, ~60s)
npm run test:e2e                # full pipeline on 3-frame clip
```

---

## Implementation Priority

| Phase | Est. Days | What it unlocks | Risk |
|-------|-----------|----------------|------|
| 0A (WGSL Adam) | 1 | GPU-native optimizer | Low |
| 0B (SMPL + Jacobian) | 4 | Differentiable SMPL | **High** — math complexity |
| 0C (6 loss kernels) | 2 | Complete loss function | Medium |
| 0D (Chain rule JVP) | 1 | Full gradient pipeline | Medium |
| 0E (Optimizer loop) | 2 | Working optimization | Medium |
| 0F (tf.js validation) | 1 | Correctness proof | Low |
| 1A (Segmentation) | 0.5 | Clean MASt3R input | Low |
| 1B-1C (MASt3R) | 4 | Paper-faithful 3D | **High** — ONNX export |
| 1D (Focal recovery) | 0.5 | Camera intrinsics | Low |
| 1E (MoveNet) | 0.5 | 2D supervision | Low |
| 1F (Contact detect) | 0.5 | Contact labels | Low |
| 2A-2F (Batch pipeline) | 4 | Offline processing | Medium |
| 3A-3C (3D rendering) | 3 | Visualization | Low |
| 4A-4D (SMPL + integration) | 2 | Polish | Low |

**Total: ~27 days. Critical path: 0B (SMPL Jacobian) → 0E (loop) → 1B (MASt3R).**

## Key Risk: SMPL Analytical Jacobian (Phase 0B)

This is the hardest part. The Rodrigues derivative + FK chain rule + LBS Jacobian requires ~400 lines of WGSL with complex math. If it proves too error-prone, fallback to **finite differences** (90 SMPL forward passes per iteration, ~24ms overhead per iteration, ~17s per frame — still acceptable for offline).
