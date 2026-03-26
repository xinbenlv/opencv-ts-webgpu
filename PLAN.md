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

## Demo & Test Strategy

Every phase of this project has two concrete deliverables alongside its implementation:

**Visual demos** — self-contained HTML files in `demo/phaseN.html`. Each file loads
directly in the browser with no build step required (ES modules via importmap or inline
scripts). They are the fastest way to verify a sub-system looks right before running
automated tests.

**Vitest unit tests** — TypeScript test files in `tests/josh/`. They exercise each
sub-system with known synthetic inputs and expected outputs so regressions are caught
immediately.

### Demo files by phase
| Phase | Demo file | Access URL |
|-------|-----------|------------|
| Phase 0 | `demo/phase0.html` | `https://localhost:PORT/demo/phase0.html` (already built) |
| Phase 1 | `demo/phase1.html` | `https://localhost:PORT/demo/phase1.html` |
| Phase 2 | `demo/phase2.html` | `https://localhost:PORT/demo/phase2.html` |
| Phase 3 | `demo/phase3.html` | `https://localhost:PORT/demo/phase3.html` |

### Running tests
```bash
npm test                        # all vitest unit tests
npm run test:gradients          # gradient validation vs tf.js (slow, ~30s)
npm run test:convergence        # synthetic optimization convergence (slow, ~60s)
npm run test:e2e                # full pipeline on 3-frame clip
```

Each sub-phase below lists its specific **Demo** and **Test** requirements. The demo
description says what must be visually verifiable; the test description says what
assertions vitest must pass.

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
- **Demo:** `demo/phase0.html` tab "Adam Optimizer" — sliders for lr/beta1/beta2, table showing params[0..4] evolving over 20 steps, comparison column with hand-computed expected values highlighted green/red based on match within 1e-5
- **Test:** `tests/josh/adam-optimizer.test.ts` — compare against known Adam step results (hand-computed for 5 params, 3 iterations); compare against tf.js `tf.train.adam()` on same inputs — values should match to 1e-5

### 0B: WGSL Differentiable SMPL Forward Pass
- **Rewrite: `demo/josh/kernels/smpl-forward.wgsl`** — shape blend + LBS
- **Rewrite: `demo/josh/kernels/smpl-joints.wgsl`** — Rodrigues + FK
- Must compute BOTH vertices AND per-vertex Jacobians d(v)/d(params)
- Analytical Jacobian of Rodrigues: dR/dw_i (known closed-form formula)
- FK chain rule: d(global_T)/d(θ_j) = parent_T × d(local_T)/d(θ_j)
- Output: vertices[6890,3] + jacobian[6890*3, N_params] (~7.4MB)
- **New file: `demo/josh/kernels/rodrigues-deriv.wgsl`** — dR(w)/dw_i
- **Demo:** `demo/phase0.html` tab "SMPL Forward" — renders a 3D scatter of 6890 vertex positions for T-pose (zero shape, zero pose) using a simple WebGL/WebGPU canvas; slider to perturb a single joint angle and watch the skeleton deform in real time; numerical readout of Jacobian vs finite-difference for the selected joint
- **Test:** `tests/josh/smpl-forward.test.ts` — load known SMPL test case (T-pose, zero shape → expected vertex positions); perturb single joint by 0.1 rad → verify FK output changes correctly; numerical gradient check: finite diff vs analytical Jacobian, relative error < 1e-4; compare tf.js SMPL forward (Phase 0F) against WGSL output — should match to 1e-5

### 0C: 6 Loss Function Kernels (WGSL)
Each kernel outputs: scalar loss value + per-vertex gradient dL/dv[6890*3]

- **Rewrite: `demo/josh/kernels/josh-contact-loss.wgsl`** — L_c1 contact scale
  - L_c1 = wc1 × Σ max(0, ‖v_contact - scene_nearest‖ - Δc1)²
  - **Demo:** `demo/phase0.html` tab "Contact Loss L_c1" — 2D side-view canvas showing a foot vertex above a floor plane; drag the vertex vertically to change the gap; live readout of L_c1 and dL/dz confirming the hinge at Δc1=0
  - **Test:** `tests/josh/loss-contact.test.ts` — known contact vertex at (0,0,1), scene surface at z=1.05; expected loss = wc1 × (0.05 - 0)² = 0.0025 (if Δc1=0); expected gradient: d/dz = 2 × wc1 × 0.05 = 0.1

- **New file: `demo/josh/kernels/josh-contact-static.wgsl`** — L_c2 contact static
  - L_c2 = wc2 × Σ max(0, ‖v_contact(t) - v_contact(t-1)‖ - Δc2)²
  - **Demo:** `demo/phase0.html` tab "Contact Static Loss L_c2" — two foot-vertex positions at t-1 (fixed) and t (draggable); slider shows displacement magnitude with live L_c2 readout and the dead-zone at Δc2=0.1 highlighted
  - **Test:** `tests/josh/loss-contact-static.test.ts` — contact vertex moved 0.2m between frames, Δc2=0.1; expected loss = wc2 × (0.2 - 0.1)² = 20 × 0.01 = 0.2

- **Rewrite: `demo/josh/kernels/josh-depth-reproj.wgsl`** — L_3D correspondence
  - L_3D = Σ ‖π⁻¹(Z^t, K, P^t) - X_matched‖² (3D point matching)
  - **Demo:** `demo/phase0.html` tab "3D Reprojection Loss L_3D" — side-by-side: depth map image with overlaid matched pixel markers, and 3D scatter of unprojected points vs ground-truth MASt3R points; numeric L_3D value updates as camera pose is perturbed via sliders
  - **Test:** `tests/josh/loss-depth-reproj.test.ts` — project known 3D point to depth map, reconstruct via π⁻¹ → should recover original point to within 1e-4; verify gradient direction points from reconstructed point toward ground-truth point

- **New file: `demo/josh/kernels/josh-reproj-2d.wgsl`** — L_2D reprojection
  - L_2D = Σ ‖π(J_3D, K) - j_2D‖² × confidence
  - **Demo:** `demo/phase0.html` tab "2D Reprojection Loss L_2D" — canvas with a person silhouette background; circles for MoveNet-detected joints and crosses for SMPL-projected joints; drag a 3D joint (x,y,z sliders) and watch the projected cross move with live L_2D readout
  - **Test:** `tests/josh/loss-reproj-2d.test.ts` — SMPL joint at (0,0,2) with f=300, cx=192 → expected pixel (192,192); move joint to (0.1,0,2) → expected pixel (207,192); verify L_2D against manually computed squared pixel error

- **New file: `demo/josh/kernels/josh-smpl-prior.wgsl`** — L_p SMPL prior
  - L_p = wp × (‖θ - θ₀‖² + ‖β - β₀‖²)
  - **Demo:** `demo/phase0.html` tab "SMPL Prior L_p" — two mesh previews side by side (prior pose θ₀ and perturbed θ); slider controls perturbation magnitude; live readout of L_p matches formula; gradient norm displayed
  - **Test:** `tests/josh/loss-smpl-prior.test.ts` — θ = θ₀ + 0.1 for all 72 → loss = 10 × 72 × 0.01 = 7.2; β = β₀ + 0.2 for all 10 → loss contribution = 10 × 10 × 0.04 = 4.0; verify gradient is exactly 2 × wp × (θ - θ₀) per parameter

- **New file: `demo/josh/kernels/josh-smooth.wgsl`** — L_smooth temporal
  - L_smooth = w_s × (‖P^t - P^{t-1}‖² + ‖θ^t - θ^{t-1}‖²)
  - **Demo:** `demo/phase0.html` tab "Smoothness Loss L_smooth" — timeline strip showing 5 frames; drag a pose (camera or joint) in the current frame; adjacent-frame poses shown as ghost overlays; L_smooth readout updates live showing pose-change contribution vs joint-change contribution separately
  - **Test:** `tests/josh/loss-smooth.test.ts` — camera moved 0.5m between frames, w_s=1 → expected loss contribution = 1 × 0.25 = 0.25; θ changed by 0.3 rad in all 72 joints → loss += 1 × 72 × 0.09 = 6.48; verify total matches sum

### 0D: Chain Rule Kernel (Jacobian-Vector Product)
- **New file: `demo/josh/kernels/jvp-gradient.wgsl`**
- gradient[j] = Σ_i J[i,j] × dL_dv[i] for all params
- Converts per-vertex loss gradients → per-parameter gradients
- 89 threads (or more for depth map params in Stage 2)
- **Demo:** `demo/phase0.html` tab "Chain Rule JVP" — matrix visualization: left panel is a heatmap of a synthetic 10×5 Jacobian, middle panel is dL/dv[10], right panel is the resulting gradient[5]; button to randomize Jacobian/dL and verify GPU result matches CPU matrix multiply shown below
- **Test:** `tests/josh/chain-rule.test.ts` — create synthetic Jacobian (identity-like) + known dL/dv; verify output matches matrix multiplication result; cross-check: tf.js `tf.grad(loss_through_smpl)` vs WGSL chain rule — should match to 1e-4

### 0E: Optimization Loop Orchestrator
- **Rewrite: `demo/josh/nodes/josh-solver.node.ts`**
- Records all 700 iterations into ONE command encoder (no mapAsync in loop!)
- Stage 1: 500 iters, lr=0.07, w2D=0, optimize σ^t + P^t + θ^t
- Stage 2: 200 iters, lr=0.014, w2D enabled, also optimize Z^t
- Single submit + single mapAsync at end
- Progress: read loss buffer every 50 iterations (separate small readback)
- **Demo:** `demo/phase0.html` tab "Optimization Loop" — synthetic scene with a stick-figure person at known position; "Run 700 iters" button triggers optimization; live loss-curve canvas updates every 50 iterations; final mesh overlaid on the synthetic scene shows convergence; elapsed time and iterations/sec displayed
- **Test:** `tests/josh/optimizer-loop.test.ts` — synthetic scene: person standing on flat ground at known position; initialize with 10% error in all params; after 700 iterations: params should converge within 5% of ground truth; loss should be strictly lower after stage 1 and again after stage 2

### 0F: tf.js Reference Implementation (for validation only)
- **New file: `demo/josh/tf/smpl-forward-ref.ts`** — tf.js SMPL forward
- **New file: `demo/josh/tf/losses-ref.ts`** — tf.js loss functions
- NOT used in production pipeline — only for `npm run test` gradient validation
- **Demo:** `demo/phase0.html` tab "tf.js vs WGSL Validation" — table of 89 parameter gradients with three columns: tf.js tf.grad() value, WGSL chain-rule value, relative error; rows highlighted red if error > 1e-4; "Run validation" button triggers both computations and populates the table
- **Test:** `tests/josh/gradient-validation.test.ts` — for random params: compute gradient via WGSL chain rule vs tf.js tf.grad(); relative error should be < 1e-4 for all 89 parameters; this is the DEFINITIVE correctness test

---

## Phase 1: Preprocessing Models

### 1A: Foreground Segmentation
- **New file: `demo/josh/nodes/segmentation.node.ts`**
- Use BodyPix (tf.js built-in) or MediaPipe Selfie Segmentation ONNX
- Output: binary mask per frame (person = 0, background = 1)
- Applied to MASt3R input to exclude human region
- **Demo:** `demo/phase1.html` tab "Segmentation" — drag-drop a video file or use the bundled JOSH demo clip; shows original frame on the left, binary mask overlay on the right; slider to scrub through frames; mask coverage percentage displayed per frame
- **Test:** `tests/josh/segmentation.test.ts` — run on JOSH demo video frame → mask should cover the person; verify mask has reasonable area (10–60% of frame); verify MASt3R pointmap filtered by mask excludes human body region

### 1B: MASt3R ONNX Export
- Clone https://github.com/naver/mast3r
- `torch.onnx.export()` with ViT-Large checkpoint
- Input: two images [1,3,H,W] (max dim 512)
- Output: pointmaps [1,H,W,3] × 2, confidence [1,H,W] × 2
- **Output: `mast3r-vit-large-fp32.onnx` (~2.75GB)** in Git LFS
- **Demo:** `demo/phase1.html` tab "MASt3R Pointmap" — upload two frames from the JOSH clip; displays dense 3D point cloud colored by confidence (hot colormap); camera baseline vector drawn as an arrow; loading spinner with progress bar during model download
- **Test:** `tests/josh/mast3r-export.test.ts` — run ONNX model on two known frames → compare pointmap with PyTorch reference output; max absolute difference < 1e-3 (FP32 rounding)

### 1C: MASt3R Node
- **New file: `demo/josh/nodes/mast3r.node.ts`**
- Load via cachedFetchModel() (Cache API for 2.75GB)
- Process frame pairs with sliding window-10 graph
- Output: per-frame dense point clouds in world coordinates
- **Demo:** `demo/phase1.html` tab "MASt3R Node" — processes the first 5 frames of the demo clip with window-10 graph; renders a combined world-space point cloud for all frames colored by frame index; camera trajectory drawn as connected arrows; shows total points per frame
- **Test:** `tests/josh/mast3r-node.test.ts` — two frames of a static scene → pointmaps should be consistent (mean point distance < 2cm); camera baseline between frames should match known motion to within 5%

### 1D: Focal Length Recovery
- **New file: `demo/josh/utils/focal-recovery.ts`**
- Port `recover_focal_shift()` from JOSH Python code
- Least-squares: minimize |f × xy/z - uv| for focal length f and depth shift
- Output: camera intrinsics K (fixed throughout optimization)
- **Demo:** `demo/phase1.html` tab "Focal Recovery" — scatter plot of projected points (u, v) from MASt3R vs true pixel positions; slider to add Gaussian noise to the points; "Recover f" button runs least-squares and displays recovered focal length vs true focal length with percent error; residual plot shown beneath
- **Test:** `tests/josh/focal-recovery.test.ts` — synthetic data: known f=300, generate projected points with no noise → recovered f within 0.5%; add σ=2px noise → recovered f within 2%; verify depth shift recovery is also within 2%

### 1E: MoveNet 2D Pose
- **New file: `demo/josh/nodes/pose-2d.node.ts`**
- MoveNet SinglePose Lightning ONNX (~12MB)
- 17 COCO keypoints → map to SMPL joints via correspondence table
- **Demo:** `demo/phase1.html` tab "MoveNet 2D Pose" — video frame with skeleton overlay drawn using the 17 detected keypoints; confidence scores shown as color (green = high, red = low); correspondence table shown below mapping each COCO joint to its SMPL joint index; scrub through frames to see skeleton track the person
- **Test:** `tests/josh/pose-2d.test.ts` — run on JOSH demo frame → should detect person with confidence > 0.5 for at least 12 of 17 keypoints; key joints (hips, shoulders, head) should be within 20px of expected ground-truth positions

### 1F: Contact Detection (Geometric)
- **New file: `demo/josh/utils/contact-detection.ts`**
- For candidate vertices (feet soles, hands), project to depth map
- If |vertex_z - depth_z| < 5cm → mark as contact
- **Demo:** `demo/phase1.html` tab "Contact Detection" — 3D view of the SMPL mesh overlaid on the MASt3R point cloud; contact vertices highlighted in red; foot and hand candidate vertices shown as larger spheres; depth threshold slider (1–10 cm) updates contact labels in real time
- **Test:** `tests/josh/contact-detection.test.ts` — person standing on flat ground → at least 4 foot-sole vertices marked as contacts; raised hand (hand z < floor z - 30 cm) → no hand vertices marked as contacts; verify threshold boundary: vertex exactly at threshold is included, vertex 1mm beyond is not

---

## Phase 2: Offline Batch Pipeline

### 2A: Frame Extraction
- **New file: `demo/josh/batch/frame-extractor.ts`**
- WebCodecs VideoDecoder, sample at 5 FPS
- **Demo:** `demo/phase2.html` tab "Frame Extractor" — drag-drop any video file; shows a grid of extracted frames (thumbnail size) labeled with timestamps; displays frame count, resolution, and extraction speed (frames/sec); progress bar fills as frames are decoded
- **Test:** `tests/josh/frame-extractor.test.ts` — extract a synthetic 10s test video at 5 FPS → should produce ~50 frames with correct resolution and monotonically increasing timestamps; verify no duplicate frames

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
- **Demo:** `demo/phase2.html` tab "Batch Pipeline" — full pipeline UI: drag-drop video, nine stacked progress bars (one per stage), live loss curve canvas updating during optimization, ETA countdown, and a "Pause / Resume" button; when complete, shows summary stats (total time, frames processed, peak GPU memory)
- **Test:** `tests/josh/batch-pipeline.test.ts` — process 5-frame synthetic sequence → all phases complete without error; result cache populated → second run reads from cache and skips computation (verify by counting model inference calls = 0 on second run)

### 2C: Keyframe Interpolation
- **New file: `demo/josh/utils/se3-interpolation.ts`**
- SLERP for rotations, linear for translations
- Interpolate camera poses + SMPL params between keyframes
- **Demo:** `demo/phase2.html` tab "SE3 Interpolation" — two SMPL skeletons rendered in 3D (keyframe A and keyframe B); a scrubber slider at t=0..1 shows the smoothly interpolated skeleton between them; rotation angle displayed numerically; toggle button to switch between SLERP and naive linear-blend to show the difference
- **Test:** `tests/josh/se3-interpolation.test.ts` — SLERP of 90° rotation at t=0.5 should give 45° (within 1e-4 rad); interpolated pose at t=0 must exactly equal keyframe A; at t=1 must exactly equal keyframe B; quaternion norm at all interpolated t must remain 1.0 (within 1e-6)

### 2D: Chunk Concatenation
- **New file: `demo/josh/utils/chunk-concat.ts`**
- Chain coordinate frames: P_chunk2 = P_last_of_chunk1 × P_chunk2_local
- **Demo:** `demo/phase2.html` tab "Chunk Concatenation" — top-down 2D view of a camera trajectory reconstructed from two synthetic 10-frame chunks; chunk 1 shown in blue, chunk 2 in orange; junction point highlighted; a "Break alignment" toggle shows what happens without frame chaining so the user can see the discontinuity
- **Test:** `tests/josh/chunk-concat.test.ts` — two chunks with known SE3 transforms → concatenated trajectory is continuous (position jump at junction < 1mm); verify that relative poses within each chunk are preserved after concatenation

### 2E: Result Cache (IndexedDB)
- **New file: `demo/josh/batch/result-cache.ts`**
- Per-frame: params, vertices, joints, camera, losses, depth
- Video content hash as key
- **Demo:** `demo/phase2.html` tab "Result Cache" — processes the demo clip once and stores results; shows IndexedDB usage in KB; "Clear cache" and "Re-run" buttons; on second load, shows "Loaded from cache in Xms" vs "Computed in Xms" comparison; lists all cached video hashes present
- **Test:** `tests/josh/result-cache.test.ts` — store a known per-frame result object → load it back → deep-equal comparison; store 10 frames → query by frame index → correct frame returned; clear cache → subsequent load returns null

### 2F: Progress UI
- **Update: `demo/index.html`** — offline mode tab
- Phase bars, frame progress, iteration counter, loss curve (canvas chart)
- ETA computation based on rolling average
- **Demo:** `demo/phase2.html` tab "Progress UI" — standalone interactive mockup of the progress UI with simulated data: animated progress bars filling at realistic speeds, a live loss-curve canvas fed by random-walk data, and an ETA counter counting down; no real computation — purely for UI review
- **Test:** `tests/josh/progress-ui.test.ts` — mount the ProgressUI component (jsdom); simulate 50% completion event → verify progress bar width is 50%; simulate ETA of 120s → verify display reads "2:00"; simulate loss values [1.0, 0.8, 0.6] → verify canvas data array matches

---

## Phase 3: 3D Rendering

### 3A: SMPL Mesh Renderer
- **New file: `demo/josh/rendering/smpl-renderer.ts`**
- WebGPU render pipeline (vertex + fragment shaders)
- **New file: `demo/josh/rendering/smpl-mesh.vert.wgsl`** — MVP projection
- **New file: `demo/josh/rendering/smpl-mesh.frag.wgsl`** — Phong lighting
- **New file: `demo/josh/rendering/compute-normals.wgsl`** — per-vertex normals
- **Demo:** `demo/phase3.html` tab "SMPL Mesh Renderer" — rotating SMPL T-pose mesh rendered in WebGPU on a canvas; sliders for ambient/diffuse/specular Phong coefficients and light direction; toggle between solid, wireframe, and vertex-normal visualization modes; FPS counter in the corner
- **Test:** `tests/josh/smpl-renderer.test.ts` — render T-pose mesh to an off-screen texture (256×256); read back pixel data; verify that the central column of pixels contains non-background colors (silhouette present); verify pixel statistics: mean luminance in bounding box > 0.3; no NaN/Inf values in render output

### 3B: Scene Point Cloud Renderer
- **New file: `demo/josh/rendering/pointcloud-renderer.ts`**
- Render MASt3R pointmaps as colored points
- **Demo:** `demo/phase3.html` tab "Point Cloud Renderer" — renders a pre-computed MASt3R point cloud from the JOSH demo clip; points colored by RGB from the source frame; mouse drag to orbit, scroll to zoom; point size slider; toggle to color by depth (jet colormap) instead of RGB; point count displayed
- **Test:** `tests/josh/pointcloud-renderer.test.ts` — create a synthetic point cloud of 1000 points in a known 1m×1m×1m cube; render to off-screen texture from a frontal camera; verify rendered pixel bounding box matches expected projection of the cube corners to within 5px; verify point count readback matches input

### 3C: Timeline Playback
- **New file: `demo/josh/ui/timeline.ts`**
- Scrubber, play/pause/step, view toggles
- Loss curve per frame (from cached results)
- **Demo:** `demo/phase3.html` tab "Timeline Playback" — full 3D viewer with all three panels: SMPL mesh, MASt3R point cloud, and overlaid video frame texture; timeline scrubber at the bottom drives all three panels in sync; play/pause/step buttons; mini loss-curve chart beneath the scrubber showing per-frame final loss; view toggles for mesh / point cloud / video
- **Test:** `tests/josh/timeline.test.ts` — initialize Timeline with 10 synthetic frames; verify scrubber at frame 5 triggers onFrameChange(5); verify play() advances frame index at approximately the correct rate; verify step-forward from frame 9 clamps at frame 9 (no overflow); verify view-toggle state is correctly reflected in the UI element's class list

---

## Phase 4: SMPL Data + Integration

### 4A: SMPL Upload UI
- **New file: `demo/josh/models/smpl-loader-ui.ts`**
- Drag-drop .pkl file, parse pickle format in JS, store in IndexedDB
- Link to smpl.is.tue.mpg.de for user to register (free academic license)
- **Demo:** `demo/phase3.html` tab "SMPL Loader" — drag-drop zone for a .pkl file; after drop, shows parsed model stats: vertex count (6890), joint count (24), shape components (10), pose components (72); renders the mean-shape T-pose mesh immediately; error message if .pkl format is unrecognised
- **Test:** `tests/josh/smpl-loader.test.ts` — upload a known minimal synthetic .pkl with correct SMPL keys → verify meanTemplate shape = [6890, 3]; verify shapeDirs shape = [6890, 3, 10]; verify J_regressor has correct dimensions; verify stored IndexedDB entry matches parsed data exactly

### 4B: Config
- **New file: `demo/josh/config.ts`**
- All paper hyperparameters as constants
  ```
  stage1: { iters: 500, lr: 0.07, w3D: 1, w2D: 0, wc1: 1, wc2: 20, wp: 10, ws: 1 }
  stage2: { iters: 200, lr: 0.014, w3D: 1, w2D: 1, wc1: 1, wc2: 20, wp: 10, ws: 1 }
  chunkSize: 100, keyframeInterval: 0.2, contactThreshold: 0.05
  Δc1: 0, Δc2: 0.1
  ```
- **Demo:** `demo/phase3.html` tab "Config Inspector" — read-only table listing every hyperparameter with its value, source (paper section), and a tooltip quoting the paper text; a "Sensitivity" column runs a mini sweep (±20%) on a cached synthetic scene and shows how much the final loss changes, letting users see which parameters matter most
- **Test:** `tests/josh/config.test.ts` — import config and verify all values match the paper exactly (stage1.iters === 500, stage2.lr === 0.014, Δc2 === 0.1, etc.); verify that config is frozen/immutable (Object.isFrozen); verify TypeScript type definitions cover all fields

### 4C: Offline Entry Point
- **New file: `demo/main-offline.ts`** — drag-drop video, batch process, 3D viewer
- **Update: `demo/index.html`** — tab: Realtime | Offline
- **Demo:** `demo/phase3.html` tab "Offline Entry Point" — the complete integrated UI: Realtime and Offline tabs; in Offline tab, drag-drop a video, see all pipeline stages run, watch the 3D result build up frame by frame; in Realtime tab, existing webcam demo still works; switching between tabs preserves state
- **Test:** `tests/josh/offline-entry.test.ts` — mount the offline entry component (jsdom + WebGPU mock); simulate drag-drop of a synthetic video blob → verify pipeline starts and emits progress events in correct order (extract → segment → mast3r → ... → optimize); verify final state contains a non-empty results map

### 4D: Keep Existing Realtime Demo
- Current realtime mode stays (MiDAS + ROMP + simple solver)
- Offline mode = faithful JOSH
- **Demo:** `demo/phase3.html` tab "Realtime vs Offline" — side-by-side comparison on the same short clip: realtime solver result (fast, approximate) on the left, JOSH offline result (slow, faithful) on the right; per-joint position error table showing improvement; processing time comparison
- **Test:** `tests/josh/realtime-compat.test.ts` — verify the existing realtime pipeline (MiDAS + ROMP + simple solver) still passes its own tests after all Phase 0–4 changes; import both realtime and offline entry points in the same test file and verify they do not share mutable global state

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
