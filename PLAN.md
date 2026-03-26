# OpenCV.ts 5.0 — JOSH Demo Implementation Plan

## Current Status (2026-03-25)

The JOSH pipeline demo runs end-to-end at ~60 FPS with WebGPU + WASM on real hardware.
All three nodes execute without errors. Visualization is working for Node A, B, and C.

### What's Working
- **Graph infrastructure**: Input/output buffer allocation, `writeInput()`, `readOutput()` for graph inputs, outputs, and inter-node edges
- **Video input**: Dropdown selector for Camera vs JOSH Demo Video (raw input from genforce/JOSH repo)
- **Node A (Depth Estimation)**: WGSL preprocessing (ImageNet normalization) + postprocessing (inverse→metric depth) shaders run on GPU. Output visualized as colorized heatmap.
- **Node B (HMR)**: GPU buffers allocated, port bindings correct. Skeleton visualization renders SMPL T-pose with 24-joint kinematic tree overlaid on darkened video frame. Reads `jointPositions` from GPU buffer each frame.
- **Node C (JOSH Solver)**: L-BFGS optimizer runs in WASM. Output visualized as colorized depth heatmap.
- **demo-webapp/**: Heroku-ready Node.js server with COOP/COEP headers, Procfile, pre-built assets + WASM pkg

### What's Placeholder / Not Yet Implemented

#### Node A — Depth Estimation
- **DNN inference is a copy-through**: Line 86-94 in `depth-estimation.node.ts` just copies `preprocessBuffer` → `rawDepthBuffer` instead of running MiDAS/DPT via onnxruntime-web
- **To implement**: Load MiDAS v2.1 small ONNX model (~17MB), create `ort.InferenceSession` with WebGPU EP, run inference between preprocess and postprocess steps
- **Model source**: https://github.com/isl-org/MiDaS/releases (onnx exports available)

#### Node B — Human Mesh Recovery (HMR)
- **All outputs zero-filled**: Lines 114-118 in `human-mesh-recovery.node.ts` — `clearBuffer` on vertices, joints, camera
- **To implement**:
  1. Load HMR ONNX model (e.g., HMR 2.0 / 4DHumans) via onnxruntime-web WebGPU EP
  2. Run regression: RGB → SMPL pose (θ: 72-d), shape (β: 10-d), camera (3-d)
  3. Implement SMPL forward pass as WGSL compute shaders:
     - Blend shape deformation: `V_shaped = meanTemplate + β · shapeBlendShapes`
     - Joint regression: `J = jointRegressor · V_shaped`
     - Forward kinematics: per-joint 4×4 transforms
     - Linear blend skinning: `V_posed = Σ w_i · T_i · V_shaped`
  4. Need SMPL model data files (mean template, blend shapes, joint regressor, skinning weights) — can use SMPL-X neutral model

#### Node C — JOSH Solver
- **Gradient computation kernels not implemented**: Lines 122-127 in `josh-solver.node.ts` — TODO comments for contact loss, depth reprojection, temporal smoothness
- **L-BFGS converges immediately** with zero gradients, output is pass-through copy of depth input
- **To implement**:
  1. Contact loss gradient kernel (WGSL): penalize SMPL foot vertices not touching depth surface
  2. Depth reprojection loss gradient kernel (WGSL): align projected SMPL mesh with depth map
  3. Temporal smoothness loss gradient kernel (WGSL): penalize jitter between consecutive frames
  4. Need normal estimation (contribution slot #2) and differentiable renderer (contribution slot #4)

### Architecture Notes
- `CompiledGraph` interface exposes: `graphInputs`, `graphOutputs`, `writeInput()`, `readOutput()`, `execute()`
- `readOutput()` checks graph output buffers first, then falls back to edge buffers (for intermediate outputs like `depthMap` from depth→solver edge)
- GPU readback is throttled to every 6 frames to avoid performance impact from staging buffer creation
- `willReadFrequently: true` set on frame capture canvas to avoid Chrome warning
- WASM pinned to `wasm-bindgen 0.2.100` for Rust 1.87 compatibility

### Build & Deploy
```bash
# Dev server (HTTPS with hot reload)
npm run dev

# Build production demo
npm run wasm:build && npm run build:demo

# Run locally
cd demo-webapp && node server.js  # http://localhost:3000

# Deploy to Heroku
cd demo-webapp
git init && git add -A && git commit -m "deploy"
heroku create && git push heroku main
```

### File Map
| File | Purpose |
|------|---------|
| `demo/main.ts` | Demo entry point, video input, frame upload, visualization |
| `demo/index.html` | UI layout with 4-panel grid, input source dropdown |
| `src/graph/compiler.ts` | Graph compiler, buffer allocation for inputs/outputs/edges |
| `src/graph/executor.ts` | Runtime executor, `writeInput()`, `readOutput()` |
| `src/graphs/josh/pipeline.ts` | JOSH pipeline builder, returns node IDs |
| `src/graphs/josh/nodes/depth-estimation.node.ts` | Node A |
| `src/graphs/josh/nodes/human-mesh-recovery.node.ts` | Node B |
| `src/graphs/josh/nodes/josh-solver.node.ts` | Node C |
| `demo-webapp/` | Heroku-deployable server + pre-built assets |
| `vite.config.pages.ts` | Production build config (no SSL, no lib mode) |
