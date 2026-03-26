# OpenCV.ts 5.0 — JOSH Demo Implementation Plan

## Current Status (2026-03-25)

The JOSH pipeline demo runs end-to-end with WebGPU + WASM on real hardware.
All three nodes execute with real implementations. MiDAS depth estimation runs
via ONNX Runtime Web, SMPL forward pass runs as WGSL compute shaders,
and the JOSH solver runs gradient computation kernels with L-BFGS optimization.

### What's Working
- **Graph infrastructure**: Input/output buffer allocation, `writeInput()`, `readOutput()` for graph inputs, outputs, and inter-node edges
- **Video input**: Dropdown selector for Camera vs JOSH Demo Video
- **Node A (Depth Estimation)**: MiDAS v2.1 small (256×256) via onnxruntime-web (WebGPU EP with WASM fallback). Includes bilinear resize 384↔256, ImageNet normalization, inverse depth → metric depth postprocessing.
- **Node B (HMR)**: Full SMPL forward pass on GPU — shape blend shapes, joint regression, forward kinematics, and linear blend skinning as WGSL compute shaders. Simulated walking animation for demo (replace with ONNX HMR for production).
- **Node C (JOSH Solver)**: L-BFGS optimizer in WASM with three GPU gradient kernels: contact loss, depth reprojection loss, temporal smoothness loss. SharedArrayBuffer bridge for WASM↔GPU data transfer.
- **Git LFS**: *.mp4, *.onnx, *.bin, *.pkl tracked via Git LFS
- **Code organization**: JOSH-specific code in `demo/josh/`, generic OpenCV infrastructure in `src/`

### Architecture: opencv-ts vs demo/josh separation

**Generic OpenCV infrastructure (`src/`):**
- `src/core/` — Branded types, BufferManager, ResourceTracker, Tensor/Mat
- `src/graph/` — GComputeNode interface, GraphBuilder, GraphCompiler, GraphExecutor
- `src/backends/` — WebGPU + WASM kernel runners, pipeline cache, SharedMemoryBridge
- `src/kernels/` — Generic compute kernels (color conversion, blur, resize, sobel)
- `src/video/` — Frame sources, double-buffering

**JOSH-specific (`demo/josh/`):**
- `demo/josh/pipeline.ts` — JOSH 3-node graph builder
- `demo/josh/nodes/depth-estimation.node.ts` — Node A: MiDAS depth via ONNX Runtime
- `demo/josh/nodes/human-mesh-recovery.node.ts` — Node B: SMPL forward pass on GPU
- `demo/josh/nodes/josh-solver.node.ts` — Node C: Joint optimization solver
- `demo/josh/kernels/` — MiDAS pre/postprocess, SMPL LBS, SMPL joints FK, contact loss, depth reprojection, temporal smoothness WGSL shaders
- `demo/josh/models/` — SMPL constants, model loader, synthetic model generator

### What Could Be Improved

#### Node A — Depth Estimation
- Currently downloads 64MB ONNX model on page load. Could add progressive loading or quantized model.
- Bilinear resize is done on CPU. Could use a WGSL resize kernel.

#### Node B — Human Mesh Recovery
- Uses simulated walking animation instead of real HMR inference. To use real HMR:
  1. Export HMR 2.0 / 4DHumans model to ONNX
  2. Load via onnxruntime-web (same pattern as MiDAS)
  3. The SMPL forward pass (GPU) is already implemented
- Uses synthetic SMPL mesh data. For production, load real SMPL model from smpl.is.tue.mpg.de (academic license required)

#### Node C — JOSH Solver
- GPU gradient kernels use simplified gradient accumulation (no atomics)
- Line search uses fixed step size (0.01) — could implement proper Armijo backtracking
- Only optimizes depth scale and camera translation, not full SMPL pose

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
| `demo/josh/pipeline.ts` | JOSH pipeline builder, returns node IDs |
| `demo/josh/nodes/depth-estimation.node.ts` | Node A: MiDAS depth via ONNX Runtime Web |
| `demo/josh/nodes/human-mesh-recovery.node.ts` | Node B: SMPL forward pass (WGSL compute) |
| `demo/josh/nodes/josh-solver.node.ts` | Node C: L-BFGS + GPU gradient kernels |
| `demo/josh/kernels/*.wgsl` | WGSL compute shaders for all GPU operations |
| `demo/josh/kernels/*.kernel.ts` | Kernel descriptors (bindings, dispatch, uniforms) |
| `demo/josh/models/smpl.ts` | SMPL constants and types |
| `demo/josh/models/smpl-synthetic.ts` | Synthetic SMPL mesh generator |
| `demo/josh/models/model-loader.ts` | Runtime model loading (ONNX, SMPL binary) |
| `demo/assets/models/midas-v2.1-small-256.onnx` | MiDAS depth estimation model (Git LFS) |
| `src/graph/compiler.ts` | Graph compiler, buffer allocation for inputs/outputs/edges |
| `src/graph/executor.ts` | Runtime executor, `writeInput()`, `readOutput()` |
| `src/backends/webgpu/kernel-runner.ts` | WGSL compilation & dispatch |
| `src/backends/wasm/kernel-runner.ts` | WASM L-BFGS optimizer wrapper |
| `wasm/src/lbfgs.rs` | L-BFGS two-loop recursion (Rust → WASM) |
| `wasm/src/contact.rs` | Contact constraint evaluator (Rust → WASM) |
| `demo-webapp/` | Heroku-deployable server + pre-built assets |
| `vite.config.pages.ts` | Production build config (no SSL, no lib mode) |
