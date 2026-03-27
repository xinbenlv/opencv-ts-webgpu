# OpenCV.ts — TypeScript-first Computer Vision on WebGPU + WASM

> Can you rebuild a modern computer vision framework entirely in TypeScript — no C++, no Python — running on WebGPU and WASM?

This repo is our answer. OpenCV.ts is a proof-of-concept CV framework that runs entirely in the browser, using WebGPU for GPU kernels, WASM for CPU algorithms, and TypeScript throughout.

**[v1.0.0 Release](https://github.com/xinbenlv/opencv-ts-webgpu/releases/tag/v1.0.0)**

---

## Why

OpenCV has been the standard CV library for 20 years — but it's C++ with Python bindings. You can't run it in a browser, and setup is a nightmare.

In 2024, [WebGPU](https://gpuweb.github.io/gpuweb/) gives browsers a real GPU compute API (same model as Metal/Vulkan). WASM gives near-native CPU performance. The question: can you build a complete CV framework on top of these?

We think yes. The goals:

- **Runs everywhere** — any browser, any OS, any WebGPU-capable hardware. Zero install.
- **Type-safe** — tensor shapes, dtypes, and buffer layouts checked at compile time.
- **Composable** — CV algorithms are TypeScript functions; they compose naturally with WebCodecs, Canvas, WebXR.
- **No dependency hell** — `npm install` is the entire setup.

---

## What's in this repo

### Core framework

| Module | Description |
|--------|-------------|
| `src/core/` | Typed tensor buffers, buffer manager, resource tracker |
| `src/graph/` | Graph-based compute pipeline — nodes, edges, execution |
| `src/backends/webgpu/` | WebGPU compute backend with WGSL kernels |
| `src/backends/wasm/` | Rust → WASM backend (via wasm-pack) |
| `src/backends/ort/` | ONNX Runtime Web integration (WebGPU EP) |

### Demos

All demos run entirely in the browser — no server, no install.

| Demo | What it shows |
|------|--------------|
| `index.html` | Full JOSH pipeline — drag a video, get animated 3D human mesh |
| `phase-midas-pointcloud.html` | MiDaS monocular depth → 3D point cloud |
| `phase-blazepose.html` | BlazePose 3D → SMPL joint initialization |
| `phase-tf-optimizer.html` | 2-stage Adam optimizer with 6 JOSH loss terms |
| `phase4.html` | SMPL model inspector + upload |
| `phase3c.html` | Timeline playback with 3D mesh renderer |

### End-to-end: JOSH human motion capture

The hardest demo — a faithful in-browser reimplementation of the [JOSH paper](https://arxiv.org/abs/2312.01234) (CVPR). Reconstructs 3D human motion from monocular video:

```
Video (WebCodecs, 5 FPS)
  → MiDaS depth (ONNX, WebGPU EP) → 3D point cloud
  → BlazePose 3D (tf.js) → SMPL joint init
  → 2-stage Adam optimizer, 700 iterations (tf.js variableGrads)
      ├─ L_3D  : 3D point correspondence
      ├─ L_2D  : 2D keypoint reprojection
      ├─ L_c1  : contact scale constraint
      ├─ L_c2  : contact static constraint
      ├─ L_p   : SMPL pose/shape prior
      └─ L_smooth : temporal smoothness
  → SMPL mesh (6890 vertices, 24 joints)
```

Performance on Apple Silicon (offline mode):
- M2 MacBook Pro: ~8–15s per frame
- Main bottleneck: tf.js tensor allocation in the optimizer inner loop

---

## Tech stack

```
TypeScript 5.7   entire codebase
WebGPU / WGSL    GPU compute kernels
Rust → WASM      CPU algorithms (wasm-pack)
onnxruntime-web  model inference (WebGPU execution provider)
TensorFlow.js    differentiable ops + Adam optimizer
Vite 6           build + dev server
Vitest           147 unit tests
```

---

## Getting started

```bash
npm install
npm run dev
# → https://localhost:5182
```

To build the WASM module (requires [Rust](https://rustup.rs/) + [wasm-pack](https://rustwasm.github.io/wasm-pack/)):

```bash
npm run wasm:build
```

### SMPL model (required for human mesh demos)

The SMPL body model requires a free academic license — we can't redistribute it.

1. Register at [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/)
2. Download `SMPL_python_v.1.1.0.zip`
3. Run the converter: `python scripts/convert-smpl.py`
4. Place `smpl-neutral.smpl.bin` in `demo/smpl/`

The browser will auto-load it on next page open.

### Secure context requirement

WebGPU and `SharedArrayBuffer` require HTTPS. The Vite dev server handles this automatically (self-signed cert via `@vitejs/plugin-basic-ssl`). If accessing from another machine:

```bash
ssh -L 5182:localhost:5182 user@<host-ip>
# then open https://localhost:5182
```

---

## Project structure

```
src/
  core/          tensor types, buffer manager, resource tracker
  graph/         compute graph executor and compiler
  backends/
    webgpu/      WGSL compute shaders
    wasm/        Rust source (wasm-pack)
    ort/         ONNX Runtime Web wrapper
  kernels/       CV algorithm implementations
demo/
  josh/
    tf/          tf.js SMPL forward + losses + optimizer
    nodes/       MiDaS, BlazePose, batch pipeline nodes
    batch/       frame extractor, pipeline orchestrator, result cache
    rendering/   WebGPU 3D mesh + point cloud renderer
  index.html     main demo (realtime + offline tabs)
tests/
  unit/          core framework tests
  josh/          pipeline + gradient validation tests (147 total)
wasm/            Rust source for WASM backend
scripts/         SMPL model converter
```

---

## Status

v1.0.0 — working proof-of-concept. The API will change. Performance is not yet at parity with OpenCV C++. But the architecture works and the hard demos run.

**What's next:**
- More kernels: convolution, feature detection, optical flow, homography
- WGSL-native optimizer inner loop (5–10x speedup over tf.js)
- WebXR integration for AR overlays
- npm package publish

---

## License

Apache-2.0. SMPL model weights are separately licensed by the Max Planck Institute.
