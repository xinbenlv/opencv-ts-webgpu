import type { GComputeNode, NodeContext, ExecutionContext, PortDescriptor } from '../../../src/graph/node.ts';
import type { NodeId } from '../../../src/core/types.ts';
import { createNodeId, computeBufferLayout, dim } from '../../../src/core/types.ts';
import type { Shape2D, Shape3D } from '../../../src/core/types.ts';
import { KernelRunner } from '../../../src/backends/webgpu/kernel-runner.ts';
import { midasPostprocessKernel } from '../kernels/depth-estimation.kernel.ts';

/**
 * MiDAS v2.1 small expects 256×256 input (NCHW: [1,3,256,256]).
 * Graph ports stay at 384×384 for pipeline resolution.
 */
const MODEL_H = 256;
const MODEL_W = 256;
const DEPTH_H = 384;
const DEPTH_W = 384;

const inputLayout = computeBufferLayout(
  [dim(DEPTH_H), dim(DEPTH_W), dim(3)] as Shape3D,
  'f32',
);

const depthLayout = computeBufferLayout(
  [dim(DEPTH_H), dim(DEPTH_W)] as Shape2D,
  'f32',
);

const INPUT_PORTS = [
  { name: 'rgbFrame', layout: inputLayout },
] as const satisfies readonly PortDescriptor[];

const OUTPUT_PORTS = [
  { name: 'depthMap', layout: depthLayout },
] as const satisfies readonly PortDescriptor[];

let ortModule: typeof import('onnxruntime-web') | null = null;

async function getOrt() {
  if (!ortModule) {
    ortModule = await import('onnxruntime-web');
  }
  return ortModule;
}

/**
 * Node A: Monocular depth estimation using MiDAS v2.1 small via ONNX Runtime Web.
 *
 * 1. Read RGB frame from GPU (separate encoder + submit for readback)
 * 2. CPU: resize 384→256, normalize, HWC→NCHW
 * 3. ONNX inference via onnxruntime-web
 * 4. CPU: resize 256→384
 * 5. GPU upload + postprocess kernel on the graph's command encoder
 */
export class DepthEstimationNode
  implements GComputeNode<typeof INPUT_PORTS, typeof OUTPUT_PORTS>
{
  readonly id: NodeId = createNodeId('depthEstimation');
  readonly name = 'DepthEstimation';
  readonly backendHint = 'webgpu' as const;
  readonly inputs = INPUT_PORTS;
  readonly outputs = OUTPUT_PORTS;

  private _kernelRunner: KernelRunner | null = null;
  private _rawDepthBuffer: GPUBuffer | null = null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private _session: any = null;
  private _device: GPUDevice | null = null;

  private _modelInput = new Float32Array(1 * 3 * MODEL_H * MODEL_W);
  private _depthOutput384 = new Float32Array(DEPTH_H * DEPTH_W);

  constructor(private readonly _modelUrl = './assets/models/midas-v2.1-small-256.onnx') {}

  async initialize(ctx: NodeContext): Promise<void> {
    this._device = ctx.device;
    this._kernelRunner = new KernelRunner(ctx.device);

    this._rawDepthBuffer = ctx.device.createBuffer({
      size: depthLayout.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      label: 'depth_raw',
    });

    const ort = await getOrt();

    // Point ORT WASM files to demo/assets/ort/ (copied from node_modules at build time)
    ort.env.wasm.wasmPaths = './assets/ort/';
    ort.env.wasm.numThreads = 1; // Avoid SharedArrayBuffer conflicts with main thread

    try {
      this._session = await ort.InferenceSession.create(this._modelUrl, {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
      });
      console.log('[DepthEstimation] Using WebGPU execution provider');
    } catch (e) {
      console.warn('[DepthEstimation] WebGPU EP unavailable, falling back to WASM:', e);
      this._session = await ort.InferenceSession.create(this._modelUrl, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
    }
    console.log('[DepthEstimation] Model loaded:', this._session.inputNames, this._session.outputNames);
  }

  async execute(ctx: ExecutionContext): Promise<void> {
    const inputBuffer = ctx.getInput('rgbFrame') as GPUBuffer;
    const outputBuffer = ctx.getOutput('depthMap') as GPUBuffer;

    if (!this._session) {
      ctx.commandEncoder.clearBuffer(outputBuffer, 0, depthLayout.byteLength);
      return;
    }

    // Step 1: GPU readback using a SEPARATE encoder (don't touch the graph's encoder)
    const readbackEncoder = this._device!.createCommandEncoder({ label: 'depth_readback' });
    const staging = this._device!.createBuffer({
      size: inputLayout.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: 'depth_staging_in',
    });
    readbackEncoder.copyBufferToBuffer(inputBuffer, 0, staging, 0, inputLayout.byteLength);
    this._device!.queue.submit([readbackEncoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const rgbData = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();

    // Step 2: CPU preprocessing — resize 384→256, HWC→NCHW, ImageNet normalize
    const MEAN = [0.485, 0.456, 0.406] as const;
    const STD = [0.229, 0.224, 0.225] as const;

    for (let y = 0; y < MODEL_H; y++) {
      for (let x = 0; x < MODEL_W; x++) {
        const srcX = (x / MODEL_W) * DEPTH_W;
        const srcY = (y / MODEL_H) * DEPTH_H;
        const x0 = Math.floor(srcX);
        const y0 = Math.floor(srcY);
        const x1 = Math.min(x0 + 1, DEPTH_W - 1);
        const y1 = Math.min(y0 + 1, DEPTH_H - 1);
        const fx = srcX - x0;
        const fy = srcY - y0;

        for (let c = 0; c < 3; c++) {
          const v00 = rgbData[(y0 * DEPTH_W + x0) * 3 + c]!;
          const v10 = rgbData[(y0 * DEPTH_W + x1) * 3 + c]!;
          const v01 = rgbData[(y1 * DEPTH_W + x0) * 3 + c]!;
          const v11 = rgbData[(y1 * DEPTH_W + x1) * 3 + c]!;
          const val = v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) +
                      v01 * (1 - fx) * fy + v11 * fx * fy;
          this._modelInput[c * MODEL_H * MODEL_W + y * MODEL_W + x] =
            (val - MEAN[c]!) / STD[c]!;
        }
      }
    }

    // Step 3: ONNX inference
    const ort = await getOrt();
    const inputTensor = new ort.Tensor('float32', this._modelInput, [1, 3, MODEL_H, MODEL_W]);
    const feeds: Record<string, InstanceType<typeof ort.Tensor>> = {};
    feeds[this._session.inputNames[0]!] = inputTensor;
    const results = await this._session.run(feeds);
    const outputTensor = results[this._session.outputNames[0]!]!;
    const rawDepth256 = outputTensor.data as Float32Array;

    // Step 4: Resize 256→384 (bilinear)
    for (let y = 0; y < DEPTH_H; y++) {
      for (let x = 0; x < DEPTH_W; x++) {
        const srcX = (x / DEPTH_W) * MODEL_W;
        const srcY = (y / DEPTH_H) * MODEL_H;
        const x0 = Math.floor(srcX);
        const y0 = Math.floor(srcY);
        const x1 = Math.min(x0 + 1, MODEL_W - 1);
        const y1 = Math.min(y0 + 1, MODEL_H - 1);
        const fx = srcX - x0;
        const fy = srcY - y0;

        const v00 = rawDepth256[y0 * MODEL_W + x0]!;
        const v10 = rawDepth256[y0 * MODEL_W + x1]!;
        const v01 = rawDepth256[y1 * MODEL_W + x0]!;
        const v11 = rawDepth256[y1 * MODEL_W + x1]!;
        this._depthOutput384[y * DEPTH_W + x] =
          v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) +
          v01 * (1 - fx) * fy + v11 * fx * fy;
      }
    }

    // Step 5: Upload raw depth to GPU and run postprocess shader
    this._device!.queue.writeBuffer(this._rawDepthBuffer!, 0, this._depthOutput384);

    const postprocessUniforms = new ArrayBuffer(16);
    const postView = new DataView(postprocessUniforms);
    postView.setUint32(0, DEPTH_W, true);
    postView.setUint32(4, DEPTH_H, true);
    postView.setFloat32(8, 1.0, true);
    postView.setFloat32(12, 0.0, true);

    await this._kernelRunner!.dispatch(
      ctx.commandEncoder,
      midasPostprocessKernel,
      [this._rawDepthBuffer!, outputBuffer],
      [DEPTH_H, DEPTH_W],
      postprocessUniforms,
    );
  }

  dispose(): void {
    this._rawDepthBuffer?.destroy();
    this._kernelRunner?.clearCache();
    this._session?.release();
  }
}
