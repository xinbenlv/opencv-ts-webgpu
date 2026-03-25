import type { GComputeNode, NodeContext, ExecutionContext, PortDescriptor } from '../../../graph/node.ts';
import type { NodeId } from '../../../core/types.ts';
import { createNodeId, computeBufferLayout, dim } from '../../../core/types.ts';
import type { Shape2D, Shape3D } from '../../../core/types.ts';
import { KernelRunner } from '../../../backends/webgpu/kernel-runner.ts';
import { midasPreprocessKernel, midasPostprocessKernel } from '../../../kernels/depth/depth-estimation.kernel.ts';

// Default resolution for depth estimation
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

/**
 * Node A: WebGPU-accelerated monocular depth estimation.
 *
 * Pipeline:
 * 1. Preprocess (WGSL): normalize RGB → ImageNet mean/std
 * 2. Inference: MiDAS/DPT via onnxruntime-web WebGPU EP
 * 3. Postprocess (WGSL): inverse depth → metric depth
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
  private _preprocessBuffer: GPUBuffer | null = null;
  private _rawDepthBuffer: GPUBuffer | null = null;

  async initialize(ctx: NodeContext): Promise<void> {
    this._kernelRunner = new KernelRunner(ctx.device);

    // Allocate intermediate buffers
    this._preprocessBuffer = ctx.device.createBuffer({
      size: inputLayout.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      label: 'depth_preprocess',
    });

    this._rawDepthBuffer = ctx.device.createBuffer({
      size: depthLayout.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      label: 'depth_raw',
    });
  }

  async execute(ctx: ExecutionContext): Promise<void> {
    const inputBuffer = ctx.getInput('rgbFrame') as GPUBuffer;
    const outputBuffer = ctx.getOutput('depthMap') as GPUBuffer;

    // Step 1: Preprocess — normalize to ImageNet mean/std
    const preprocessUniforms = new ArrayBuffer(8);
    const preprocessView = new DataView(preprocessUniforms);
    preprocessView.setUint32(0, DEPTH_W, true);
    preprocessView.setUint32(4, DEPTH_H, true);

    await this._kernelRunner!.dispatch(
      ctx.commandEncoder,
      midasPreprocessKernel,
      [inputBuffer, this._preprocessBuffer!],
      [DEPTH_H, DEPTH_W],
      preprocessUniforms,
    );

    // Step 2: DNN inference would happen here via onnxruntime-web
    // For now, copy preprocessed data to raw depth as placeholder
    ctx.commandEncoder.copyBufferToBuffer(
      this._preprocessBuffer!,
      0,
      this._rawDepthBuffer!,
      0,
      Math.min(inputLayout.byteLength, depthLayout.byteLength),
    );

    // Step 3: Postprocess — inverse depth to metric depth
    const postprocessUniforms = new ArrayBuffer(16);
    const postView = new DataView(postprocessUniforms);
    postView.setUint32(0, DEPTH_W, true);
    postView.setUint32(4, DEPTH_H, true);
    postView.setFloat32(8, 1.0, true); // scale
    postView.setFloat32(12, 0.0, true); // shift

    await this._kernelRunner!.dispatch(
      ctx.commandEncoder,
      midasPostprocessKernel,
      [this._rawDepthBuffer!, outputBuffer],
      [DEPTH_H, DEPTH_W],
      postprocessUniforms,
    );
  }

  dispose(): void {
    this._preprocessBuffer?.destroy();
    this._rawDepthBuffer?.destroy();
    this._kernelRunner?.clearCache();
  }
}
