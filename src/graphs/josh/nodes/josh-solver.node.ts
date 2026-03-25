import type { GComputeNode, NodeContext, ExecutionContext, PortDescriptor } from '../../../graph/node.ts';
import type { NodeId, Shape2D } from '../../../core/types.ts';
import { createNodeId, computeBufferLayout, dim } from '../../../core/types.ts';
import { WasmKernelRunner } from '../../../backends/wasm/kernel-runner.ts';
import { SharedMemoryBridge } from '../../../backends/wasm/shared-memory.ts';
import type { WasmHandle, WasmLBFGS, WasmContact } from '../../../backends/wasm/runtime.ts';
import {
  SMPL_VERTEX_COUNT,
  SMPL_JOINT_COUNT,
  SMPL_POSE_DIM,
  SMPL_SHAPE_DIM,
} from '../models/smpl.ts';

const DEPTH_H = 384;
const DEPTH_W = 384;

const depthLayout = computeBufferLayout([dim(DEPTH_H), dim(DEPTH_W)] as Shape2D, 'f32');
const verticesLayout = computeBufferLayout([dim(SMPL_VERTEX_COUNT), dim(3)] as Shape2D, 'f32');
const jointsLayout = computeBufferLayout([dim(SMPL_JOINT_COUNT), dim(3)] as Shape2D, 'f32');
const cameraInLayout = computeBufferLayout([dim(3)] as readonly [ReturnType<typeof dim>], 'f32');
const camera4x4Layout = computeBufferLayout([dim(4), dim(4)] as Shape2D, 'f32');

const INPUT_PORTS = [
  { name: 'depthMap', layout: depthLayout },
  { name: 'smplVertices', layout: verticesLayout },
  { name: 'jointPositions', layout: jointsLayout },
  { name: 'initialCamera', layout: cameraInLayout },
] as const satisfies readonly PortDescriptor[];

const OUTPUT_PORTS = [
  { name: 'optimizedDepth', layout: depthLayout },
  { name: 'refinedVertices', layout: verticesLayout },
  { name: 'cameraExtrinsics', layout: camera4x4Layout },
] as const satisfies readonly PortDescriptor[];

/**
 * Optimization parameter layout for the JOSH solver.
 *
 * Total parameters = SMPL_POSE_DIM + SMPL_SHAPE_DIM + 6 (camera) + 1 (depth scale)
 *                  = 72 + 10 + 6 + 1 = 89
 */
const PARAM_DIM = SMPL_POSE_DIM + SMPL_SHAPE_DIM + 6 + 1;

/** Maximum L-BFGS iterations per frame */
const MAX_ITERATIONS = 5;

/**
 * Node C: JOSH Solver — Hybrid WASM/WebGPU joint optimization.
 *
 * This node performs the core joint optimization from the JOSH paper:
 * - Scene geometry (depth map refinement)
 * - Camera pose trajectory
 * - Human motion (SMPL parameters)
 *
 * Architecture:
 * - L-BFGS optimizer runs in WASM (Rust) for numerical stability
 * - Gradient computation (contact loss, depth reprojection, temporal smoothness)
 *   dispatched as WebGPU compute shaders for parallelism
 * - SharedArrayBuffer bridges WASM ↔ GPU data transfer
 */
export class JoshSolverNode
  implements GComputeNode<typeof INPUT_PORTS, typeof OUTPUT_PORTS>
{
  readonly id: NodeId = createNodeId('joshSolver');
  readonly name = 'JOSHSolver';
  readonly backendHint = 'wasm' as const; // Primary backend is WASM (optimizer loop)
  readonly inputs = INPUT_PORTS;
  readonly outputs = OUTPUT_PORTS;

  private _wasmRunner: WasmKernelRunner | null = null;
  private _lbfgs: WasmHandle<WasmLBFGS> | null = null;
  private _contact: WasmHandle<WasmContact> | null = null;
  private _sharedMem: SharedMemoryBridge | null = null;

  // GPU buffers for gradient computation
  private _gradientBuffer: GPUBuffer | null = null;
  private _lossBuffer: GPUBuffer | null = null;

  async initialize(ctx: NodeContext): Promise<void> {
    // Initialize WASM backend
    this._wasmRunner = new WasmKernelRunner();
    this._lbfgs = await this._wasmRunner.createLBFGS(PARAM_DIM, 7, 1e-5);
    this._contact = await this._wasmRunner.createContactEvaluator(0.05);

    // SharedArrayBuffer for WASM ↔ GPU bridge
    // Holds: parameters (PARAM_DIM × 8 bytes) + gradients (PARAM_DIM × 8 bytes)
    this._sharedMem = new SharedMemoryBridge(PARAM_DIM * 8 * 2);

    // GPU buffers for gradient computation
    this._gradientBuffer = ctx.device.createBuffer({
      size: PARAM_DIM * 4, // f32
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      label: 'josh_gradients',
    });

    this._lossBuffer = ctx.device.createBuffer({
      size: 4 * 4, // 4 loss components as f32
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      label: 'josh_loss',
    });
  }

  async execute(ctx: ExecutionContext): Promise<void> {
    const depthIn = ctx.getInput('depthMap') as GPUBuffer;
    const verticesIn = ctx.getInput('smplVertices') as GPUBuffer;
    ctx.getInput('jointPositions'); // consume for port binding
    ctx.getInput('initialCamera'); // consume for port binding

    const depthOut = ctx.getOutput('optimizedDepth') as GPUBuffer;
    const verticesOut = ctx.getOutput('refinedVertices') as GPUBuffer;
    const cameraOut = ctx.getOutput('cameraExtrinsics') as GPUBuffer;

    // ─── Joint Optimization Loop ───
    // Each iteration:
    //   1. GPU: compute gradient components (contact, depth reprojection, smoothness)
    //   2. GPU→CPU: read back gradients via SharedArrayBuffer
    //   3. WASM: L-BFGS step (compute search direction)
    //   4. WASM: line search & parameter update
    //   5. CPU→GPU: upload new parameters

    for (let iter = 0; iter < MAX_ITERATIONS; iter++) {
      // Step 1: Compute gradients on GPU
      // TODO: dispatch gradient computation kernels:
      //   - Contact loss gradient (requires contribution slot #2: normal estimation)
      //   - Depth reprojection loss gradient (requires contribution slot #4: differentiable renderer)
      //   - Temporal smoothness loss gradient

      // Step 2: Read back gradients via SharedArrayBuffer bridge
      // In production: this._sharedMem bridges WASM↔GPU zero-copy
      // Contact evaluator: this._contact computes contact loss
      void this._sharedMem;
      void this._contact;
      const gradientData = await ctx.bufferManager.gpuToCpu(
        ctx.bufferManager.cpuToGpu(
          new Float32Array(PARAM_DIM).buffer,
          computeBufferLayout([dim(PARAM_DIM)] as readonly [ReturnType<typeof dim>], 'f32'),
        ),
      );

      // Step 3: L-BFGS step in WASM
      const gradient64 = new Float64Array(PARAM_DIM);
      const grad32 = new Float32Array(gradientData);
      for (let i = 0; i < PARAM_DIM; i++) {
        gradient64[i] = grad32[i]!;
      }

      const direction = this._wasmRunner!.lbfgsStep(gradient64);

      // Step 4: Simple line search (backtracking Armijo)
      const stepSize = 0.01;
      const currentParams = this._lbfgs!.inner.getParameters();
      const newParams = new Float64Array(PARAM_DIM);
      for (let i = 0; i < PARAM_DIM; i++) {
        newParams[i] = currentParams[i]! + stepSize * direction[i]!;
      }

      // Step 5: Update L-BFGS state
      const converged = this._wasmRunner!.lbfgsUpdate(newParams, gradient64);
      if (converged) break;
    }

    // Write optimized outputs
    // For now, pass through inputs as outputs (placeholder)
    ctx.commandEncoder.copyBufferToBuffer(depthIn, 0, depthOut, 0, depthLayout.byteLength);
    ctx.commandEncoder.copyBufferToBuffer(
      verticesIn,
      0,
      verticesOut,
      0,
      verticesLayout.byteLength,
    );
    ctx.commandEncoder.clearBuffer(cameraOut, 0, camera4x4Layout.byteLength);
  }

  dispose(): void {
    this._wasmRunner?.dispose();
    this._gradientBuffer?.destroy();
    this._lossBuffer?.destroy();
    this._wasmRunner = null;
    this._lbfgs = null;
    this._contact = null;
    this._sharedMem = null;
  }
}
