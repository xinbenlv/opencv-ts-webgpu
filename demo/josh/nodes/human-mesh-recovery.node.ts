import type { GComputeNode, NodeContext, ExecutionContext, PortDescriptor } from '../../../src/graph/node.ts';
import type { NodeId, Shape2D, Shape3D } from '../../../src/core/types.ts';
import { createNodeId, computeBufferLayout, dim } from '../../../src/core/types.ts';
import { KernelRunner } from '../../../src/backends/webgpu/kernel-runner.ts';
import {
  SMPL_VERTEX_COUNT,
  SMPL_JOINT_COUNT,
  SMPL_POSE_DIM,
  SMPL_SHAPE_DIM,
} from '../models/smpl.ts';
import { buildSyntheticSmplModel, axisAngleToRotMat, getKinematicTreeI32 } from '../models/smpl-synthetic.ts';
import { smplJointsKernel, smplForwardKernel } from '../kernels/smpl.kernel.ts';

const HMR_H = 384;
const HMR_W = 384;

const rgbLayout = computeBufferLayout(
  [dim(HMR_H), dim(HMR_W), dim(3)] as Shape3D,
  'f32',
);

const verticesLayout = computeBufferLayout(
  [dim(SMPL_VERTEX_COUNT), dim(3)] as Shape2D,
  'f32',
);

const jointsLayout = computeBufferLayout(
  [dim(SMPL_JOINT_COUNT), dim(3)] as Shape2D,
  'f32',
);

const cameraLayout = computeBufferLayout(
  [dim(3)] as readonly [typeof dim extends (n: 3) => infer R ? R : never],
  'f32',
);

const INPUT_PORTS = [
  { name: 'rgbFrame', layout: rgbLayout },
] as const satisfies readonly PortDescriptor[];

const OUTPUT_PORTS = [
  { name: 'smplVertices', layout: verticesLayout },
  { name: 'jointPositions', layout: jointsLayout },
  { name: 'estimatedCamera', layout: cameraLayout },
] as const satisfies readonly PortDescriptor[];

/**
 * Node B: Human Mesh Recovery with SMPL Forward Pass on GPU.
 *
 * Pipeline:
 * 1. Generate plausible pose parameters (CPU — simulated HMR)
 * 2. GPU: Joint regression + forward kinematics (smplJointsKernel)
 * 3. GPU: Shape blend + Linear Blend Skinning (smplForwardKernel)
 */
export class HumanMeshRecoveryNode
  implements GComputeNode<typeof INPUT_PORTS, typeof OUTPUT_PORTS>
{
  readonly id: NodeId = createNodeId('humanMeshRecovery');
  readonly name = 'HumanMeshRecovery';
  readonly backendHint = 'webgpu' as const;
  readonly inputs = INPUT_PORTS;
  readonly outputs = OUTPUT_PORTS;

  private _kernelRunner: KernelRunner | null = null;
  private _device: GPUDevice | null = null;

  // GPU buffers for SMPL model data (persistent)
  private _meanTemplateBuffer: GPUBuffer | null = null;
  private _shapeBlendBuffer: GPUBuffer | null = null;
  private _skinWeightsBuffer: GPUBuffer | null = null;
  private _skinIndicesBuffer: GPUBuffer | null = null;
  private _jointRegressorBuffer: GPUBuffer | null = null;
  private _parentIndicesBuffer: GPUBuffer | null = null;

  // GPU buffers for per-frame data
  private _shapedVerticesBuffer: GPUBuffer | null = null;
  private _localRotationsBuffer: GPUBuffer | null = null;
  private _shapeParamsBuffer: GPUBuffer | null = null;
  private _jointTransformsBuffer: GPUBuffer | null = null;
  private _jointPositionsBuffer: GPUBuffer | null = null;

  // CPU-side pose state
  private _poseParams = new Float32Array(SMPL_POSE_DIM);
  private _shapeParams = new Float32Array(SMPL_SHAPE_DIM);

  async initialize(ctx: NodeContext): Promise<void> {
    this._device = ctx.device;
    this._kernelRunner = new KernelRunner(ctx.device);

    const smpl = buildSyntheticSmplModel();

    this._meanTemplateBuffer = this._createStorageBuffer(ctx.device, smpl.meanTemplate, 'smpl_mean_template');
    this._shapeBlendBuffer = this._createStorageBuffer(ctx.device, smpl.shapeBlendShapes, 'smpl_shape_blends');
    this._skinWeightsBuffer = this._createStorageBuffer(ctx.device, smpl.skinningWeights, 'smpl_skin_weights');
    this._skinIndicesBuffer = this._createStorageBuffer(ctx.device, smpl.skinningIndices, 'smpl_skin_indices');
    this._jointRegressorBuffer = this._createStorageBuffer(ctx.device, smpl.jointRegressor, 'smpl_joint_regressor');
    this._parentIndicesBuffer = this._createStorageBuffer(ctx.device, getKinematicTreeI32(), 'smpl_parent_indices');

    this._shapedVerticesBuffer = ctx.device.createBuffer({
      size: SMPL_VERTEX_COUNT * 3 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      label: 'smpl_shaped_vertices',
    });

    this._localRotationsBuffer = ctx.device.createBuffer({
      size: SMPL_JOINT_COUNT * 9 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'smpl_local_rotations',
    });

    this._shapeParamsBuffer = ctx.device.createBuffer({
      size: SMPL_SHAPE_DIM * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'smpl_shape_params',
    });

    this._jointTransformsBuffer = ctx.device.createBuffer({
      size: SMPL_JOINT_COUNT * 16 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      label: 'smpl_joint_transforms',
    });

    this._jointPositionsBuffer = ctx.device.createBuffer({
      size: SMPL_JOINT_COUNT * 3 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      label: 'smpl_joint_positions',
    });

    console.log('[HMR] Initialized with synthetic SMPL model');
  }

  async execute(ctx: ExecutionContext): Promise<void> {
    ctx.getInput('rgbFrame');
    const verticesOut = ctx.getOutput('smplVertices') as GPUBuffer;
    const jointsOut = ctx.getOutput('jointPositions') as GPUBuffer;
    const cameraOut = ctx.getOutput('estimatedCamera') as GPUBuffer;

    // Step 1: Generate pose parameters (simulated HMR)
    this._updatePoseFromFrame(ctx.frameIndex);

    // Step 2: Convert axis-angle to rotation matrices and upload
    const localRotations = new Float32Array(SMPL_JOINT_COUNT * 9);
    for (let j = 0; j < SMPL_JOINT_COUNT; j++) {
      const rotMat = axisAngleToRotMat(
        this._poseParams[j * 3]!,
        this._poseParams[j * 3 + 1]!,
        this._poseParams[j * 3 + 2]!,
      );
      localRotations.set(rotMat, j * 9);
    }

    this._device!.queue.writeBuffer(this._localRotationsBuffer!, 0, localRotations);
    this._device!.queue.writeBuffer(this._shapeParamsBuffer!, 0, this._shapeParams);

    // Copy mean template to shaped vertices for joint regression
    ctx.commandEncoder.copyBufferToBuffer(
      this._meanTemplateBuffer!, 0,
      this._shapedVerticesBuffer!, 0,
      SMPL_VERTEX_COUNT * 3 * 4,
    );

    // Step 3: GPU — Joint regression + forward kinematics
    const jointsUniforms = new ArrayBuffer(16);
    const jv = new DataView(jointsUniforms);
    jv.setUint32(0, SMPL_VERTEX_COUNT, true);
    jv.setUint32(4, SMPL_JOINT_COUNT, true);
    jv.setUint32(8, 0, true);
    jv.setUint32(12, 0, true);

    await this._kernelRunner!.dispatch(
      ctx.commandEncoder,
      smplJointsKernel,
      [
        this._shapedVerticesBuffer!,
        this._jointRegressorBuffer!,
        this._localRotationsBuffer!,
        this._parentIndicesBuffer!,
        this._jointTransformsBuffer!,
        this._jointPositionsBuffer!,
      ],
      [SMPL_VERTEX_COUNT],
      jointsUniforms,
    );

    // Step 4: GPU — Shape blend + LBS
    const forwardUniforms = new ArrayBuffer(16);
    const fv = new DataView(forwardUniforms);
    fv.setUint32(0, SMPL_VERTEX_COUNT, true);
    fv.setUint32(4, SMPL_JOINT_COUNT, true);
    fv.setUint32(8, SMPL_SHAPE_DIM, true);
    fv.setUint32(12, 0, true);

    await this._kernelRunner!.dispatch(
      ctx.commandEncoder,
      smplForwardKernel,
      [
        this._meanTemplateBuffer!,
        this._shapeBlendBuffer!,
        this._skinWeightsBuffer!,
        this._skinIndicesBuffer!,
        this._jointTransformsBuffer!,
        this._shapeParamsBuffer!,
        verticesOut,
      ],
      [SMPL_VERTEX_COUNT],
      forwardUniforms,
    );

    // Copy joint positions to output
    ctx.commandEncoder.copyBufferToBuffer(
      this._jointPositionsBuffer!, 0,
      jointsOut, 0,
      SMPL_JOINT_COUNT * 3 * 4,
    );

    // Camera params (weak-perspective: [scale, tx, ty])
    const camData = new Float32Array([1.0, 0.0, 0.0]);
    this._device!.queue.writeBuffer(cameraOut, 0, camData);
  }

  /**
   * Simulate plausible human motion — walking + breathing cycle.
   * In production, replace with ONNX HMR inference.
   */
  private _updatePoseFromFrame(frameIndex: number): void {
    const t = frameIndex * 0.05;
    this._poseParams.fill(0);

    const walkPhase = Math.sin(t);
    const walkAmp = 0.3;

    // Hip flexion (walking)
    this._poseParams[1 * 3]  =  walkPhase * walkAmp;
    this._poseParams[2 * 3]  = -walkPhase * walkAmp;

    // Knee flexion
    this._poseParams[4 * 3]  = Math.max(0, -walkPhase) * walkAmp * 0.8;
    this._poseParams[5 * 3]  = Math.max(0,  walkPhase) * walkAmp * 0.8;

    // Arm swing (opposite to legs)
    this._poseParams[16 * 3] = -walkPhase * walkAmp * 0.4;
    this._poseParams[17 * 3] =  walkPhase * walkAmp * 0.4;

    // Elbow bend
    this._poseParams[18 * 3 + 2] = -0.3;
    this._poseParams[19 * 3 + 2] =  0.3;

    // Spine twist
    this._poseParams[3 * 3 + 1] = walkPhase * 0.05;
    this._poseParams[6 * 3 + 1] = walkPhase * 0.03;

    // Breathing
    this._shapeParams[0] = Math.sin(t * 0.3) * 0.02;
  }

  private _createStorageBuffer(device: GPUDevice, data: Float32Array | Int32Array | Uint32Array, label: string): GPUBuffer {
    const buffer = device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      label,
    });
    device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
    return buffer;
  }

  dispose(): void {
    this._meanTemplateBuffer?.destroy();
    this._shapeBlendBuffer?.destroy();
    this._skinWeightsBuffer?.destroy();
    this._skinIndicesBuffer?.destroy();
    this._jointRegressorBuffer?.destroy();
    this._parentIndicesBuffer?.destroy();
    this._shapedVerticesBuffer?.destroy();
    this._localRotationsBuffer?.destroy();
    this._shapeParamsBuffer?.destroy();
    this._jointTransformsBuffer?.destroy();
    this._jointPositionsBuffer?.destroy();
    this._kernelRunner?.clearCache();
  }
}
