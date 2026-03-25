import type { GComputeNode, NodeContext, ExecutionContext, PortDescriptor } from '../../../graph/node.ts';
import type { NodeId, Shape2D, Shape3D } from '../../../core/types.ts';
import { createNodeId, computeBufferLayout, dim } from '../../../core/types.ts';
import {
  SMPL_VERTEX_COUNT,
  SMPL_JOINT_COUNT,
  SMPL_POSE_DIM,
  SMPL_SHAPE_DIM,
} from '../models/smpl.ts';

// Resolution matching depth estimation node
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
 * Node B: WebGPU-accelerated Human Mesh Recovery (SMPL).
 *
 * Pipeline:
 * 1. Regress SMPL parameters (pose θ: 72-d, shape β: 10-d) from RGB
 * 2. Forward kinematics → joint transforms
 * 3. Linear blend skinning → posed mesh vertices
 *
 * All buffers remain device-resident for zero CPU-GPU transfer.
 */
export class HumanMeshRecoveryNode
  implements GComputeNode<typeof INPUT_PORTS, typeof OUTPUT_PORTS>
{
  readonly id: NodeId = createNodeId('humanMeshRecovery');
  readonly name = 'HumanMeshRecovery';
  readonly backendHint = 'webgpu' as const;
  readonly inputs = INPUT_PORTS;
  readonly outputs = OUTPUT_PORTS;

  // Internal GPU buffers for SMPL parameters
  private _poseBuffer: GPUBuffer | null = null;
  private _shapeBuffer: GPUBuffer | null = null;
  private _jointTransformsBuffer: GPUBuffer | null = null;

  async initialize(ctx: NodeContext): Promise<void> {
    const device = ctx.device;

    // SMPL pose parameters (24 joints × 3 axis-angle = 72 floats)
    this._poseBuffer = device.createBuffer({
      size: SMPL_POSE_DIM * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'smpl_pose',
    });

    // SMPL shape parameters (10 PCA components)
    this._shapeBuffer = device.createBuffer({
      size: SMPL_SHAPE_DIM * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'smpl_shape',
    });

    // Joint transforms (24 joints × 4×4 matrix = 384 floats)
    this._jointTransformsBuffer = device.createBuffer({
      size: SMPL_JOINT_COUNT * 16 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'smpl_joint_transforms',
    });
  }

  async execute(ctx: ExecutionContext): Promise<void> {
    ctx.getInput('rgbFrame'); // consume input to satisfy port binding
    const verticesOut = ctx.getOutput('smplVertices') as GPUBuffer;
    const jointsOut = ctx.getOutput('jointPositions') as GPUBuffer;
    const cameraOut = ctx.getOutput('estimatedCamera') as GPUBuffer;

    // Step 1: HMR regression network (onnxruntime-web WebGPU EP)
    // Produces pose θ, shape β, camera params
    // TODO: integrate onnxruntime-web inference session

    // Step 2: SMPL forward pass as WGSL compute kernels:
    //   a) Blend shape deformation: V_shaped = meanTemplate + β · shapeBlendShapes
    //   b) Joint regression: J = jointRegressor · V_shaped
    //   c) Forward kinematics: compute per-joint 4×4 transforms
    //   d) Linear blend skinning: V_posed = Σ w_i · T_i · V_shaped
    // TODO: dispatch SMPL LBS kernel (contribution slot #3)

    // Step 3: Extract joint positions from transforms → jointsOut
    // Step 4: Estimate weak-perspective camera → cameraOut

    // Placeholder: zero-fill outputs to maintain pipeline integrity
    const encoder = ctx.commandEncoder;
    encoder.clearBuffer(verticesOut, 0, verticesLayout.byteLength);
    encoder.clearBuffer(jointsOut, 0, jointsLayout.byteLength);
    encoder.clearBuffer(cameraOut, 0, cameraLayout.byteLength);
  }

  dispose(): void {
    this._poseBuffer?.destroy();
    this._shapeBuffer?.destroy();
    this._jointTransformsBuffer?.destroy();
  }
}
