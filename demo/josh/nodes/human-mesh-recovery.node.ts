import type { GComputeNode, NodeContext, ExecutionContext, PortDescriptor } from '../../../src/graph/node.ts';
import type { NodeId, Shape2D, Shape3D } from '../../../src/core/types.ts';
import { createNodeId, computeBufferLayout, dim } from '../../../src/core/types.ts';
import {
  SMPL_VERTEX_COUNT,
  SMPL_JOINT_COUNT,
  SMPL_POSE_DIM,
  SMPL_KINEMATIC_TREE,
} from '../models/smpl.ts';
import { axisAngleToRotMat, getTposeJoints, buildSyntheticSmplModel } from '../models/smpl-synthetic.ts';
import { ROMPNode } from './romp.node.ts';

const HMR_H = 384;
const HMR_W = 384;

const rgbLayout = computeBufferLayout([dim(HMR_H), dim(HMR_W), dim(3)] as Shape3D, 'f32');
const verticesLayout = computeBufferLayout([dim(SMPL_VERTEX_COUNT), dim(3)] as Shape2D, 'f32');
const jointsLayout = computeBufferLayout([dim(SMPL_JOINT_COUNT), dim(3)] as Shape2D, 'f32');
const cameraLayout = computeBufferLayout(
  [dim(3)] as readonly [typeof dim extends (n: 3) => infer R ? R : never], 'f32',
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
 * CPU-side SMPL forward kinematics: axis-angle pose → 3D joint positions.
 * Uses the kinematic tree to compose transforms from root to leaves.
 */
function computeJointsFromPose(poseAxisAngle: Float32Array, tposeJoints: Float32Array): Float32Array {
  const joints = new Float32Array(SMPL_JOINT_COUNT * 3);
  // Global 4x4 transforms per joint
  const transforms = new Float64Array(SMPL_JOINT_COUNT * 16);

  for (let j = 0; j < SMPL_JOINT_COUNT; j++) {
    const rot = axisAngleToRotMat(
      poseAxisAngle[j * 3]!,
      poseAxisAngle[j * 3 + 1]!,
      poseAxisAngle[j * 3 + 2]!,
    );

    // Build local 4x4: [R | t; 0 0 0 1] column-major
    const local = new Float64Array(16);
    local[0] = rot[0]!; local[1] = rot[3]!; local[2] = rot[6]!; local[3] = 0;
    local[4] = rot[1]!; local[5] = rot[4]!; local[6] = rot[7]!; local[7] = 0;
    local[8] = rot[2]!; local[9] = rot[5]!; local[10] = rot[8]!; local[11] = 0;
    local[12] = tposeJoints[j * 3]!;
    local[13] = tposeJoints[j * 3 + 1]!;
    local[14] = tposeJoints[j * 3 + 2]!;
    local[15] = 1;

    const parent = SMPL_KINEMATIC_TREE[j]!;
    const tBase = j * 16;

    if (parent < 0) {
      transforms.set(local, tBase);
    } else {
      // Relative translation
      local[12] -= tposeJoints[parent * 3]!;
      local[13] -= tposeJoints[parent * 3 + 1]!;
      local[14] -= tposeJoints[parent * 3 + 2]!;
      // T_global = T_parent * T_local
      const pBase = parent * 16;
      for (let col = 0; col < 4; col++) {
        for (let row = 0; row < 4; row++) {
          let sum = 0;
          for (let k = 0; k < 4; k++) {
            sum += transforms[pBase + k * 4 + row]! * local[col * 4 + k]!;
          }
          transforms[tBase + col * 4 + row] = sum;
        }
      }
    }

    // Joint position = translation column of global transform
    joints[j * 3] = transforms[tBase + 12]!;
    joints[j * 3 + 1] = transforms[tBase + 13]!;
    joints[j * 3 + 2] = transforms[tBase + 14]!;
  }

  return joints;
}

/**
 * Node B: Human Mesh Recovery using BlazePose 3D (via ROMPNode) + CPU SMPL FK.
 *
 * BlazePose: image → 33 3D landmarks → SMPL axis-angle pose init
 * CPU FK: pose θ → joint positions (using kinematic tree)
 * Synthetic mesh: vertices from buildSyntheticSmplModel()
 *
 * Uses GPU readback to get the current frame as ImageData, then passes to
 * BlazePose via an OffscreenCanvas.
 */
export class HumanMeshRecoveryNode
  implements GComputeNode<typeof INPUT_PORTS, typeof OUTPUT_PORTS>
{
  readonly id: NodeId = createNodeId('humanMeshRecovery');
  readonly name = 'HumanMeshRecovery';
  readonly backendHint = 'webgpu' as const;
  readonly inputs = INPUT_PORTS;
  readonly outputs = OUTPUT_PORTS;

  private _device: GPUDevice | null = null;
  private _rompNode = new ROMPNode();
  private _useSimulation = false;
  private _tposeJoints: Float32Array | null = null;

  private _verticesOut = new Float32Array(SMPL_VERTEX_COUNT * 3);
  private _jointsOut = new Float32Array(SMPL_JOINT_COUNT * 3);

  async initialize(ctx: NodeContext): Promise<void> {
    this._device = ctx.device;
    this._tposeJoints = getTposeJoints();

    const synth = buildSyntheticSmplModel();
    this._verticesOut.set(synth.meanTemplate.subarray(0, this._verticesOut.length));

    const status: ((id: string, s: string, t: string) => void) | undefined =
      (globalThis as any).__joshLoadingStatus;

    status?.('hmrModel', 'active', 'Node B: Loading BlazePose 3D...');
    try {
      await this._rompNode.load();
      status?.('hmrModel', 'done', 'Node B: BlazePose 3D ready');
      status?.('smplModel', 'done', 'Node B: SMPL forward kinematics (CPU)');
    } catch (e) {
      console.warn('[HMR] BlazePose load failed, using simulated animation:', e);
      status?.('hmrModel', 'warn', 'Node B: BlazePose unavailable — using simulated pose');
      this._useSimulation = true;
    }
  }

  async execute(ctx: ExecutionContext): Promise<void> {
    const inputBuffer = ctx.getInput('rgbFrame') as GPUBuffer;
    const verticesOut = ctx.getOutput('smplVertices') as GPUBuffer;
    const jointsOut = ctx.getOutput('jointPositions') as GPUBuffer;
    const cameraOut = ctx.getOutput('estimatedCamera') as GPUBuffer;

    if (this._useSimulation) {
      this._runSimulation(ctx.frameIndex, verticesOut, jointsOut, cameraOut);
      return;
    }

    // Throttle ROMP: only run inference every 10 frames (it's ~500ms per run on WASM)
    // Reuse last result for intermediate frames
    if (ctx.frameIndex % 10 !== 0 && ctx.frameIndex > 1) {
      this._device!.queue.writeBuffer(verticesOut, 0, this._verticesOut);
      this._device!.queue.writeBuffer(jointsOut, 0, this._jointsOut);
      this._device!.queue.writeBuffer(cameraOut, 0, new Float32Array([1.0, 0.0, 0.0]));
      return;
    }

    try {
      await this._runInference(inputBuffer, verticesOut, jointsOut, cameraOut);
    } catch (e) {
      console.error('[HMR] Inference error, falling back to simulation:', e);
      this._useSimulation = true;
      this._runSimulation(ctx.frameIndex, verticesOut, jointsOut, cameraOut);
    }
  }

  private async _runInference(
    inputBuffer: GPUBuffer,
    verticesOut: GPUBuffer,
    jointsOut: GPUBuffer,
    cameraOut: GPUBuffer,
  ): Promise<void> {
    // GPU readback → Float32Array [HMR_H * HMR_W * 3] (values in [0,1])
    const readbackEncoder = this._device!.createCommandEncoder({ label: 'hmr_readback' });
    const staging = this._device!.createBuffer({
      size: rgbLayout.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: 'hmr_staging',
    });
    readbackEncoder.copyBufferToBuffer(inputBuffer, 0, staging, 0, rgbLayout.byteLength);
    this._device!.queue.submit([readbackEncoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const rgbData = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();

    // Convert float32 [0,1] → Uint8ClampedArray RGBA for ImageData
    const rgba = new Uint8ClampedArray(HMR_W * HMR_H * 4);
    for (let i = 0; i < HMR_W * HMR_H; i++) {
      rgba[i * 4]     = Math.round((rgbData[i * 3]     ?? 0) * 255);
      rgba[i * 4 + 1] = Math.round((rgbData[i * 3 + 1] ?? 0) * 255);
      rgba[i * 4 + 2] = Math.round((rgbData[i * 3 + 2] ?? 0) * 255);
      rgba[i * 4 + 3] = 255;
    }

    // Pass ImageData directly to BlazePose (supported input type)
    const imageData = new ImageData(rgba, HMR_W, HMR_H);

    const result = await this._rompNode.estimate(imageData);

    const camera = new Float32Array(3);
    const poseAxisAngle = new Float32Array(SMPL_POSE_DIM);

    if (result && result.confidence > 0.2) {
      poseAxisAngle.set(result.pose);
      camera.set(result.cam);
    } else {
      // Low confidence — keep T-pose
      camera[0] = 1.0;
    }

    const joints = computeJointsFromPose(poseAxisAngle, this._tposeJoints!);
    this._jointsOut.set(joints);

    this._device!.queue.writeBuffer(verticesOut, 0, this._verticesOut);
    this._device!.queue.writeBuffer(jointsOut, 0, this._jointsOut);
    this._device!.queue.writeBuffer(cameraOut, 0, camera);
  }

  private _runSimulation(
    frameIndex: number,
    verticesOut: GPUBuffer,
    jointsOut: GPUBuffer,
    cameraOut: GPUBuffer,
  ): void {
    const t = frameIndex * 0.05;
    const walkPhase = Math.sin(t);
    const walkAmp = 0.3;

    const tpose = getTposeJoints();
    this._jointsOut.set(tpose);
    this._jointsOut[1 * 3 + 2] = walkPhase * walkAmp * 0.2;
    this._jointsOut[2 * 3 + 2] = -walkPhase * walkAmp * 0.2;
    this._jointsOut[4 * 3 + 1] = 0.45 + Math.max(0, -walkPhase) * walkAmp * 0.3;
    this._jointsOut[5 * 3 + 1] = 0.45 + Math.max(0, walkPhase) * walkAmp * 0.3;

    this._device!.queue.writeBuffer(verticesOut, 0, this._verticesOut);
    this._device!.queue.writeBuffer(jointsOut, 0, this._jointsOut);
    this._device!.queue.writeBuffer(cameraOut, 0, new Float32Array([1.0, 0.0, 0.0]));
  }

  dispose(): void {
    this._rompNode.dispose();
  }
}
