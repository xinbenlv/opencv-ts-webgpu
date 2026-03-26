import type { GComputeNode, NodeContext, ExecutionContext, PortDescriptor } from '../../../src/graph/node.ts';
import type { NodeId, Shape2D, Shape3D } from '../../../src/core/types.ts';
import { createNodeId, computeBufferLayout, dim } from '../../../src/core/types.ts';
import {
  SMPL_VERTEX_COUNT,
  SMPL_JOINT_COUNT,
  SMPL_POSE_DIM,
  SMPL_KINEMATIC_TREE,
} from '../models/smpl.ts';
import { cachedFetchModel } from '../models/cached-fetch.ts';
import { axisAngleToRotMat, getTposeJoints, buildSyntheticSmplModel } from '../models/smpl-synthetic.ts';

const HMR_H = 384;
const HMR_W = 384;
const ROMP_H = 512;
const ROMP_W = 512;

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

let ortModule: typeof import('onnxruntime-web') | null = null;
async function getOrt() {
  if (!ortModule) ortModule = await import('onnxruntime-web');
  return ortModule;
}

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
 * Node B: Human Mesh Recovery using ROMP ONNX + CPU SMPL forward kinematics.
 *
 * ROMP: image → SMPL params (pose θ, shape β, camera)
 * CPU FK: pose θ → joint positions (using kinematic tree)
 * Synthetic mesh: vertices from buildSyntheticSmplModel()
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
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private _rompSession: any = null;
  private _useSimulation = false;
  private _tposeJoints: Float32Array | null = null;

  private _rompInput = new Float32Array(1 * ROMP_H * ROMP_W * 3);
  private _verticesOut = new Float32Array(SMPL_VERTEX_COUNT * 3);
  private _jointsOut = new Float32Array(SMPL_JOINT_COUNT * 3);

  constructor(
    private readonly _rompModelUrl = './assets/models/romp.onnx',
  ) {}

  async initialize(ctx: NodeContext): Promise<void> {
    this._device = ctx.device;
    this._tposeJoints = getTposeJoints();

    // Pre-fill vertices from synthetic model (used for mesh output)
    const synth = buildSyntheticSmplModel();
    this._verticesOut.set(synth.meanTemplate.subarray(0, this._verticesOut.length));

    const ort = await getOrt();
    ort.env.wasm.wasmPaths = './assets/ort/';
    ort.env.wasm.numThreads = 1;

    const status: ((id: string, s: string, t: string) => void) | undefined = (globalThis as any).__joshLoadingStatus;

    try {
      const rompBuf = await cachedFetchModel(
        this._rompModelUrl, 'hmrModel', 'Node B: ROMP pose model', '111 MB', status,
      );
      status?.('hmrModel', 'active', 'Node B: Creating ROMP session...');
      this._rompSession = await ort.InferenceSession.create(rompBuf, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      status?.('hmrModel', 'done', 'Node B: ROMP pose model ready');
      status?.('smplModel', 'done', 'Node B: SMPL forward kinematics (CPU)');
      console.log('[HMR] ROMP loaded:', this._rompSession.inputNames, '→', this._rompSession.outputNames);
    } catch (e) {
      console.warn('[HMR] ROMP loading failed, using simulated animation:', e);
      status?.('hmrModel', 'warn', 'Node B: ROMP unavailable — using simulated pose');
      status?.('smplModel', 'warn', 'Node B: Using simulated mesh');
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
    // GPU readback
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

    // Resize 384→512 NHWC
    for (let y = 0; y < ROMP_H; y++) {
      for (let x = 0; x < ROMP_W; x++) {
        const x0 = Math.min(Math.floor((x / ROMP_W) * HMR_W), HMR_W - 1);
        const y0 = Math.min(Math.floor((y / ROMP_H) * HMR_H), HMR_H - 1);
        const srcIdx = (y0 * HMR_W + x0) * 3;
        const dstIdx = (y * ROMP_W + x) * 3;
        this._rompInput[dstIdx] = rgbData[srcIdx]!;
        this._rompInput[dstIdx + 1] = rgbData[srcIdx + 1]!;
        this._rompInput[dstIdx + 2] = rgbData[srcIdx + 2]!;
      }
    }

    // ROMP inference
    const ort = await getOrt();
    const rompTensor = new ort.Tensor('float32', this._rompInput, [1, ROMP_H, ROMP_W, 3]);
    const rompFeeds: Record<string, InstanceType<typeof ort.Tensor>> = {};
    rompFeeds[this._rompSession.inputNames[0]!] = rompTensor;
    const rompResults = await this._rompSession.run(rompFeeds);
    const centerMaps = rompResults[this._rompSession.outputNames[0]!]!.data as Float32Array;
    const paramsMaps = rompResults[this._rompSession.outputNames[1]!]!.data as Float32Array;

    // Find best detection
    const mapSize = 64;
    let bestIdx = 0;
    let bestVal = -Infinity;
    for (let i = 0; i < mapSize * mapSize; i++) {
      if (centerMaps[i]! > bestVal) {
        bestVal = centerMaps[i]!;
        bestIdx = i;
      }
    }

    // Extract SMPL params: [0-2] camera, [3-74] pose (72), [75-84] shape (10)
    const camera = new Float32Array(3);
    const poseAxisAngle = new Float32Array(SMPL_POSE_DIM);

    for (let c = 0; c < 3; c++) {
      camera[c] = paramsMaps[c * mapSize * mapSize + bestIdx]!;
    }
    for (let c = 0; c < SMPL_POSE_DIM; c++) {
      poseAxisAngle[c] = paramsMaps[(c + 3) * mapSize * mapSize + bestIdx]!;
    }

    // CPU SMPL forward kinematics: pose → joints
    const joints = computeJointsFromPose(poseAxisAngle, this._tposeJoints!);
    this._jointsOut.set(joints);

    // Upload to GPU
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
    this._rompSession?.release();
  }
}
