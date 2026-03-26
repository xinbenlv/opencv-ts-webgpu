import type { GComputeNode, NodeContext, ExecutionContext, PortDescriptor } from '../../../src/graph/node.ts';
import type { NodeId, Shape2D, Shape3D } from '../../../src/core/types.ts';
import { createNodeId, computeBufferLayout, dim } from '../../../src/core/types.ts';
import {
  SMPL_VERTEX_COUNT,
  SMPL_JOINT_COUNT,
  SMPL_POSE_DIM,
  SMPL_SHAPE_DIM,
} from '../models/smpl.ts';

const HMR_H = 384;
const HMR_W = 384;

/** ROMP expects 512×512 input */
const ROMP_H = 512;
const ROMP_W = 512;

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

let ortModule: typeof import('onnxruntime-web') | null = null;
async function getOrt() {
  if (!ortModule) ortModule = await import('onnxruntime-web');
  return ortModule;
}

/**
 * Node B: Human Mesh Recovery using ROMP + nosmpl ONNX models.
 *
 * Pipeline:
 * 1. GPU readback → CPU: read RGB frame
 * 2. CPU: resize 384→512, prepare ROMP input [1,512,512,3]
 * 3. ROMP inference → center_maps + params_maps (SMPL params per pixel)
 * 4. Post-process: find best detection, extract pose θ (72), shape β (10), camera (3)
 * 5. nosmpl inference: SMPL params → vertices [6890,3] + joints [45,3]
 * 6. Upload vertices + joints to GPU output buffers
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

  // ONNX sessions
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private _rompSession: any = null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private _smplSession: any = null;

  // Reusable CPU arrays
  private _rompInput = new Float32Array(1 * ROMP_H * ROMP_W * 3);
  private _verticesOut = new Float32Array(SMPL_VERTEX_COUNT * 3);
  private _jointsOut = new Float32Array(SMPL_JOINT_COUNT * 3);

  constructor(
    private readonly _rompModelUrl = './assets/models/romp.onnx',
    private readonly _smplModelUrl = './assets/models/smpl_sim.onnx',
  ) {}

  async initialize(ctx: NodeContext): Promise<void> {
    this._device = ctx.device;

    const ort = await getOrt();
    ort.env.wasm.wasmPaths = './assets/ort/';
    ort.env.wasm.numThreads = 1;

    // Load ROMP model
    console.log('[HMR] Loading ROMP model...');
    try {
      this._rompSession = await ort.InferenceSession.create(this._rompModelUrl, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      console.log('[HMR] ROMP loaded:', this._rompSession.inputNames, '→', this._rompSession.outputNames);
    } catch (e) {
      console.error('[HMR] Failed to load ROMP:', e);
    }

    // Load SMPL forward pass model
    console.log('[HMR] Loading SMPL model...');
    try {
      this._smplSession = await ort.InferenceSession.create(this._smplModelUrl, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      console.log('[HMR] SMPL loaded:', this._smplSession.inputNames, '→', this._smplSession.outputNames);
    } catch (e) {
      console.error('[HMR] Failed to load SMPL:', e);
    }
  }

  async execute(ctx: ExecutionContext): Promise<void> {
    const inputBuffer = ctx.getInput('rgbFrame') as GPUBuffer;
    const verticesOut = ctx.getOutput('smplVertices') as GPUBuffer;
    const jointsOut = ctx.getOutput('jointPositions') as GPUBuffer;
    const cameraOut = ctx.getOutput('estimatedCamera') as GPUBuffer;

    if (!this._rompSession || !this._smplSession) {
      ctx.commandEncoder.clearBuffer(verticesOut, 0, verticesLayout.byteLength);
      ctx.commandEncoder.clearBuffer(jointsOut, 0, jointsLayout.byteLength);
      ctx.commandEncoder.clearBuffer(cameraOut, 0, cameraLayout.byteLength);
      return;
    }

    // Step 1: GPU readback
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

    // Step 2: Resize 384→512 for ROMP, keep NHWC format [1,512,512,3]
    for (let y = 0; y < ROMP_H; y++) {
      for (let x = 0; x < ROMP_W; x++) {
        const srcX = (x / ROMP_W) * HMR_W;
        const srcY = (y / ROMP_H) * HMR_H;
        const x0 = Math.min(Math.floor(srcX), HMR_W - 1);
        const y0 = Math.min(Math.floor(srcY), HMR_H - 1);
        const srcIdx = (y0 * HMR_W + x0) * 3;
        const dstIdx = (y * ROMP_W + x) * 3;
        // ROMP expects [0,1] float RGB in NHWC
        this._rompInput[dstIdx] = rgbData[srcIdx]!;
        this._rompInput[dstIdx + 1] = rgbData[srcIdx + 1]!;
        this._rompInput[dstIdx + 2] = rgbData[srcIdx + 2]!;
      }
    }

    // Step 3: ROMP inference
    const ort = await getOrt();
    const rompTensor = new ort.Tensor('float32', this._rompInput, [1, ROMP_H, ROMP_W, 3]);
    const rompFeeds: Record<string, InstanceType<typeof ort.Tensor>> = {};
    rompFeeds[this._rompSession.inputNames[0]!] = rompTensor;

    const rompResults = await this._rompSession.run(rompFeeds);
    const centerMaps = rompResults[this._rompSession.outputNames[0]!]!.data as Float32Array;
    const paramsMaps = rompResults[this._rompSession.outputNames[1]!]!.data as Float32Array;

    // Step 4: Post-process — find pixel with highest center response
    const mapSize = 64;
    let bestIdx = 0;
    let bestVal = -Infinity;
    for (let i = 0; i < mapSize * mapSize; i++) {
      if (centerMaps[i]! > bestVal) {
        bestVal = centerMaps[i]!;
        bestIdx = i;
      }
    }

    // Extract SMPL params at best detection location
    // params_maps layout: [1, 145, 64, 64] — 145 channels at each spatial location
    // Channel layout: [0-2] camera, [3-74] pose (72), [75-84] shape (10), [85-144] extra
    const camera = new Float32Array(3);
    const poseAxisAngle = new Float32Array(SMPL_POSE_DIM); // 72
    const shape = new Float32Array(SMPL_SHAPE_DIM); // 10

    for (let c = 0; c < 3; c++) {
      camera[c] = paramsMaps[c * mapSize * mapSize + bestIdx]!;
    }
    for (let c = 0; c < SMPL_POSE_DIM; c++) {
      poseAxisAngle[c] = paramsMaps[(c + 3) * mapSize * mapSize + bestIdx]!;
    }
    for (let c = 0; c < SMPL_SHAPE_DIM; c++) {
      shape[c] = paramsMaps[(c + 75) * mapSize * mapSize + bestIdx]!;
    }

    // Step 5: SMPL forward pass — convert axis-angle to rotation format
    // nosmpl expects global_orient [1,1,3] and body [1,23,3]
    const globalOrient = new Float32Array(3);
    globalOrient[0] = poseAxisAngle[0]!;
    globalOrient[1] = poseAxisAngle[1]!;
    globalOrient[2] = poseAxisAngle[2]!;

    const bodyPose = new Float32Array(23 * 3);
    for (let i = 0; i < 23 * 3; i++) {
      bodyPose[i] = poseAxisAngle[i + 3]!;
    }

    const orientTensor = new ort.Tensor('float32', globalOrient, [1, 1, 3]);
    const bodyTensor = new ort.Tensor('float32', bodyPose, [1, 23, 3]);
    const smplFeeds: Record<string, InstanceType<typeof ort.Tensor>> = {};
    smplFeeds[this._smplSession.inputNames[0]!] = orientTensor;
    smplFeeds[this._smplSession.inputNames[1]!] = bodyTensor;

    const smplResults = await this._smplSession.run(smplFeeds);
    const vertices = smplResults['vertices']!.data as Float32Array;
    const joints = smplResults['joints']!.data as Float32Array;

    // Step 6: Copy to output arrays (truncate joints from 45 to 24 for SMPL)
    const numVerts = Math.min(vertices.length, this._verticesOut.length);
    this._verticesOut.set(vertices.subarray(0, numVerts));

    // Copy first 24 joints (SMPL joints from 45 total)
    for (let j = 0; j < SMPL_JOINT_COUNT; j++) {
      this._jointsOut[j * 3] = joints[j * 3]!;
      this._jointsOut[j * 3 + 1] = joints[j * 3 + 1]!;
      this._jointsOut[j * 3 + 2] = joints[j * 3 + 2]!;
    }

    // Upload to GPU
    this._device!.queue.writeBuffer(verticesOut, 0, this._verticesOut);
    this._device!.queue.writeBuffer(jointsOut, 0, this._jointsOut);
    this._device!.queue.writeBuffer(cameraOut, 0, camera);
  }

  dispose(): void {
    this._rompSession?.release();
    this._smplSession?.release();
  }
}
