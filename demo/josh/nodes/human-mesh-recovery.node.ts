import type { GComputeNode, NodeContext, ExecutionContext, PortDescriptor } from '../../../src/graph/node.ts';
import type { NodeId, Shape2D, Shape3D } from '../../../src/core/types.ts';
import { createNodeId, computeBufferLayout, dim } from '../../../src/core/types.ts';
import {
  SMPL_VERTEX_COUNT,
  SMPL_JOINT_COUNT,
  SMPL_POSE_DIM,
} from '../models/smpl.ts';

const HMR_H = 384;
const HMR_W = 384;
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
 * Fetch model as ArrayBuffer with retry logic for large files over slow networks.
 */
async function fetchModelBuffer(url: string, retries = 3): Promise<ArrayBuffer> {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      console.log(`[HMR] Fetch attempt ${attempt}/${retries}: ${url}`);
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status} fetching ${url}`);
      return await response.arrayBuffer();
    } catch (e) {
      if (attempt === retries) throw e;
      console.warn(`[HMR] Fetch attempt ${attempt} failed, retrying in 2s...`, e);
      await new Promise(r => setTimeout(r, 2000));
    }
  }
  throw new Error('unreachable');
}

/**
 * Node B: Human Mesh Recovery using ROMP + nosmpl ONNX models.
 *
 * Falls back to simulated animation if models fail to load.
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
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private _smplSession: any = null;
  private _useSimulation = false;

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

    const status: ((id: string, s: string, t: string) => void) | undefined = (globalThis as any).__joshLoadingStatus;

    try {
      status?.('hmrModel', 'active', 'Node B: Downloading ROMP pose model (111 MB)...');
      console.log('[HMR] Downloading ROMP model (111MB)...');
      const rompBuf = await fetchModelBuffer(this._rompModelUrl);
      status?.('hmrModel', 'active', 'Node B: Creating ROMP session...');
      console.log('[HMR] Creating ROMP session...');
      this._rompSession = await ort.InferenceSession.create(rompBuf, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      status?.('hmrModel', 'done', 'Node B: ROMP pose model ready');
      console.log('[HMR] ROMP loaded:', this._rompSession.inputNames, '→', this._rompSession.outputNames);

      status?.('smplModel', 'active', 'Node B: Downloading SMPL model (17 MB)...');
      console.log('[HMR] Downloading SMPL model (17MB)...');
      const smplBuf = await fetchModelBuffer(this._smplModelUrl);
      status?.('smplModel', 'active', 'Node B: Creating SMPL session...');
      console.log('[HMR] Creating SMPL session...');
      this._smplSession = await ort.InferenceSession.create(smplBuf, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      status?.('smplModel', 'done', 'Node B: SMPL forward pass model ready');
      console.log('[HMR] SMPL loaded:', this._smplSession.inputNames, '→', this._smplSession.outputNames);
    } catch (e) {
      console.warn('[HMR] Model loading failed, using simulated animation:', e);
      status?.('hmrModel', 'warn', 'Node B: ROMP unavailable — using simulated pose');
      status?.('smplModel', 'warn', 'Node B: SMPL unavailable — using simulated mesh');
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

    // Extract SMPL params: [0-2] camera, [3-74] pose, [75-84] shape
    const camera = new Float32Array(3);
    const poseAxisAngle = new Float32Array(SMPL_POSE_DIM);

    for (let c = 0; c < 3; c++) {
      camera[c] = paramsMaps[c * mapSize * mapSize + bestIdx]!;
    }
    for (let c = 0; c < SMPL_POSE_DIM; c++) {
      poseAxisAngle[c] = paramsMaps[(c + 3) * mapSize * mapSize + bestIdx]!;
    }

    // SMPL forward pass
    const globalOrient = new Float32Array([poseAxisAngle[0]!, poseAxisAngle[1]!, poseAxisAngle[2]!]);
    const bodyPose = new Float32Array(23 * 3);
    for (let i = 0; i < 23 * 3; i++) bodyPose[i] = poseAxisAngle[i + 3]!;

    const orientTensor = new ort.Tensor('float32', globalOrient, [1, 1, 3]);
    const bodyTensor = new ort.Tensor('float32', bodyPose, [1, 23, 3]);
    const smplFeeds: Record<string, InstanceType<typeof ort.Tensor>> = {};
    smplFeeds[this._smplSession.inputNames[0]!] = orientTensor;
    smplFeeds[this._smplSession.inputNames[1]!] = bodyTensor;
    const smplResults = await this._smplSession.run(smplFeeds);
    const vertices = smplResults['vertices']!.data as Float32Array;
    const joints = smplResults['joints']!.data as Float32Array;

    // Copy to output
    const numVerts = Math.min(vertices.length, this._verticesOut.length);
    this._verticesOut.set(vertices.subarray(0, numVerts));
    for (let j = 0; j < SMPL_JOINT_COUNT; j++) {
      this._jointsOut[j * 3] = joints[j * 3]!;
      this._jointsOut[j * 3 + 1] = joints[j * 3 + 1]!;
      this._jointsOut[j * 3 + 2] = joints[j * 3 + 2]!;
    }

    this._device!.queue.writeBuffer(verticesOut, 0, this._verticesOut);
    this._device!.queue.writeBuffer(jointsOut, 0, this._jointsOut);
    this._device!.queue.writeBuffer(cameraOut, 0, camera);
  }

  /**
   * Simulation fallback when models aren't available.
   * Produces walking animation so the pipeline keeps running.
   */
  private _runSimulation(
    frameIndex: number,
    verticesOut: GPUBuffer,
    jointsOut: GPUBuffer,
    cameraOut: GPUBuffer,
  ): void {
    const t = frameIndex * 0.05;
    const walkPhase = Math.sin(t);
    const walkAmp = 0.3;

    // Generate joint positions from walking animation
    const TPOSE: [number, number, number][] = [
      [0,0.9,0],[0.1,0.85,0],[-0.1,0.85,0],[0,1.05,0],[0.1,0.45,0],[-0.1,0.45,0],
      [0,1.2,0],[0.1,0.05,0],[-0.1,0.05,0],[0,1.35,0],[0.1,0,0.05],[-0.1,0,0.05],
      [0,1.5,0],[0.1,1.45,0],[-0.1,1.45,0],[0,1.65,0],[0.22,1.42,0],[-0.22,1.42,0],
      [0.45,1.42,0],[-0.45,1.42,0],[0.65,1.42,0],[-0.65,1.42,0],[0.72,1.42,0],[-0.72,1.42,0],
    ];

    for (let j = 0; j < SMPL_JOINT_COUNT; j++) {
      const pos = TPOSE[j]!;
      this._jointsOut[j * 3] = pos[0];
      this._jointsOut[j * 3 + 1] = pos[1];
      this._jointsOut[j * 3 + 2] = pos[2];
    }
    // Add walking motion to hips/knees
    this._jointsOut[1 * 3 + 2] = walkPhase * walkAmp * 0.2;
    this._jointsOut[2 * 3 + 2] = -walkPhase * walkAmp * 0.2;
    this._jointsOut[4 * 3 + 1] = 0.45 + Math.max(0, -walkPhase) * walkAmp * 0.3;
    this._jointsOut[5 * 3 + 1] = 0.45 + Math.max(0, walkPhase) * walkAmp * 0.3;

    // Zero out vertices (no mesh in simulation mode)
    this._verticesOut.fill(0);

    this._device!.queue.writeBuffer(verticesOut, 0, this._verticesOut);
    this._device!.queue.writeBuffer(jointsOut, 0, this._jointsOut);
    this._device!.queue.writeBuffer(cameraOut, 0, new Float32Array([1.0, 0.0, 0.0]));
  }

  dispose(): void {
    this._rompSession?.release();
    this._smplSession?.release();
  }
}
