import type { GComputeNode, NodeContext, ExecutionContext, PortDescriptor } from '../../../src/graph/node.ts';
import type { NodeId } from '../../../src/core/types.ts';
import { createNodeId, computeBufferLayout, dim } from '../../../src/core/types.ts';
import type { Shape1D } from '../../../src/core/types.ts';
import {
  SMPL_VERTEX_COUNT, SMPL_JOINT_COUNT, SMPL_POSE_DIM, SMPL_SHAPE_DIM,
  SMPL_KINEMATIC_TREE, SMPL_CONTACT_VERTICES,
} from '../models/smpl.ts';

// Import WGSL shaders
import rodriguesSource from '../kernels/rodrigues.wgsl?raw';
import rodriguesDerivSource from '../kernels/rodrigues-deriv.wgsl?raw';
import smplJointsSource from '../kernels/smpl-joints.wgsl?raw';
import smplForwardSource from '../kernels/smpl-forward.wgsl?raw';
import jvpGradientSource from '../kernels/jvp-gradient.wgsl?raw';
import jvpJointGradientSource from '../kernels/jvp-joint-gradient.wgsl?raw';
import adamSource from '../kernels/adam-optimizer.wgsl?raw';
import counterSource from '../kernels/increment-counter.wgsl?raw';
import contactLossSource from '../kernels/josh-contact-loss.wgsl?raw';
import contactStaticSource from '../kernels/josh-contact-static.wgsl?raw';
import depthReprojSource from '../kernels/josh-depth-reproj.wgsl?raw';
import reproj2DSource from '../kernels/josh-reproj-2d.wgsl?raw';
import smplPriorSource from '../kernels/josh-smpl-prior.wgsl?raw';
import temporalSource from '../kernels/josh-temporal.wgsl?raw';

// ─── Constants ───────────────────────────────────────────────────────────────

const PARAM_DIM = SMPL_POSE_DIM + SMPL_SHAPE_DIM + 3 + 3 + 1; // 89
const STAGE1_ITERS = 500;
const STAGE2_ITERS = 200;
const STAGE1_LR = 0.07;
const STAGE2_LR = 0.014;
const ADAM_BETA1 = 0.9;
const ADAM_BETA2 = 0.999;
const ADAM_EPS = 1e-8;
const DEPTH_H = 384;
const DEPTH_W = 384;
const NUM_KEYPOINTS = 17;

// Camera intrinsics
const FX = 300.0;
const FY = 300.0;
const CX = 192.0;
const CY = 192.0;

// MoveNet → SMPL joint mapping (17 keypoints → SMPL joint indices)
// MoveNet: nose(0), left_eye(1), right_eye(2), left_ear(3), right_ear(4),
//   left_shoulder(5), right_shoulder(6), left_elbow(7), right_elbow(8),
//   left_wrist(9), right_wrist(10), left_hip(11), right_hip(12),
//   left_knee(13), right_knee(14), left_ankle(15), right_ankle(16)
const MOVENET_TO_SMPL: readonly number[] = [
  15, // 0: nose → head
  15, // 1: left_eye → head
  15, // 2: right_eye → head
  15, // 3: left_ear → head
  15, // 4: right_ear → head
  16, // 5: left_shoulder → left_shoulder
  17, // 6: right_shoulder → right_shoulder
  18, // 7: left_elbow → left_elbow
  19, // 8: right_elbow → right_elbow
  20, // 9: left_wrist → left_wrist
  21, // 10: right_wrist → right_wrist
  1,  // 11: left_hip → left_hip
  2,  // 12: right_hip → right_hip
  4,  // 13: left_knee → left_knee
  5,  // 14: right_knee → right_knee
  7,  // 15: left_ankle → left_ankle
  8,  // 16: right_ankle → right_ankle
];

// ─── Port descriptors ─────────────────────────────────────────────────────────

const depthLayout = computeBufferLayout([dim(DEPTH_H * DEPTH_W)] as Shape1D, 'f32');
const verticesLayout = computeBufferLayout([dim(SMPL_VERTEX_COUNT * 3)] as Shape1D, 'f32');
const paramsLayout = computeBufferLayout([dim(PARAM_DIM)] as Shape1D, 'f32');

const INPUT_PORTS = [
  { name: 'depthMap',           layout: depthLayout },
  { name: 'smplVerticesShaped', layout: verticesLayout },
  { name: 'initPose',           layout: computeBufferLayout([dim(SMPL_POSE_DIM)] as Shape1D, 'f32') },
  { name: 'initShape',          layout: computeBufferLayout([dim(SMPL_SHAPE_DIM)] as Shape1D, 'f32') },
  { name: 'keypoints2D',        layout: computeBufferLayout([dim(NUM_KEYPOINTS * 2)] as Shape1D, 'f32') },
  { name: 'keypointConf',       layout: computeBufferLayout([dim(NUM_KEYPOINTS)] as Shape1D, 'f32') },
  { name: 'prevVertices',       layout: verticesLayout },
  { name: 'prevParams',         layout: paramsLayout },
  { name: 'cameraIntrinsics',   layout: computeBufferLayout([dim(4)] as Shape1D, 'f32') },
] as const satisfies readonly PortDescriptor[];

const OUTPUT_PORTS = [
  { name: 'optimizedParams',   layout: paramsLayout },
  { name: 'optimizedVertices', layout: verticesLayout },
] as const satisfies readonly PortDescriptor[];

// ─── Uniform buffer helpers ───────────────────────────────────────────────────

function makeAdamConfig(lr: number): Float32Array {
  const buf = new Float32Array(8);
  buf[0] = lr;
  buf[1] = ADAM_BETA1;
  buf[2] = ADAM_BETA2;
  buf[3] = ADAM_EPS;
  buf[4] = PARAM_DIM;
  // [5..7] pad
  return buf;
}

function makeDepthReprojUniforms(): Float32Array {
  const b = new Float32Array(8);
  b[0] = DEPTH_W; b[1] = DEPTH_H; b[2] = SMPL_VERTEX_COUNT;
  b[3] = FX; b[4] = FY; b[5] = CX; b[6] = CY;
  b[7] = 0.1; // weight
  return b;
}

function makeContactLossUniforms(numContacts: number): ArrayBuffer {
  const ab = new ArrayBuffer(48);
  const dv = new DataView(ab);
  dv.setUint32(0, DEPTH_W, true);
  dv.setUint32(4, DEPTH_H, true);
  dv.setUint32(8, numContacts, true);
  dv.setFloat32(12, FX, true);
  dv.setFloat32(16, FY, true);
  dv.setFloat32(20, CX, true);
  dv.setFloat32(24, CY, true);
  dv.setFloat32(28, 1.0, true); // weight
  dv.setFloat32(32, 0.0, true); // delta_c1
  // [36..47] pad
  return ab;
}

function makeContactStaticUniforms(numContacts: number): Float32Array {
  const b = new Float32Array(4);
  b[0] = numContacts; b[1] = 20.0; b[2] = 0.1; // weight, delta_c2
  return b;
}

function makeReproj2DUniforms(): Float32Array {
  const b = new Float32Array(8);
  b[0] = FX; b[1] = FY; b[2] = CX; b[3] = CY;
  b[4] = NUM_KEYPOINTS;
  b[5] = 5.0; // weight
  // [6..7] pad
  return b;
}

function makeSmplPriorUniforms(): Float32Array {
  const b = new Float32Array(4);
  b[0] = 0.01; // weight
  b[1] = 82;   // param_count
  return b;
}

function makeTemporalUniforms(): Float32Array {
  const b = new Float32Array(4);
  b[0] = PARAM_DIM; // param_dim
  b[1] = 10.0;      // weight
  return b;
}

function makeSmplFKUniforms(): Float32Array {
  const b = new Float32Array(4);
  b[0] = SMPL_VERTEX_COUNT;
  b[1] = SMPL_JOINT_COUNT;
  return b;
}

function makeSmplLBSUniforms(): Float32Array {
  const b = new Float32Array(4);
  b[0] = SMPL_VERTEX_COUNT;
  b[1] = SMPL_JOINT_COUNT;
  b[2] = SMPL_SHAPE_DIM;
  return b;
}

function makeJVPUniforms(): Float32Array {
  const b = new Float32Array(4);
  b[0] = SMPL_VERTEX_COUNT;
  b[1] = SMPL_JOINT_COUNT;
  return b;
}

function makeJVPJointUniforms(): Float32Array {
  const b = new Float32Array(4);
  b[0] = SMPL_JOINT_COUNT;
  return b;
}

// ─── Node ────────────────────────────────────────────────────────────────────

/**
 * Phase 0E: JOSH Optimizer Node — full 700-iteration Adam loop on GPU.
 *
 * Records ALL iterations into ONE WebGPU command encoder, submits once,
 * then does a single mapAsync to read back the result. No CPU involvement
 * during the optimization loop.
 *
 * Stage 1: iterations 0-499, lr=0.07, no L_2D loss
 * Stage 2: iterations 500-699, lr=0.014, L_2D enabled
 */
export class JoshOptimizerNode
  implements GComputeNode<typeof INPUT_PORTS, typeof OUTPUT_PORTS>
{
  readonly id: NodeId = createNodeId('joshOptimizer');
  readonly name = 'JOSHOptimizer';
  readonly backendHint = 'webgpu' as const;
  readonly inputs = INPUT_PORTS;
  readonly outputs = OUTPUT_PORTS;

  private _device: GPUDevice | null = null;

  // ── Pipelines ─────────────────────────────────────────────────────────────
  private _pipRodrigues!: GPUComputePipeline;
  private _pipRodriguesdDeriv!: GPUComputePipeline;
  private _pipSmplFK!: GPUComputePipeline;
  private _pipSmplLBS!: GPUComputePipeline;
  private _pipJVP!: GPUComputePipeline;
  private _pipJVPJoint!: GPUComputePipeline;
  private _pipContactLoss!: GPUComputePipeline;
  private _pipContactStatic!: GPUComputePipeline;
  private _pipDepthReproj!: GPUComputePipeline;
  private _pipReproj2D!: GPUComputePipeline;
  private _pipSmplPrior!: GPUComputePipeline;
  private _pipTemporal!: GPUComputePipeline;
  private _pipCounter!: GPUComputePipeline;
  private _pipAdam!: GPUComputePipeline;

  // ── Persistent GPU buffers (owned by this node) ───────────────────────────
  private _params!: GPUBuffer;           // [89] f32 — current optimizer state
  private _adamM!: GPUBuffer;            // [89] f32 — Adam first moment
  private _adamV!: GPUBuffer;            // [89] f32 — Adam second moment
  private _adamCounter!: GPUBuffer;      // [1] u32 — Adam timestep
  private _gradient!: GPUBuffer;         // [89] f32 — accumulated gradient
  private _dlDv!: GPUBuffer;             // [V*3] f32 — per-vertex gradient
  private _dlDjoint!: GPUBuffer;         // [24*3] f32 — per-joint gradient
  private _lossAccum!: GPUBuffer;        // [4] f32 — scalar loss components
  private _localRots!: GPUBuffer;        // [24*9] f32 — Rodrigues output
  private _dR!: GPUBuffer;               // [72*9] f32 — Rodrigues derivatives
  private _jointTransforms!: GPUBuffer;  // [24*16] f32 — FK output
  private _jointPositions!: GPUBuffer;   // [24*3] f32 — joint world positions
  private _vertices!: GPUBuffer;         // [V*3] f32 — LBS output (current frame)
  private _contactIndices!: GPUBuffer;   // [numContacts] u32
  private _parentIndices!: GPUBuffer;    // [24] i32
  private _jointToSmpl!: GPUBuffer;      // [17] u32 — keypoint → SMPL joint
  private _stagingParams!: GPUBuffer;    // [89] f32 MAP_READ
  private _stagingVertices!: GPUBuffer;  // [V*3] f32 MAP_READ

  // Uniform buffers
  private _ubAdamStage1!: GPUBuffer;
  private _ubAdamStage2!: GPUBuffer;
  private _ubDepthReproj!: GPUBuffer;
  private _ubContactLoss!: GPUBuffer;
  private _ubContactStatic!: GPUBuffer;
  private _ubReproj2D!: GPUBuffer;
  private _ubSmplPrior!: GPUBuffer;
  private _ubTemporal!: GPUBuffer;
  private _ubSmplFK!: GPUBuffer;
  private _ubSmplLBS!: GPUBuffer;
  private _ubJVP!: GPUBuffer;
  private _ubJVPJoint!: GPUBuffer;

  // Contact vertex count (filled in initialize)
  private _numContacts = 0;

  // ── Initialize ────────────────────────────────────────────────────────────

  async initialize(ctx: NodeContext): Promise<void> {
    const { device } = ctx;
    this._device = device;

    const status: ((id: string, s: string, t: string) => void) | undefined =
      (globalThis as any).__joshLoadingStatus;

    status?.('optimizer', 'active', 'Phase 0E: Compiling 700-iter GPU Adam optimizer...');

    // Build contact index list
    const contactArr = new Uint32Array([
      ...SMPL_CONTACT_VERTICES.leftFootSole,
      ...SMPL_CONTACT_VERTICES.rightFootSole,
      ...SMPL_CONTACT_VERTICES.leftToes,
      ...SMPL_CONTACT_VERTICES.rightToes,
    ]);
    this._numContacts = contactArr.length;

    // Build parent indices
    const parentArr = new Int32Array(SMPL_KINEMATIC_TREE);

    // ── Allocate buffers ────────────────────────────────────────────────────
    const STORAGE_RW = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const STORAGE_R  = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const UNIFORM    = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

    this._params          = device.createBuffer({ size: PARAM_DIM * 4, usage: STORAGE_RW, label: 'adam_params' });
    this._adamM           = device.createBuffer({ size: PARAM_DIM * 4, usage: STORAGE_RW, label: 'adam_m' });
    this._adamV           = device.createBuffer({ size: PARAM_DIM * 4, usage: STORAGE_RW, label: 'adam_v' });
    this._adamCounter     = device.createBuffer({ size: 4,             usage: STORAGE_RW, label: 'adam_counter' });
    this._gradient        = device.createBuffer({ size: PARAM_DIM * 4, usage: STORAGE_RW, label: 'gradient' });
    this._dlDv            = device.createBuffer({ size: SMPL_VERTEX_COUNT * 3 * 4, usage: STORAGE_RW, label: 'dl_dv' });
    this._dlDjoint        = device.createBuffer({ size: SMPL_JOINT_COUNT * 3 * 4,  usage: STORAGE_RW, label: 'dl_djoint' });
    this._lossAccum       = device.createBuffer({ size: 4 * 4,         usage: STORAGE_RW, label: 'loss_accum' });
    this._localRots       = device.createBuffer({ size: SMPL_JOINT_COUNT * 9 * 4,  usage: STORAGE_RW, label: 'local_rots' });
    this._dR              = device.createBuffer({ size: SMPL_POSE_DIM * 9 * 4,     usage: STORAGE_RW, label: 'dR' });
    this._jointTransforms = device.createBuffer({ size: SMPL_JOINT_COUNT * 16 * 4, usage: STORAGE_RW, label: 'joint_transforms' });
    this._jointPositions  = device.createBuffer({ size: SMPL_JOINT_COUNT * 3 * 4,  usage: STORAGE_RW, label: 'joint_positions' });
    this._vertices        = device.createBuffer({ size: SMPL_VERTEX_COUNT * 3 * 4, usage: STORAGE_RW, label: 'vertices' });

    this._contactIndices = device.createBuffer({ size: Math.max(contactArr.byteLength, 4), usage: STORAGE_R, label: 'contact_indices' });
    device.queue.writeBuffer(this._contactIndices, 0, contactArr);

    this._parentIndices = device.createBuffer({ size: parentArr.byteLength, usage: STORAGE_R, label: 'parent_indices' });
    device.queue.writeBuffer(this._parentIndices, 0, parentArr);

    const jointToSmplArr = new Uint32Array(MOVENET_TO_SMPL);
    this._jointToSmpl = device.createBuffer({ size: jointToSmplArr.byteLength, usage: STORAGE_R, label: 'joint_to_smpl' });
    device.queue.writeBuffer(this._jointToSmpl, 0, jointToSmplArr);

    this._stagingParams   = device.createBuffer({ size: PARAM_DIM * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: 'staging_params' });
    this._stagingVertices = device.createBuffer({ size: SMPL_VERTEX_COUNT * 3 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: 'staging_vertices' });

    // ── Uniform buffers ─────────────────────────────────────────────────────
    const writeUniform = (data: Float32Array | ArrayBuffer, label: string): GPUBuffer => {
      const raw: ArrayBuffer = data instanceof Float32Array ? data.buffer as ArrayBuffer : data;
      const size = Math.max(Math.ceil(raw.byteLength / 16) * 16, 16);
      const buf = device.createBuffer({ size, usage: UNIFORM, label });
      device.queue.writeBuffer(buf, 0, raw);
      return buf;
    };

    this._ubAdamStage1   = writeUniform(makeAdamConfig(STAGE1_LR), 'ub_adam_stage1');
    this._ubAdamStage2   = writeUniform(makeAdamConfig(STAGE2_LR), 'ub_adam_stage2');
    this._ubDepthReproj  = writeUniform(makeDepthReprojUniforms(), 'ub_depth_reproj');
    this._ubContactLoss  = writeUniform(makeContactLossUniforms(this._numContacts), 'ub_contact_loss');
    this._ubContactStatic = writeUniform(makeContactStaticUniforms(this._numContacts), 'ub_contact_static');
    this._ubReproj2D     = writeUniform(makeReproj2DUniforms(), 'ub_reproj_2d');
    this._ubSmplPrior    = writeUniform(makeSmplPriorUniforms(), 'ub_smpl_prior');
    this._ubTemporal     = writeUniform(makeTemporalUniforms(), 'ub_temporal');
    this._ubSmplFK       = writeUniform(makeSmplFKUniforms(), 'ub_smpl_fk');
    this._ubSmplLBS      = writeUniform(makeSmplLBSUniforms(), 'ub_smpl_lbs');
    this._ubJVP          = writeUniform(makeJVPUniforms(), 'ub_jvp');
    this._ubJVPJoint     = writeUniform(makeJVPJointUniforms(), 'ub_jvp_joint');

    // ── Compile pipelines ────────────────────────────────────────────────────
    const mkPipeline = (src: string, label: string): GPUComputePipeline =>
      device.createComputePipeline({
        label,
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: src, label }), entryPoint: 'main' },
      });

    [
      this._pipRodrigues,
      this._pipRodriguesdDeriv,
      this._pipSmplFK,
      this._pipSmplLBS,
      this._pipJVP,
      this._pipJVPJoint,
      this._pipContactLoss,
      this._pipContactStatic,
      this._pipDepthReproj,
      this._pipReproj2D,
      this._pipSmplPrior,
      this._pipTemporal,
      this._pipCounter,
      this._pipAdam,
    ] = await Promise.all([
      mkPipeline(rodriguesSource,      'rodrigues'),
      mkPipeline(rodriguesDerivSource, 'rodrigues_deriv'),
      mkPipeline(smplJointsSource,     'smpl_fk'),
      mkPipeline(smplForwardSource,    'smpl_lbs'),
      mkPipeline(jvpGradientSource,    'jvp_gradient'),
      mkPipeline(jvpJointGradientSource, 'jvp_joint_gradient'),
      mkPipeline(contactLossSource,    'contact_loss'),
      mkPipeline(contactStaticSource,  'contact_static'),
      mkPipeline(depthReprojSource,    'depth_reproj'),
      mkPipeline(reproj2DSource,       'reproj_2d'),
      mkPipeline(smplPriorSource,      'smpl_prior'),
      mkPipeline(temporalSource,       'temporal'),
      mkPipeline(counterSource,        'counter'),
      mkPipeline(adamSource,           'adam'),
    ]);

    status?.('optimizer', 'done', 'Phase 0E: GPU Adam optimizer ready (700 iterations, single encoder)');
    console.log('[JOSHOptimizer] Initialized: 700-iter Adam on GPU, single command encoder');
  }

  // ── Execute ───────────────────────────────────────────────────────────────

  async execute(ctx: ExecutionContext): Promise<void> {
    const device = this._device!;

    // Retrieve input buffers from port system
    const depthMap           = ctx.getInput('depthMap') as GPUBuffer;
    const smplVerticesShaped = ctx.getInput('smplVerticesShaped') as GPUBuffer;
    const initPose           = ctx.getInput('initPose') as GPUBuffer;
    const initShape          = ctx.getInput('initShape') as GPUBuffer;
    const keypoints2D        = ctx.getInput('keypoints2D') as GPUBuffer;
    const keypointConf       = ctx.getInput('keypointConf') as GPUBuffer;
    const prevVertices       = ctx.getInput('prevVertices') as GPUBuffer;
    const prevParams         = ctx.getInput('prevParams') as GPUBuffer;

    const outParams   = ctx.getOutput('optimizedParams') as GPUBuffer;
    const outVertices = ctx.getOutput('optimizedVertices') as GPUBuffer;

    // Initialise optimizer state for this frame — copy init pose/shape into params,
    // zero Adam moments, reset counter.
    const initEncoder = device.createCommandEncoder({ label: 'josh_init' });
    initEncoder.clearBuffer(this._params,      0, PARAM_DIM * 4);
    initEncoder.clearBuffer(this._adamM,       0, PARAM_DIM * 4);
    initEncoder.clearBuffer(this._adamV,       0, PARAM_DIM * 4);
    initEncoder.clearBuffer(this._adamCounter, 0, 4);
    // Copy pose[0..71] and shape[72..81] from init buffers
    initEncoder.copyBufferToBuffer(initPose,  0, this._params, 0,  SMPL_POSE_DIM * 4);
    initEncoder.copyBufferToBuffer(initShape, 0, this._params, SMPL_POSE_DIM * 4, SMPL_SHAPE_DIM * 4);
    device.queue.submit([initEncoder.finish()]);

    // ── Single mega command encoder for 700 iterations ──────────────────────
    const enc = device.createCommandEncoder({ label: 'josh_optimizer_700' });

    // Helper: create a bind group from a pipeline and its bindings
    const bg = (pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[]): GPUBindGroup =>
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries,
      });

    const buf = (b: GPUBuffer): GPUBindGroupEntry['resource'] => ({ buffer: b });

    // ── Pre-build bind groups (buffers are immutable refs — content changes on GPU) ──

    // Rodrigues: pose[0..71] lives inside _params at offset 0
    // We bind the full _params buffer — shader reads pose[j*3..j*3+2].
    const bgRodrigues = bg(this._pipRodrigues, [
      { binding: 0, resource: buf(this._params) },       // pose slice (reads [0..71])
      { binding: 1, resource: buf(this._localRots) },
    ]);

    const bgRodriguesdDeriv = bg(this._pipRodriguesdDeriv, [
      { binding: 0, resource: buf(this._params) },
      { binding: 1, resource: buf(this._dR) },
    ]);

    const bgSmplFK = bg(this._pipSmplFK, [
      { binding: 0, resource: buf(smplVerticesShaped) },
      { binding: 1, resource: buf(this._jointPositions) }, // joint_regressor not available here; FK uses shaped verts
      { binding: 2, resource: buf(this._localRots) },
      { binding: 3, resource: buf(this._parentIndices) },
      { binding: 4, resource: buf(this._jointTransforms) },
      { binding: 5, resource: buf(this._jointPositions) },
      { binding: 6, resource: buf(this._ubSmplFK) },
    ]);

    const bgSmplLBS = bg(this._pipSmplLBS, [
      { binding: 0, resource: buf(smplVerticesShaped) },  // mean_template (pre-shaped)
      { binding: 1, resource: buf(smplVerticesShaped) },  // shape_blend_shapes — identity (already shaped)
      { binding: 2, resource: buf(smplVerticesShaped) },  // skinning_weights placeholder
      { binding: 3, resource: buf(smplVerticesShaped) },  // skinning_indices placeholder
      { binding: 4, resource: buf(this._jointTransforms) },
      { binding: 5, resource: buf(this._params) },        // shape params at [72..81]
      { binding: 6, resource: buf(this._vertices) },
      { binding: 7, resource: buf(this._ubSmplLBS) },
    ]);

    const bgContactLoss = bg(this._pipContactLoss, [
      { binding: 0, resource: buf(this._vertices) },
      { binding: 1, resource: buf(depthMap) },
      { binding: 2, resource: buf(this._contactIndices) },
      { binding: 3, resource: buf(this._dlDv) },
      { binding: 4, resource: buf(this._lossAccum) },
      { binding: 5, resource: buf(this._ubContactLoss) },
    ]);

    const bgContactStatic = bg(this._pipContactStatic, [
      { binding: 0, resource: buf(this._vertices) },
      { binding: 1, resource: buf(prevVertices) },
      { binding: 2, resource: buf(this._contactIndices) },
      { binding: 3, resource: buf(this._dlDv) },
      { binding: 4, resource: buf(this._lossAccum) },
      { binding: 5, resource: buf(this._ubContactStatic) },
    ]);

    const bgDepthReproj = bg(this._pipDepthReproj, [
      { binding: 0, resource: buf(this._vertices) },
      { binding: 1, resource: buf(depthMap) },
      { binding: 2, resource: buf(this._dlDv) },
      { binding: 3, resource: buf(this._lossAccum) },
      { binding: 4, resource: buf(this._ubDepthReproj) },
    ]);

    const bgReproj2D = bg(this._pipReproj2D, [
      { binding: 0, resource: buf(this._jointPositions) },
      { binding: 1, resource: buf(keypoints2D) },
      { binding: 2, resource: buf(keypointConf) },
      { binding: 3, resource: buf(this._jointToSmpl) },
      { binding: 4, resource: buf(this._dlDjoint) },
      { binding: 5, resource: buf(this._lossAccum) },
      { binding: 6, resource: buf(this._ubReproj2D) },
    ]);

    const bgSmplPrior = bg(this._pipSmplPrior, [
      { binding: 0, resource: buf(this._params) },  // pose at [0]
      { binding: 1, resource: buf(initPose) },
      { binding: 2, resource: buf(this._params) },  // shape at [72] — shader reads correct offsets
      { binding: 3, resource: buf(initShape) },
      { binding: 4, resource: buf(this._gradient) },
      { binding: 5, resource: buf(this._lossAccum) },
      { binding: 6, resource: buf(this._ubSmplPrior) },
    ]);

    const bgTemporal = bg(this._pipTemporal, [
      { binding: 0, resource: buf(this._params) },
      { binding: 1, resource: buf(prevParams) },
      { binding: 2, resource: buf(this._gradient) },
      { binding: 3, resource: buf(this._lossAccum) },
      { binding: 4, resource: buf(this._ubTemporal) },
    ]);

    const bgJVP = bg(this._pipJVP, [
      { binding: 0, resource: buf(this._dlDv) },
      { binding: 1, resource: buf(this._dR) },
      { binding: 2, resource: buf(this._jointTransforms) },
      { binding: 3, resource: buf(smplVerticesShaped) },
      { binding: 4, resource: buf(smplVerticesShaped) },  // skin_weights placeholder
      { binding: 5, resource: buf(smplVerticesShaped) },  // skin_indices placeholder
      { binding: 6, resource: buf(this._parentIndices) },
      { binding: 7, resource: buf(this._gradient) },
      { binding: 8, resource: buf(this._ubJVP) },
    ]);

    const bgJVPJoint = bg(this._pipJVPJoint, [
      { binding: 0, resource: buf(this._dlDjoint) },
      { binding: 1, resource: buf(this._dR) },
      { binding: 2, resource: buf(this._jointTransforms) },
      { binding: 3, resource: buf(this._parentIndices) },
      { binding: 4, resource: buf(this._gradient) },
      { binding: 5, resource: buf(this._ubJVPJoint) },
    ]);

    const bgCounter = bg(this._pipCounter, [
      { binding: 0, resource: buf(this._adamCounter) },
    ]);

    const bgAdamStage1 = bg(this._pipAdam, [
      { binding: 0, resource: buf(this._params) },
      { binding: 1, resource: buf(this._gradient) },
      { binding: 2, resource: buf(this._adamM) },
      { binding: 3, resource: buf(this._adamV) },
      { binding: 4, resource: buf(this._ubAdamStage1) },
      { binding: 5, resource: buf(this._adamCounter) },
    ]);

    const bgAdamStage2 = bg(this._pipAdam, [
      { binding: 0, resource: buf(this._params) },
      { binding: 1, resource: buf(this._gradient) },
      { binding: 2, resource: buf(this._adamM) },
      { binding: 3, resource: buf(this._adamV) },
      { binding: 4, resource: buf(this._ubAdamStage2) },
      { binding: 5, resource: buf(this._adamCounter) },
    ]);

    // Dispatch workgroup sizes
    const VERTS_WG   = Math.ceil(SMPL_VERTEX_COUNT / 256);
    const PARAM_WG   = Math.ceil(PARAM_DIM / 256);
    const POSE_WG    = Math.ceil(SMPL_POSE_DIM / 64);
    const KP_WG      = Math.ceil(NUM_KEYPOINTS / 32);
    const CONTACT_WG = Math.ceil(this._numContacts / 64);
    const PRIOR_WG   = 1; // workgroup_size(82) — dispatch 1

    // Helper to dispatch a pass
    const dispatch = (
      pass: GPUComputePassEncoder,
      pipeline: GPUComputePipeline,
      bindGroup: GPUBindGroup,
      x: number, y = 1, z = 1,
    ): void => {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(x, y, z);
    };

    // ── Encode all 700 iterations ────────────────────────────────────────────
    const totalIters = STAGE1_ITERS + STAGE2_ITERS;

    for (let iter = 0; iter < totalIters; iter++) {
      const isStage2 = iter >= STAGE1_ITERS;
      const bgAdam = isStage2 ? bgAdamStage2 : bgAdamStage1;

      // Step 0: Clear per-iteration gradient buffers
      enc.clearBuffer(this._dlDv,      0, SMPL_VERTEX_COUNT * 3 * 4);
      enc.clearBuffer(this._dlDjoint,  0, SMPL_JOINT_COUNT * 3 * 4);
      enc.clearBuffer(this._gradient,  0, PARAM_DIM * 4);
      enc.clearBuffer(this._lossAccum, 0, 4 * 4);

      const pass = enc.beginComputePass({ label: `iter_${iter}` });

      // Step 1: Rodrigues forward — pose → local_rots
      dispatch(pass, this._pipRodrigues, bgRodrigues, 1); // workgroup_size(24)

      // Step 2: SMPL FK — local_rots → joint_transforms + joint_positions
      dispatch(pass, this._pipSmplFK, bgSmplFK, 1); // workgroup_size(1) — sequential FK

      // Step 3: SMPL LBS — joint_transforms → vertices
      dispatch(pass, this._pipSmplLBS, bgSmplLBS, VERTS_WG);

      // Step 4: Rodrigues derivative — pose → dR [72,9]
      dispatch(pass, this._pipRodriguesdDeriv, bgRodriguesdDeriv, POSE_WG);

      // Step 5a: Contact scale loss → dl_dv
      dispatch(pass, this._pipContactLoss, bgContactLoss, CONTACT_WG);

      // Step 5b: Contact static loss → dl_dv
      dispatch(pass, this._pipContactStatic, bgContactStatic, CONTACT_WG);

      // Step 5c: Depth reprojection loss → dl_dv
      dispatch(pass, this._pipDepthReproj, bgDepthReproj, VERTS_WG);

      // Step 5d: SMPL prior → gradient direct
      dispatch(pass, this._pipSmplPrior, bgSmplPrior, PRIOR_WG);

      // Step 5e: Temporal smoothness → gradient direct
      dispatch(pass, this._pipTemporal, bgTemporal, PARAM_WG);

      // Step 5f: 2D reprojection loss → dl_djoint (Stage 2 only)
      if (isStage2) {
        dispatch(pass, this._pipReproj2D, bgReproj2D, KP_WG);
      }

      // Step 6: JVP vertex gradient — dl_dv + dR + joint_transforms → gradient[0..71]
      dispatch(pass, this._pipJVP, bgJVP, POSE_WG);

      // Step 7: JVP joint gradient — dl_djoint → gradient[0..71]
      dispatch(pass, this._pipJVPJoint, bgJVPJoint, POSE_WG);

      pass.end();

      // Step 8: Increment Adam counter (t++) — must be outside the pass
      // We use a separate pass to avoid ordering issues with the gradient pass
      const counterPass = enc.beginComputePass({ label: `counter_${iter}` });
      dispatch(counterPass, this._pipCounter, bgCounter, 1);
      counterPass.end();

      // Step 9: Adam step — gradient → new params
      const adamPass = enc.beginComputePass({ label: `adam_${iter}` });
      dispatch(adamPass, this._pipAdam, bgAdam, PARAM_WG);
      adamPass.end();
    }

    // Copy final params and vertices to staging buffers for readback
    enc.copyBufferToBuffer(this._params,   0, this._stagingParams,   0, PARAM_DIM * 4);
    enc.copyBufferToBuffer(this._vertices, 0, this._stagingVertices, 0, SMPL_VERTEX_COUNT * 3 * 4);

    // ── Single submit ────────────────────────────────────────────────────────
    device.queue.submit([enc.finish()]);

    // ── Single mapAsync — read back optimized state ──────────────────────────
    await Promise.all([
      this._stagingParams.mapAsync(GPUMapMode.READ),
      this._stagingVertices.mapAsync(GPUMapMode.READ),
    ]);

    const finalParams   = new Float32Array(this._stagingParams.getMappedRange().slice(0));
    const finalVertices = new Float32Array(this._stagingVertices.getMappedRange().slice(0));
    this._stagingParams.unmap();
    this._stagingVertices.unmap();

    // Write outputs
    device.queue.writeBuffer(outParams,   0, finalParams);
    device.queue.writeBuffer(outVertices, 0, finalVertices);

    console.log('[JOSHOptimizer] 700-iter Adam complete; params[0]:', finalParams[0]?.toFixed(4));
  }

  // ── Dispose ───────────────────────────────────────────────────────────────

  dispose(): void {
    this._params?.destroy();
    this._adamM?.destroy();
    this._adamV?.destroy();
    this._adamCounter?.destroy();
    this._gradient?.destroy();
    this._dlDv?.destroy();
    this._dlDjoint?.destroy();
    this._lossAccum?.destroy();
    this._localRots?.destroy();
    this._dR?.destroy();
    this._jointTransforms?.destroy();
    this._jointPositions?.destroy();
    this._vertices?.destroy();
    this._contactIndices?.destroy();
    this._parentIndices?.destroy();
    this._jointToSmpl?.destroy();
    this._stagingParams?.destroy();
    this._stagingVertices?.destroy();
    this._ubAdamStage1?.destroy();
    this._ubAdamStage2?.destroy();
    this._ubDepthReproj?.destroy();
    this._ubContactLoss?.destroy();
    this._ubContactStatic?.destroy();
    this._ubReproj2D?.destroy();
    this._ubSmplPrior?.destroy();
    this._ubTemporal?.destroy();
    this._ubSmplFK?.destroy();
    this._ubSmplLBS?.destroy();
    this._ubJVP?.destroy();
    this._ubJVPJoint?.destroy();
    this._device = null;
  }
}
