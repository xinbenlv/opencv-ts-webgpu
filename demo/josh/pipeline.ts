import { createGraphBuilder } from '../../src/graph/graph.ts';
import type { NodeContext } from '../../src/graph/node.ts';
import type { CompiledGraph } from '../../src/graph/compiler.ts';
import type { NodeId } from '../../src/core/types.ts';
import { DepthEstimationNode } from './nodes/depth-estimation.node.ts';
import { HumanMeshRecoveryNode } from './nodes/human-mesh-recovery.node.ts';
import { JoshSolverNode } from './nodes/josh-solver.node.ts';
import { cachedFetchModel } from './models/cached-fetch.ts';

export interface JoshPipelineResult {
  pipeline: CompiledGraph;
  depthNodeId: NodeId;
  hmrNodeId: NodeId;
  solverNodeId: NodeId;
}

/**
 * Pre-download all models in parallel before graph compilation.
 * This overlaps the 64MB + 111MB + 17MB downloads instead of doing them sequentially.
 */
async function prefetchModels(): Promise<void> {
  const status: ((id: string, s: string, t: string) => void) | undefined =
    (globalThis as any).__joshLoadingStatus;

  await Promise.allSettled([
    cachedFetchModel('./assets/models/midas-v2.1-small-256.onnx', 'depthModel', 'Node A: MiDAS depth model', '64 MB', status),
  ]);
  // Models are now in Cache API — node.initialize() will get instant cache hits
}

/**
 * Build the JOSH (Joint Optimization for 4D Human-Scene Reconstruction) pipeline.
 *
 * Graph topology:
 *
 *   VideoFrame ──┬──► [Node A: Depth Estimation] ──► depthMap ──┐
 *                │                                                │
 *                └──► [Node B: Human Mesh Recovery] ──┬──────────► [Node C: JOSH Solver]
 *                                                     │           │
 *                                   smplVertices ─────┘           ├──► optimizedDepth
 *                                   jointPositions ───────────────┤
 *                                   estimatedCamera ──────────────┘──► refinedVertices
 *                                                                      cameraExtrinsics
 *
 * Island fusion result:
 *   Island 0 [WebGPU]: depthNode, hmrNode (parallel — no inter-dependency)
 *   Island 1 [Hybrid]: solverNode (waits for Island 0)
 */
export async function buildJoshPipeline(ctx: NodeContext): Promise<JoshPipelineResult> {
  // Download all models in parallel (cached after first load)
  await prefetchModels();

  const depthNode = new DepthEstimationNode();
  const hmrNode = new HumanMeshRecoveryNode();
  const solverNode = new JoshSolverNode();

  const graph = createGraphBuilder('josh-4d-reconstruction')
    .addNode(depthNode)
    .addNode(hmrNode)
    .addNode(solverNode)
    .setInput(depthNode.id, 'rgbFrame')
    .setInput(hmrNode.id, 'rgbFrame')
    .connect(depthNode.id, 'depthMap', solverNode.id, 'depthMap')
    .connect(hmrNode.id, 'smplVertices', solverNode.id, 'smplVertices')
    .connect(hmrNode.id, 'jointPositions', solverNode.id, 'jointPositions')
    .connect(hmrNode.id, 'estimatedCamera', solverNode.id, 'initialCamera')
    .setOutput(solverNode.id, 'optimizedDepth')
    .setOutput(solverNode.id, 'refinedVertices')
    .setOutput(solverNode.id, 'cameraExtrinsics');

  const pipeline = await graph.compile(ctx);
  return {
    pipeline,
    depthNodeId: depthNode.id,
    hmrNodeId: hmrNode.id,
    solverNodeId: solverNode.id,
  };
}
