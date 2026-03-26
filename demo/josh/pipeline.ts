import { createGraphBuilder } from '../../src/graph/graph.ts';
import type { NodeContext } from '../../src/graph/node.ts';
import type { CompiledGraph } from '../../src/graph/compiler.ts';
import type { NodeId } from '../../src/core/types.ts';
import { DepthEstimationNode } from './nodes/depth-estimation.node.ts';
import { HumanMeshRecoveryNode } from './nodes/human-mesh-recovery.node.ts';
import { JoshSolverNode } from './nodes/josh-solver.node.ts';

export interface JoshPipelineResult {
  pipeline: CompiledGraph;
  depthNodeId: NodeId;
  hmrNodeId: NodeId;
  solverNodeId: NodeId;
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
  const depthNode = new DepthEstimationNode();
  const hmrNode = new HumanMeshRecoveryNode();
  const solverNode = new JoshSolverNode();

  const graph = createGraphBuilder('josh-4d-reconstruction')
    // Add nodes
    .addNode(depthNode)
    .addNode(hmrNode)
    .addNode(solverNode)

    // Graph inputs: video frame fans out to both depth and HMR
    .setInput(depthNode.id, 'rgbFrame')
    .setInput(hmrNode.id, 'rgbFrame')

    // Depth → Solver
    .connect(depthNode.id, 'depthMap', solverNode.id, 'depthMap')

    // HMR → Solver
    .connect(hmrNode.id, 'smplVertices', solverNode.id, 'smplVertices')
    .connect(hmrNode.id, 'jointPositions', solverNode.id, 'jointPositions')
    .connect(hmrNode.id, 'estimatedCamera', solverNode.id, 'initialCamera')

    // Graph outputs
    .setOutput(solverNode.id, 'optimizedDepth')
    .setOutput(solverNode.id, 'refinedVertices')
    .setOutput(solverNode.id, 'cameraExtrinsics');

  // Compile: validates DAG, performs island fusion, allocates buffers
  const pipeline = await graph.compile(ctx);
  return {
    pipeline,
    depthNodeId: depthNode.id,
    hmrNodeId: hmrNode.id,
    solverNodeId: solverNode.id,
  };
}
