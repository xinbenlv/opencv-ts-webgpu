import type { GraphId, NodeId } from '../core/types.ts';
import type { GComputeNode, ExecutionContext, NodeContext } from './node.ts';
import type { Edge } from './edge.ts';
import type { CompiledGraph, Island } from './compiler.ts';

/**
 * Runtime executor for a compiled graph.
 * Dispatches nodes in topological order, managing buffer bindings per frame.
 */
export class GraphExecutor implements CompiledGraph {
  constructor(
    readonly id: GraphId,
    private readonly _nodes: ReadonlyMap<NodeId, GComputeNode>,
    private readonly _edges: readonly Edge[],
    readonly executionOrder: readonly NodeId[],
    readonly islands: readonly Island[],
    readonly graphInputs: ReadonlyArray<{ nodeId: NodeId; portName: string }>,
    readonly graphOutputs: ReadonlyArray<{ nodeId: NodeId; portName: string }>,
    private readonly _ctx: NodeContext,
  ) {}

  async execute(frameIndex: number): Promise<void> {
    const commandEncoder = this._ctx.device.createCommandEncoder({
      label: `frame_${frameIndex}`,
    });

    const timestampMs = performance.now();

    // Execute each island in sequence
    for (const island of this.islands) {
      // Within an island, nodes with no inter-dependency could run in parallel.
      // For now, execute sequentially in topological order.
      for (const nodeId of island.nodes) {
        const node = this._nodes.get(nodeId)!;
        const execCtx = this._createExecutionContext(
          nodeId,
          commandEncoder,
          frameIndex,
          timestampMs,
        );
        await node.execute(execCtx);
      }
    }

    // Submit all GPU commands for this frame
    this._ctx.device.queue.submit([commandEncoder.finish()]);
  }

  dispose(): void {
    for (const node of this._nodes.values()) {
      node.dispose();
    }
    // Release edge buffers
    for (const edge of this._edges) {
      if (edge.bufferId) {
        try {
          this._ctx.bufferManager.release(edge.bufferId);
        } catch {
          // Buffer may already be released
        }
      }
    }
  }

  private _createExecutionContext(
    nodeId: NodeId,
    commandEncoder: GPUCommandEncoder,
    frameIndex: number,
    timestampMs: number,
  ): ExecutionContext {
    const bm = this._ctx.bufferManager;

    return {
      device: this._ctx.device,
      bufferManager: bm,
      resourceTracker: this._ctx.resourceTracker,
      commandEncoder,
      frameIndex,
      timestampMs,

      getInput: (name: string): GPUBuffer | ArrayBuffer => {
        const edge = this._edges.find(
          (e) => e.targetNode === nodeId && e.targetPort === name,
        );
        if (!edge?.bufferId) {
          throw new Error(`No input buffer bound for port "${name}" on node "${nodeId}".`);
        }
        return bm.getGpuBuffer(edge.bufferId);
      },

      getOutput: (name: string): GPUBuffer | ArrayBuffer => {
        const edge = this._edges.find(
          (e) => e.sourceNode === nodeId && e.sourcePort === name,
        );
        if (!edge?.bufferId) {
          throw new Error(`No output buffer bound for port "${name}" on node "${nodeId}".`);
        }
        return bm.getGpuBuffer(edge.bufferId);
      },
    };
  }
}
