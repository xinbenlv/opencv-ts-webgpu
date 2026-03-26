import type { GraphId, NodeId, BufferId } from '../core/types.ts';
import type { GComputeNode, ExecutionContext, NodeContext } from './node.ts';
import type { Edge } from './edge.ts';
import type { CompiledGraph, Island } from './compiler.ts';

/**
 * Runtime executor for a compiled graph.
 * Dispatches nodes in topological order, managing buffer bindings per frame.
 */
export class GraphExecutor implements CompiledGraph {
  private readonly _graphInputBufferMap: Map<string, BufferId>;
  private readonly _graphOutputBufferMap: Map<string, BufferId>;

  constructor(
    readonly id: GraphId,
    private readonly _nodes: ReadonlyMap<NodeId, GComputeNode>,
    private readonly _edges: readonly Edge[],
    readonly executionOrder: readonly NodeId[],
    readonly islands: readonly Island[],
    readonly graphInputs: ReadonlyArray<{ nodeId: NodeId; portName: string; bufferId?: string }>,
    readonly graphOutputs: ReadonlyArray<{ nodeId: NodeId; portName: string; bufferId?: string }>,
    private readonly _ctx: NodeContext,
  ) {
    this._graphInputBufferMap = new Map(
      graphInputs
        .filter((gi) => gi.bufferId != null)
        .map((gi) => [`${gi.nodeId}:${gi.portName}`, gi.bufferId! as BufferId]),
    );
    this._graphOutputBufferMap = new Map(
      graphOutputs
        .filter((go) => go.bufferId != null)
        .map((go) => [`${go.nodeId}:${go.portName}`, go.bufferId! as BufferId]),
    );
  }

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

  writeInput(nodeId: NodeId, portName: string, data: ArrayBufferLike): void {
    const bufferId = this._graphInputBufferMap.get(`${nodeId}:${portName}`);
    if (!bufferId) {
      throw new Error(`No graph input buffer for port "${portName}" on node "${nodeId}".`);
    }
    const gpuBuffer = this._ctx.bufferManager.getGpuBuffer(bufferId);
    this._ctx.device.queue.writeBuffer(gpuBuffer, 0, data);
  }

  async readOutput(nodeId: NodeId, portName: string): Promise<ArrayBuffer> {
    // Check graph output buffers first
    const graphBufferId = this._graphOutputBufferMap.get(`${nodeId}:${portName}`);
    if (graphBufferId) {
      return this._ctx.bufferManager.gpuToCpu(graphBufferId);
    }
    // Fall back to edge buffers (for intermediate node outputs)
    const edge = this._edges.find(
      (e) => e.sourceNode === nodeId && e.sourcePort === portName,
    );
    if (edge?.bufferId) {
      return this._ctx.bufferManager.gpuToCpu(edge.bufferId);
    }
    throw new Error(`No output buffer for port "${portName}" on node "${nodeId}".`);
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
        // Check graph input buffers first
        const graphInputBufferId = this._graphInputBufferMap.get(`${nodeId}:${name}`);
        if (graphInputBufferId) {
          return bm.getGpuBuffer(graphInputBufferId);
        }
        const edge = this._edges.find(
          (e) => e.targetNode === nodeId && e.targetPort === name,
        );
        if (!edge?.bufferId) {
          throw new Error(`No input buffer bound for port "${name}" on node "${nodeId}".`);
        }
        return bm.getGpuBuffer(edge.bufferId);
      },

      getOutput: (name: string): GPUBuffer | ArrayBuffer => {
        // Check graph output buffers first
        const graphOutputBufferId = this._graphOutputBufferMap.get(`${nodeId}:${name}`);
        if (graphOutputBufferId) {
          return bm.getGpuBuffer(graphOutputBufferId);
        }
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
