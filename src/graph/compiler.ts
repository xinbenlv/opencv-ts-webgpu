import type { GraphId, NodeId } from '../core/types.ts';
import { GraphError, ShapeError } from '../core/errors.ts';
import type { GComputeNode, NodeContext } from './node.ts';
import type { Edge } from './edge.ts';
import { GraphExecutor } from './executor.ts';

/**
 * An Island is a maximal subgraph of nodes sharing the same backend.
 * Mirrors the OpenCV G-API island fusion optimization.
 */
export interface Island {
  readonly backend: 'webgpu' | 'wasm';
  readonly nodes: readonly NodeId[];
}

/**
 * A compiled, executable graph.
 */
export interface CompiledGraph {
  readonly id: GraphId;
  readonly executionOrder: readonly NodeId[];
  readonly islands: readonly Island[];
  readonly graphInputs: ReadonlyArray<{ nodeId: NodeId; portName: string }>;
  readonly graphOutputs: ReadonlyArray<{ nodeId: NodeId; portName: string }>;
  execute(frameIndex: number): Promise<void>;
  writeInput(nodeId: NodeId, portName: string, data: ArrayBufferLike): void;
  readOutput(nodeId: NodeId, portName: string): Promise<ArrayBuffer>;
  dispose(): void;
}

/**
 * Three-pass graph compiler:
 * 1. Validation — cycle detection, port connectivity, shape compatibility
 * 2. Island fusion — group nodes by backend into maximal connected subgraphs
 * 3. Buffer allocation — assign BufferIds to edges, maximize reuse
 */
export class GraphCompiler {
  constructor(
    private readonly _graphId: GraphId,
    private readonly _nodes: ReadonlyMap<NodeId, GComputeNode>,
    private readonly _edges: readonly Edge[],
    private readonly _graphInputs: ReadonlyArray<{ nodeId: NodeId; portName: string }>,
    private readonly _graphOutputs: ReadonlyArray<{ nodeId: NodeId; portName: string }>,
  ) {}

  async compile(ctx: NodeContext): Promise<CompiledGraph> {
    // Pass 1: Validate
    this._validateConnectivity();
    const order = this._topologicalSort();
    this._validateShapes();

    // Pass 2: Island fusion
    const islands = this._fuseIslands(order);

    // Pass 3: Allocate buffers
    this._allocateBuffers(ctx, order);

    // Initialize all nodes
    for (const nodeId of order) {
      const node = this._nodes.get(nodeId)!;
      await node.initialize(ctx);
    }

    return new GraphExecutor(
      this._graphId,
      this._nodes,
      this._edges,
      order,
      islands,
      this._graphInputs,
      this._graphOutputs,
      ctx,
    );
  }

  // ─── Pass 1: Validation ───

  private _validateConnectivity(): void {
    for (const edge of this._edges) {
      const sourceNode = this._nodes.get(edge.sourceNode);
      const targetNode = this._nodes.get(edge.targetNode);
      if (!sourceNode) {
        throw new GraphError(`Edge references unknown source node "${edge.sourceNode}".`);
      }
      if (!targetNode) {
        throw new GraphError(`Edge references unknown target node "${edge.targetNode}".`);
      }

      const sourcePort = sourceNode.outputs.find((p) => p.name === edge.sourcePort);
      if (!sourcePort) {
        throw new GraphError(
          `Node "${sourceNode.name}" has no output port "${edge.sourcePort}".`,
        );
      }

      const targetPort = targetNode.inputs.find((p) => p.name === edge.targetPort);
      if (!targetPort) {
        throw new GraphError(
          `Node "${targetNode.name}" has no input port "${edge.targetPort}".`,
        );
      }
    }
  }

  private _validateShapes(): void {
    for (const edge of this._edges) {
      const sourceNode = this._nodes.get(edge.sourceNode)!;
      const targetNode = this._nodes.get(edge.targetNode)!;

      const sourcePort = sourceNode.outputs.find((p) => p.name === edge.sourcePort)!;
      const targetPort = targetNode.inputs.find((p) => p.name === edge.targetPort)!;

      if (sourcePort.layout.byteLength !== targetPort.layout.byteLength) {
        throw new ShapeError(
          `Shape mismatch on edge ${sourceNode.name}.${edge.sourcePort} → ` +
            `${targetNode.name}.${edge.targetPort}: ` +
            `${sourcePort.layout.byteLength} bytes vs ${targetPort.layout.byteLength} bytes.`,
        );
      }

      if (sourcePort.layout.dtype !== targetPort.layout.dtype) {
        throw new ShapeError(
          `DType mismatch on edge ${sourceNode.name}.${edge.sourcePort} → ` +
            `${targetNode.name}.${edge.targetPort}: ` +
            `${sourcePort.layout.dtype} vs ${targetPort.layout.dtype}.`,
        );
      }
    }
  }

  // ─── Topological sort with cycle detection ───

  private _topologicalSort(): NodeId[] {
    const inDegree = new Map<NodeId, number>();
    const adjacency = new Map<NodeId, NodeId[]>();

    for (const nodeId of this._nodes.keys()) {
      inDegree.set(nodeId, 0);
      adjacency.set(nodeId, []);
    }

    for (const edge of this._edges) {
      adjacency.get(edge.sourceNode)!.push(edge.targetNode);
      inDegree.set(edge.targetNode, inDegree.get(edge.targetNode)! + 1);
    }

    const queue: NodeId[] = [];
    for (const [nodeId, degree] of inDegree) {
      if (degree === 0) queue.push(nodeId);
    }

    const order: NodeId[] = [];
    while (queue.length > 0) {
      const current = queue.shift()!;
      order.push(current);
      for (const neighbor of adjacency.get(current)!) {
        const newDegree = inDegree.get(neighbor)! - 1;
        inDegree.set(neighbor, newDegree);
        if (newDegree === 0) queue.push(neighbor);
      }
    }

    if (order.length !== this._nodes.size) {
      throw new GraphError(
        `Graph contains a cycle. Processed ${order.length} of ${this._nodes.size} nodes.`,
      );
    }

    return order;
  }

  // ─── Pass 2: Island fusion ───

  private _fuseIslands(order: readonly NodeId[]): Island[] {
    const islands: Island[] = [];
    let currentIsland: { backend: 'webgpu' | 'wasm'; nodes: NodeId[] } | null = null;

    for (const nodeId of order) {
      const node = this._nodes.get(nodeId)!;
      const backend = node.backendHint === 'auto' ? 'webgpu' : node.backendHint;

      if (currentIsland && currentIsland.backend === backend) {
        currentIsland.nodes.push(nodeId);
      } else {
        if (currentIsland) islands.push(currentIsland);
        currentIsland = { backend, nodes: [nodeId] };
      }
    }

    if (currentIsland) islands.push(currentIsland);
    return islands;
  }

  // ─── Pass 3: Buffer allocation ───

  private _allocateBuffers(
    ctx: NodeContext,
    order: readonly NodeId[],
  ): void {
    // Allocate buffers for graph input ports
    for (const graphInput of this._graphInputs) {
      const node = this._nodes.get(graphInput.nodeId)!;
      const port = node.inputs.find((p) => p.name === graphInput.portName)!;
      const id = ctx.bufferManager.acquire(
        port.layout,
        `graphInput.${graphInput.nodeId}.${graphInput.portName}`,
      );
      (graphInput as { bufferId?: typeof id }).bufferId = id;
    }

    // Allocate buffers for graph output ports
    for (const graphOutput of this._graphOutputs) {
      const node = this._nodes.get(graphOutput.nodeId)!;
      const port = node.outputs.find((p) => p.name === graphOutput.portName)!;
      const id = ctx.bufferManager.acquire(
        port.layout,
        `graphOutput.${graphOutput.nodeId}.${graphOutput.portName}`,
      );
      (graphOutput as { bufferId?: typeof id }).bufferId = id;
    }

    for (const edge of this._edges) {
      const sourceNode = this._nodes.get(edge.sourceNode)!;
      const sourcePort = sourceNode.outputs.find((p) => p.name === edge.sourcePort)!;
      const id = ctx.bufferManager.acquire(
        sourcePort.layout,
        `${edge.sourceNode}.${edge.sourcePort}→${edge.targetNode}.${edge.targetPort}`,
      );
      (edge as { bufferId?: typeof id }).bufferId = id;
    }

    // Track last consumer for each buffer to enable future reuse
    const _lastConsumer = new Map<string, number>();
    for (let i = 0; i < order.length; i++) {
      const nodeId = order[i]!;
      for (const edge of this._edges) {
        if (edge.targetNode === nodeId && edge.bufferId) {
          _lastConsumer.set(edge.bufferId, i);
        }
      }
    }
  }
}
