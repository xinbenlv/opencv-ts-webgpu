import type { GraphId, NodeId } from '../core/types.ts';
import { createGraphId } from '../core/types.ts';
import { GraphError } from '../core/errors.ts';
import type { GComputeNode, NodeContext } from './node.ts';
import type { Edge } from './edge.ts';
import { GraphCompiler, type CompiledGraph } from './compiler.ts';

/**
 * Fluent builder for constructing a compute graph.
 */
export interface IGraphBuilder {
  addNode(node: GComputeNode): IGraphBuilder;
  connect(
    source: NodeId,
    sourcePort: string,
    target: NodeId,
    targetPort: string,
  ): IGraphBuilder;
  setInput(nodeId: NodeId, portName: string): IGraphBuilder;
  setOutput(nodeId: NodeId, portName: string): IGraphBuilder;
  compile(ctx: NodeContext): Promise<CompiledGraph>;
}

export class GraphBuilder implements IGraphBuilder {
  readonly id: GraphId;
  private readonly _nodes = new Map<NodeId, GComputeNode>();
  private readonly _edges: Edge[] = [];
  private readonly _graphInputs: Array<{ nodeId: NodeId; portName: string }> = [];
  private readonly _graphOutputs: Array<{ nodeId: NodeId; portName: string }> = [];

  constructor(name: string) {
    this.id = createGraphId(name);
  }

  addNode(node: GComputeNode): this {
    if (this._nodes.has(node.id)) {
      throw new GraphError(`Node "${node.id}" already exists in graph.`);
    }
    this._nodes.set(node.id, node);
    return this;
  }

  connect(
    source: NodeId,
    sourcePort: string,
    target: NodeId,
    targetPort: string,
  ): this {
    this._assertNodeExists(source);
    this._assertNodeExists(target);
    this._edges.push({ sourceNode: source, sourcePort, targetNode: target, targetPort });
    return this;
  }

  setInput(nodeId: NodeId, portName: string): this {
    this._assertNodeExists(nodeId);
    this._graphInputs.push({ nodeId, portName });
    return this;
  }

  setOutput(nodeId: NodeId, portName: string): this {
    this._assertNodeExists(nodeId);
    this._graphOutputs.push({ nodeId, portName });
    return this;
  }

  async compile(ctx: NodeContext): Promise<CompiledGraph> {
    const compiler = new GraphCompiler(
      this.id,
      this._nodes,
      this._edges,
      this._graphInputs,
      this._graphOutputs,
    );
    return compiler.compile(ctx);
  }

  private _assertNodeExists(id: NodeId): void {
    if (!this._nodes.has(id)) {
      throw new GraphError(`Node "${id}" not found in graph.`);
    }
  }
}

export function createGraphBuilder(name: string): GraphBuilder {
  return new GraphBuilder(name);
}
