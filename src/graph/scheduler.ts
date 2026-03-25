import type { NodeId } from '../core/types.ts';
import type { Edge } from './edge.ts';

/**
 * Identifies nodes within an island that can execute concurrently.
 * Two nodes in the same island are independent if neither is an ancestor of the other.
 *
 * Returns an array of "waves" — each wave is a set of nodes that can run in parallel.
 */
export function computeParallelWaves(
  nodes: readonly NodeId[],
  edges: readonly Edge[],
): NodeId[][] {
  // Build in-degree map restricted to nodes in this set
  const nodeSet = new Set(nodes);
  const inDegree = new Map<NodeId, number>();
  const adjacency = new Map<NodeId, NodeId[]>();

  for (const nodeId of nodes) {
    inDegree.set(nodeId, 0);
    adjacency.set(nodeId, []);
  }

  for (const edge of edges) {
    if (nodeSet.has(edge.sourceNode) && nodeSet.has(edge.targetNode)) {
      adjacency.get(edge.sourceNode)!.push(edge.targetNode);
      inDegree.set(edge.targetNode, inDegree.get(edge.targetNode)! + 1);
    }
  }

  const waves: NodeId[][] = [];

  while (true) {
    // Collect all nodes with in-degree 0
    const wave: NodeId[] = [];
    for (const [nodeId, degree] of inDegree) {
      if (degree === 0) wave.push(nodeId);
    }

    if (wave.length === 0) break;

    waves.push(wave);

    // Remove processed nodes and update in-degrees
    for (const nodeId of wave) {
      inDegree.delete(nodeId);
      for (const neighbor of adjacency.get(nodeId) ?? []) {
        if (inDegree.has(neighbor)) {
          inDegree.set(neighbor, inDegree.get(neighbor)! - 1);
        }
      }
    }
  }

  return waves;
}
