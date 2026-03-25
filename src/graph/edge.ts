import type { BufferId, NodeId } from '../core/types.ts';

/**
 * A typed edge connecting an output port of one node to an input port of another.
 * Carries shape metadata for compile-time validation.
 */
export interface Edge {
  readonly sourceNode: NodeId;
  readonly sourcePort: string;
  readonly targetNode: NodeId;
  readonly targetPort: string;
  /** Assigned during compilation — the buffer backing this edge. */
  bufferId?: BufferId;
}
