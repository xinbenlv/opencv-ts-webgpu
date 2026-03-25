import { describe, it, expect } from 'vitest';
import { computeParallelWaves } from '@graph/scheduler.ts';
import { createNodeId } from '@core/types.ts';
import type { Edge } from '@graph/edge.ts';

describe('computeParallelWaves', () => {
  it('should identify independent nodes as a single wave', () => {
    const a = createNodeId('a');
    const b = createNodeId('b');
    const c = createNodeId('c');

    const waves = computeParallelWaves([a, b, c], []);

    expect(waves).toHaveLength(1);
    expect(waves[0]).toHaveLength(3);
    expect(waves[0]).toContain(a);
    expect(waves[0]).toContain(b);
    expect(waves[0]).toContain(c);
  });

  it('should create sequential waves for a chain', () => {
    const a = createNodeId('a');
    const b = createNodeId('b');
    const c = createNodeId('c');

    const edges: Edge[] = [
      { sourceNode: a, sourcePort: 'out', targetNode: b, targetPort: 'in' },
      { sourceNode: b, sourcePort: 'out', targetNode: c, targetPort: 'in' },
    ];

    const waves = computeParallelWaves([a, b, c], edges);

    expect(waves).toHaveLength(3);
    expect(waves[0]).toEqual([a]);
    expect(waves[1]).toEqual([b]);
    expect(waves[2]).toEqual([c]);
  });

  it('should parallelize a diamond DAG', () => {
    //     A
    //    / \
    //   B   C
    //    \ /
    //     D
    const a = createNodeId('a');
    const b = createNodeId('b');
    const c = createNodeId('c');
    const d = createNodeId('d');

    const edges: Edge[] = [
      { sourceNode: a, sourcePort: 'out', targetNode: b, targetPort: 'in' },
      { sourceNode: a, sourcePort: 'out', targetNode: c, targetPort: 'in' },
      { sourceNode: b, sourcePort: 'out', targetNode: d, targetPort: 'in' },
      { sourceNode: c, sourcePort: 'out', targetNode: d, targetPort: 'in' },
    ];

    const waves = computeParallelWaves([a, b, c, d], edges);

    expect(waves).toHaveLength(3);
    expect(waves[0]).toEqual([a]);
    expect(waves[1]).toHaveLength(2);
    expect(waves[1]).toContain(b);
    expect(waves[1]).toContain(c);
    expect(waves[2]).toEqual([d]);
  });
});
