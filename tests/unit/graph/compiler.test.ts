import { describe, it, expect, vi } from 'vitest';
import { GraphBuilder } from '@graph/graph.ts';
import { GraphError, ShapeError } from '@core/errors.ts';
import { createNodeId, computeBufferLayout, dim } from '@core/types.ts';
import type { GComputeNode, NodeContext, PortDescriptor } from '@graph/node.ts';
import type { Shape2D } from '@core/types.ts';

// ─── Test helpers ───

function createMockNode(
  name: string,
  inputs: PortDescriptor[],
  outputs: PortDescriptor[],
  backend: 'webgpu' | 'wasm' = 'webgpu',
): GComputeNode {
  return {
    id: createNodeId(name),
    name,
    backendHint: backend,
    inputs,
    outputs,
    initialize: vi.fn().mockResolvedValue(undefined),
    execute: vi.fn().mockResolvedValue(undefined),
    dispose: vi.fn(),
  };
}

const layout32x32 = computeBufferLayout([dim(32), dim(32)] as Shape2D, 'f32');
const layout64x64 = computeBufferLayout([dim(64), dim(64)] as Shape2D, 'f32');

// Mock NodeContext — no real GPU device for unit tests
function mockNodeContext(): NodeContext {
  const mockBuffer = {
    destroy: vi.fn(),
    size: 0,
    usage: 0,
    mapState: 'unmapped' as const,
    label: '',
    getMappedRange: vi.fn(),
    mapAsync: vi.fn(),
    unmap: vi.fn(),
  };

  return {
    device: {} as GPUDevice,
    bufferManager: {
      acquire: vi.fn().mockReturnValue('buf_test0' as never),
      release: vi.fn(),
      getGpuBuffer: vi.fn().mockReturnValue(mockBuffer),
      getCpuBuffer: vi.fn().mockReturnValue(new ArrayBuffer(0)),
      gpuToCpu: vi.fn(),
      cpuToGpu: vi.fn(),
      swap: vi.fn(),
      getFront: vi.fn(),
      getBack: vi.fn(),
      createDoubleBuffer: vi.fn(),
      allocatedBytes: 0,
      pooledBytes: 0,
      dispose: vi.fn(),
    } as never,
    resourceTracker: {
      track: vi.fn((r: unknown) => r),
      createScope: vi.fn(),
      untrack: vi.fn(),
      dispose: vi.fn(),
      isDisposed: false,
      trackedCount: 0,
    } as never,
  };
}

describe('GraphCompiler', () => {
  it('should compile a simple two-node graph', async () => {
    const nodeA = createMockNode(
      'A',
      [],
      [{ name: 'out', layout: layout32x32 }],
    );
    const nodeB = createMockNode(
      'B',
      [{ name: 'in', layout: layout32x32 }],
      [],
    );

    const graph = new GraphBuilder('test')
      .addNode(nodeA)
      .addNode(nodeB)
      .connect(nodeA.id, 'out', nodeB.id, 'in');

    const compiled = await graph.compile(mockNodeContext());

    expect(compiled.executionOrder).toHaveLength(2);
    expect(compiled.executionOrder[0]).toBe(nodeA.id);
    expect(compiled.executionOrder[1]).toBe(nodeB.id);
    expect(nodeA.initialize).toHaveBeenCalled();
    expect(nodeB.initialize).toHaveBeenCalled();
  });

  it('should detect duplicate node IDs', () => {
    const nodeA = createMockNode('A', [], []);
    const graph = new GraphBuilder('test').addNode(nodeA);

    expect(() => graph.addNode(nodeA)).toThrow(GraphError);
  });

  it('should detect references to non-existent nodes', () => {
    const nodeA = createMockNode('A', [], []);
    const fakeId = createNodeId('fake');
    const graph = new GraphBuilder('test').addNode(nodeA);

    expect(() => graph.connect(nodeA.id, 'out', fakeId, 'in')).toThrow(GraphError);
  });

  it('should detect shape mismatches between connected ports', async () => {
    const nodeA = createMockNode(
      'A',
      [],
      [{ name: 'out', layout: layout32x32 }],
    );
    const nodeB = createMockNode(
      'B',
      [{ name: 'in', layout: layout64x64 }], // different size!
      [],
    );

    const graph = new GraphBuilder('test')
      .addNode(nodeA)
      .addNode(nodeB)
      .connect(nodeA.id, 'out', nodeB.id, 'in');

    await expect(graph.compile(mockNodeContext())).rejects.toThrow(ShapeError);
  });

  it('should detect cycles in the graph', async () => {
    const nodeA = createMockNode(
      'A',
      [{ name: 'in', layout: layout32x32 }],
      [{ name: 'out', layout: layout32x32 }],
    );
    const nodeB = createMockNode(
      'B',
      [{ name: 'in', layout: layout32x32 }],
      [{ name: 'out', layout: layout32x32 }],
    );

    const graph = new GraphBuilder('test')
      .addNode(nodeA)
      .addNode(nodeB)
      .connect(nodeA.id, 'out', nodeB.id, 'in')
      .connect(nodeB.id, 'out', nodeA.id, 'in'); // cycle!

    await expect(graph.compile(mockNodeContext())).rejects.toThrow(GraphError);
  });

  it('should produce correct island fusion for mixed backends', async () => {
    const gpuNode1 = createMockNode(
      'gpu1',
      [],
      [{ name: 'out', layout: layout32x32 }],
      'webgpu',
    );
    const gpuNode2 = createMockNode(
      'gpu2',
      [],
      [{ name: 'out', layout: layout32x32 }],
      'webgpu',
    );
    const wasmNode = createMockNode(
      'wasm1',
      [
        { name: 'in1', layout: layout32x32 },
        { name: 'in2', layout: layout32x32 },
      ],
      [],
      'wasm',
    );

    const graph = new GraphBuilder('test')
      .addNode(gpuNode1)
      .addNode(gpuNode2)
      .addNode(wasmNode)
      .connect(gpuNode1.id, 'out', wasmNode.id, 'in1')
      .connect(gpuNode2.id, 'out', wasmNode.id, 'in2');

    const compiled = await graph.compile(mockNodeContext());

    expect(compiled.islands.length).toBeGreaterThanOrEqual(1);
    // The GPU nodes should be in an island, WASM node in another
    const gpuIsland = compiled.islands.find((i) => i.backend === 'webgpu');
    const wasmIsland = compiled.islands.find((i) => i.backend === 'wasm');
    expect(gpuIsland).toBeDefined();
    expect(wasmIsland).toBeDefined();
  });
});
