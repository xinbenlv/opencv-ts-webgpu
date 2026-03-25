import type { BufferLayout, NodeId } from '../core/types.ts';
import type { BufferManager } from '../core/buffer-manager.ts';
import type { IResourceTracker } from '../core/resource-tracker.ts';

// ─── Port descriptors ───
// Declare what a node consumes and produces. Enables static graph analysis.

export interface PortDescriptor {
  readonly name: string;
  readonly layout: BufferLayout;
  readonly optional?: boolean;
}

// ─── Backend affinity ───

export type BackendHint = 'webgpu' | 'wasm' | 'auto';

// ─── Node context (available during initialization) ───

export interface NodeContext {
  readonly device: GPUDevice;
  readonly bufferManager: BufferManager;
  readonly resourceTracker: IResourceTracker;
}

// ─── Execution context (available during each frame) ───

export interface ExecutionContext extends NodeContext {
  getInput(name: string): GPUBuffer | ArrayBuffer;
  getOutput(name: string): GPUBuffer | ArrayBuffer;
  readonly commandEncoder: GPUCommandEncoder;
  readonly frameIndex: number;
  readonly timestampMs: number;
}

// ─── GComputeNode ───
// The core compute node interface. Every pipeline stage implements this.

export interface GComputeNode<
  TInputs extends readonly PortDescriptor[] = readonly PortDescriptor[],
  TOutputs extends readonly PortDescriptor[] = readonly PortDescriptor[],
> {
  readonly id: NodeId;
  readonly name: string;
  readonly backendHint: BackendHint;

  readonly inputs: TInputs;
  readonly outputs: TOutputs;

  /** Called once during graph compilation — acquire persistent resources. */
  initialize(ctx: NodeContext): Promise<void>;

  /** Called every frame. */
  execute(ctx: ExecutionContext): Promise<void>;

  /** Deterministic cleanup. */
  dispose(): void;
}
