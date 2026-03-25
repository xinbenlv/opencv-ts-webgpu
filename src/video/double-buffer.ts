import type { BufferLayout, BufferId } from '../core/types.ts';
import type { BufferManager } from '../core/buffer-manager.ts';
import type { Disposable } from '../core/resource-tracker.ts';

/**
 * Double-buffer (ping-pong) wrapper for real-time video pipelines.
 *
 * Allows GPU to process frame N+1 while frame N's results are being
 * read back or rendered.
 */
export class DoubleBuffer implements Disposable {
  private readonly _slotName: string;
  private readonly _bufferManager: BufferManager;
  private _frame = 0;

  constructor(
    bufferManager: BufferManager,
    slotName: string,
    layout: BufferLayout,
  ) {
    this._bufferManager = bufferManager;
    this._slotName = slotName;
    bufferManager.createDoubleBuffer(slotName, layout);
  }

  /**
   * Get the buffer to write the current frame into.
   */
  get writeBuffer(): BufferId {
    return this._bufferManager.getBack(this._slotName);
  }

  /**
   * Get the buffer containing the previous frame's results.
   */
  get readBuffer(): BufferId {
    return this._bufferManager.getFront(this._slotName);
  }

  /**
   * Swap buffers — call at the end of each frame.
   */
  swap(): void {
    this._bufferManager.swap(this._slotName);
    this._frame++;
  }

  get frameCount(): number {
    return this._frame;
  }

  dispose(): void {
    // BufferManager owns the buffers — nothing to do here
  }
}
