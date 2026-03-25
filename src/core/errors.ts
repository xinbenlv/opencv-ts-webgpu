/**
 * Base error class for all OpenCV.ts errors.
 */
export class CvError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'CvError';
  }
}

/**
 * Thrown when a resource is accessed after disposal.
 */
export class DisposedError extends CvError {
  constructor(resourceName: string) {
    super(`Resource "${resourceName}" has already been disposed.`);
    this.name = 'DisposedError';
  }
}

/**
 * Thrown on buffer pool exhaustion or invalid buffer access.
 */
export class BufferError extends CvError {
  constructor(message: string) {
    super(message);
    this.name = 'BufferError';
  }
}

/**
 * Thrown when graph compilation detects structural issues.
 */
export class GraphError extends CvError {
  constructor(message: string) {
    super(message);
    this.name = 'GraphError';
  }
}

/**
 * Thrown when tensor shapes are incompatible for an operation.
 */
export class ShapeError extends CvError {
  constructor(message: string) {
    super(message);
    this.name = 'ShapeError';
  }
}

/**
 * Thrown when a WebGPU operation fails.
 */
export class GpuError extends CvError {
  constructor(message: string) {
    super(message);
    this.name = 'GpuError';
  }
}

/**
 * Thrown when a WASM backend operation fails.
 */
export class WasmError extends CvError {
  constructor(message: string) {
    super(message);
    this.name = 'WasmError';
  }
}
