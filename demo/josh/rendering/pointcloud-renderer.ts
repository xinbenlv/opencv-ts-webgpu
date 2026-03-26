/**
 * PointCloudRenderer — renders a set of coloured 3-D points using WebGPU.
 *
 * Each point is drawn as a screen-space quad (2 triangles) so it stays a
 * fixed pixel size regardless of depth.  The fragment shader discards pixels
 * outside the unit circle to give round points.
 *
 * Usage:
 *   const renderer = new PointCloudRenderer(device, canvas);
 *   renderer.setPoints(positions, colors);   // Float32Array [N*3] each
 *   renderer.render(camera);
 */

import vertWgsl from './pointcloud.vert.wgsl?raw';
import fragWgsl from './pointcloud.frag.wgsl?raw';
import type { RenderCamera } from './smpl-renderer';
import { mat4LookAt, mat4Multiply, mat4Perspective } from './math';

export type { RenderCamera };

/** Uniform layout: mvp (16) + point_size (1) + pad (3) = 20 floats = 80 bytes. */
const UNIFORM_FLOATS = 20;
const UNIFORM_BYTES  = UNIFORM_FLOATS * 4;

/** Default point size in pixels. */
const DEFAULT_POINT_SIZE = 4.0;

export class PointCloudRenderer {
  private readonly device: GPUDevice;
  private readonly canvas: HTMLCanvasElement;
  private readonly context: GPUCanvasContext;
  private readonly format: GPUTextureFormat;

  private pipeline: GPURenderPipeline | null = null;
  private uniformBuffer: GPUBuffer;
  private bindGroup: GPUBindGroup | null = null;
  private depthTexture: GPUTexture | null = null;

  private positionBuffer: GPUBuffer | null = null;
  private colorBuffer:    GPUBuffer | null = null;
  private pointCount = 0;

  /** Exposed so callers can tweak at runtime. */
  pointSize: number = DEFAULT_POINT_SIZE;

  constructor(device: GPUDevice, canvas: HTMLCanvasElement) {
    this.device = device;
    this.canvas = canvas;

    const ctx = canvas.getContext('webgpu');
    if (!ctx) throw new Error('Could not get WebGPU context from canvas');
    this.context = ctx;

    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format: this.format, alphaMode: 'premultiplied' });

    this.uniformBuffer = device.createBuffer({
      label: 'pc-uniforms',
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.buildPipeline();
    this.ensureDepthTexture();
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Upload point positions and colours.
   * @param positions  Float32Array of length N*3  (x0,y0,z0, x1,y1,z1, …)
   * @param colors     Float32Array of length N*3  (r0,g0,b0, …) values in [0,1]
   */
  setPoints(positions: Float32Array, colors: Float32Array): void {
    if (positions.length !== colors.length) {
      throw new Error('positions and colors must have the same length (N*3)');
    }
    this.pointCount = positions.length / 3;

    this.positionBuffer?.destroy();
    this.colorBuffer?.destroy();

    this.positionBuffer = this.uploadBuffer(positions, GPUBufferUsage.STORAGE);
    this.colorBuffer    = this.uploadBuffer(colors,    GPUBufferUsage.STORAGE);

    this.rebuildBindGroup();
  }

  /** Render one frame. */
  render(camera: RenderCamera): void {
    if (!this.pipeline || !this.bindGroup || this.pointCount === 0) return;

    this.ensureDepthTexture();
    this.writeUniforms(camera);

    const encoder = this.device.createCommandEncoder({ label: 'pc-render' });
    const colorView = this.context.getCurrentTexture().createView();

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: colorView,
        clearValue: { r: 0.08, g: 0.08, b: 0.12, a: 1.0 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: this.depthTexture!.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    // 6 vertices per point (two triangles)
    pass.draw(this.pointCount * 6);
    pass.end();

    this.device.queue.submit([encoder.finish()]);
  }

  dispose(): void {
    this.uniformBuffer.destroy();
    this.positionBuffer?.destroy();
    this.colorBuffer?.destroy();
    this.depthTexture?.destroy();
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  private buildPipeline(): void {
    const { device, format } = this;

    const bindGroupLayout = device.createBindGroupLayout({
      label: 'pc-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer:  { type: 'uniform'           } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer:  { type: 'read-only-storage'  } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer:  { type: 'read-only-storage'  } },
      ],
    });

    this.pipeline = device.createRenderPipeline({
      label: 'pc-pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      vertex: {
        module: device.createShaderModule({ label: 'pc-vert', code: vertWgsl }),
        entryPoint: 'main',
      },
      fragment: {
        module: device.createShaderModule({ label: 'pc-frag', code: fragWgsl }),
        entryPoint: 'main',
        targets: [{
          format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one',       dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    });

    this.rebuildBindGroup();
  }

  private rebuildBindGroup(): void {
    if (!this.pipeline || !this.positionBuffer || !this.colorBuffer) return;

    this.bindGroup = this.device.createBindGroup({
      label: 'pc-bg',
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer  } },
        { binding: 1, resource: { buffer: this.positionBuffer } },
        { binding: 2, resource: { buffer: this.colorBuffer    } },
      ],
    });
  }

  private ensureDepthTexture(): void {
    const w = this.canvas.width;
    const h = this.canvas.height;
    if (this.depthTexture && this.depthTexture.width === w && this.depthTexture.height === h) return;
    this.depthTexture?.destroy();
    this.depthTexture = this.device.createTexture({
      label: 'pc-depth',
      size: [w, h],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  private writeUniforms(camera: RenderCamera): void {
    const aspect = this.canvas.width / this.canvas.height;
    const proj   = mat4Perspective(camera.fovY, aspect, camera.near, camera.far);
    const view   = mat4LookAt(camera.eye, camera.target, camera.up);
    const mvp    = mat4Multiply(proj, view);

    const data = new Float32Array(UNIFORM_FLOATS);
    data.set(mvp, 0);
    data[16] = this.pointSize;
    // data[17-19] = padding

    this.device.queue.writeBuffer(this.uniformBuffer, 0, data);
  }

  private uploadBuffer(data: Float32Array, usage: GPUBufferUsageFlags): GPUBuffer {
    const buf = this.device.createBuffer({
      size: data.byteLength,
      usage: usage | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(buf.getMappedRange()).set(data);
    buf.unmap();
    return buf;
  }
}
