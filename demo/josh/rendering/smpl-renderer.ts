/**
 * SmplRenderer — WebGPU-based 3D mesh renderer for the SMPL body model.
 *
 * Usage:
 *   const renderer = new SmplRenderer(device, canvas);
 *   renderer.setMesh(vertices, normals, faces);
 *   // per-frame:
 *   renderer.updateVertices(posedVertices);
 *   renderer.render(camera);
 */

import vertWgsl from './smpl-mesh.vert.wgsl?raw';
import fragWgsl from './smpl-mesh.frag.wgsl?raw';
import { mat4Identity, mat4LookAt, mat4Multiply, mat4Perspective } from './math';

export interface RenderCamera {
  eye: [number, number, number];
  target: [number, number, number];
  up: [number, number, number];
  fovY: number;
  near: number;
  far: number;
}

/** Byte size of the uniform block (mvp + model + light_dir + pad = 16+16+4 floats = 144 bytes). */
const UNIFORM_FLOATS = 16 + 16 + 4;
const UNIFORM_BYTES = UNIFORM_FLOATS * 4;

export class SmplRenderer {
  private readonly device: GPUDevice;
  private readonly canvas: HTMLCanvasElement;
  private readonly context: GPUCanvasContext;
  private readonly format: GPUTextureFormat;

  private pipeline: GPURenderPipeline | null = null;
  private uniformBuffer: GPUBuffer;
  private bindGroup: GPUBindGroup | null = null;
  private depthTexture: GPUTexture | null = null;

  // Mesh buffers
  private vertexBuffer: GPUBuffer | null = null;
  private normalBuffer: GPUBuffer | null = null;
  private indexBuffer: GPUBuffer | null = null;
  private indexCount = 0;
  private vertexCount = 0;

  constructor(device: GPUDevice, canvas: HTMLCanvasElement) {
    this.device = device;
    this.canvas = canvas;

    const ctx = canvas.getContext('webgpu');
    if (!ctx) throw new Error('Could not get WebGPU context from canvas');
    this.context = ctx;

    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format: this.format, alphaMode: 'premultiplied' });

    this.uniformBuffer = device.createBuffer({
      label: 'smpl-uniforms',
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.buildPipeline();
    this.ensureDepthTexture();
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /** Upload mesh topology and normals. Call once (or when mesh changes). */
  setMesh(vertices: Float32Array, normals: Float32Array, faces: Uint32Array): void {
    this.vertexCount = vertices.length / 3;
    this.indexCount = faces.length;

    this.vertexBuffer?.destroy();
    this.normalBuffer?.destroy();
    this.indexBuffer?.destroy();

    this.vertexBuffer = this.uploadBuffer(vertices, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST);
    this.normalBuffer = this.uploadBuffer(normals, GPUBufferUsage.VERTEX);
    this.indexBuffer  = this.uploadBuffer(faces,    GPUBufferUsage.INDEX);

    this.rebuildBindGroup();
  }

  /** Fast-path: only updates the vertex positions (e.g. each posed frame). */
  updateVertices(vertices: Float32Array): void {
    if (!this.vertexBuffer) {
      throw new Error('Call setMesh() before updateVertices()');
    }
    this.device.queue.writeBuffer(this.vertexBuffer, 0, vertices);
  }

  /** Render one frame to the canvas. */
  render(camera: RenderCamera): void {
    if (!this.pipeline || !this.bindGroup || !this.vertexBuffer || !this.normalBuffer || !this.indexBuffer) {
      return; // nothing to draw yet
    }

    this.ensureDepthTexture();
    this.writeUniforms(camera);

    const encoder = this.device.createCommandEncoder({ label: 'smpl-render' });
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
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setVertexBuffer(1, this.normalBuffer);
    pass.setIndexBuffer(this.indexBuffer, 'uint32');
    pass.drawIndexed(this.indexCount);
    pass.end();

    this.device.queue.submit([encoder.finish()]);
  }

  dispose(): void {
    this.uniformBuffer.destroy();
    this.vertexBuffer?.destroy();
    this.normalBuffer?.destroy();
    this.indexBuffer?.destroy();
    this.depthTexture?.destroy();
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  private buildPipeline(): void {
    const { device, format } = this;

    const bindGroupLayout = device.createBindGroupLayout({
      label: 'smpl-bgl',
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      }],
    });

    const vertModule = device.createShaderModule({ label: 'smpl-vert', code: vertWgsl });
    const fragModule = device.createShaderModule({ label: 'smpl-frag', code: fragWgsl });

    this.pipeline = device.createRenderPipeline({
      label: 'smpl-pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      vertex: {
        module: vertModule,
        entryPoint: 'main',
        buffers: [
          // slot 0: positions (xyz)
          {
            arrayStride: 12,
            attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
          },
          // slot 1: normals (xyz)
          {
            arrayStride: 12,
            attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }],
          },
        ],
      },
      fragment: {
        module: fragModule,
        entryPoint: 'main',
        targets: [{ format }],
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'back',
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    });

    // Rebuild bind group now that pipeline layout exists (uniform buffer is already created)
    this.rebuildBindGroup();
  }

  private rebuildBindGroup(): void {
    if (!this.pipeline) return;

    this.bindGroup = this.device.createBindGroup({
      label: 'smpl-bg',
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }],
    });
  }

  private ensureDepthTexture(): void {
    const w = this.canvas.width;
    const h = this.canvas.height;
    if (this.depthTexture && this.depthTexture.width === w && this.depthTexture.height === h) return;
    this.depthTexture?.destroy();
    this.depthTexture = this.device.createTexture({
      label: 'smpl-depth',
      size: [w, h],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  private writeUniforms(camera: RenderCamera): void {
    const { canvas } = this;
    const aspect = canvas.width / canvas.height;

    const proj = mat4Perspective(camera.fovY, aspect, camera.near, camera.far);
    const view = mat4LookAt(camera.eye, camera.target, camera.up);
    const model = mat4Identity();
    const mvp = mat4Multiply(proj, mat4Multiply(view, model));

    // Directional light: slightly from above-right
    const lightDir = new Float32Array([0.6, 1.0, 0.8]);
    const len = Math.sqrt(lightDir[0] ** 2 + lightDir[1] ** 2 + lightDir[2] ** 2);
    lightDir[0] /= len; lightDir[1] /= len; lightDir[2] /= len;

    const data = new Float32Array(UNIFORM_FLOATS);
    data.set(mvp, 0);
    data.set(model, 16);
    data.set(lightDir, 32);
    // data[35] = _pad (0.0)

    this.device.queue.writeBuffer(this.uniformBuffer, 0, data);
  }

  private uploadBuffer(data: Float32Array | Uint32Array, usage: GPUBufferUsageFlags): GPUBuffer {
    const buf = this.device.createBuffer({
      size: data.byteLength,
      usage: usage | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    const dst = data instanceof Float32Array
      ? new Float32Array(buf.getMappedRange())
      : new Uint32Array(buf.getMappedRange());
    dst.set(data);
    buf.unmap();
    return buf;
  }
}
