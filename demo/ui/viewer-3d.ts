/**
 * 3D Viewer for JOSH pipeline output.
 *
 * Renders the reconstructed scene depth map and SMPL human mesh
 * using raw WebGPU render pipeline (no Three.js dependency).
 */
export class Viewer3D {
  private readonly _canvas: HTMLCanvasElement;
  private readonly _device: GPUDevice;
  private _context: GPUCanvasContext | null = null;
  private _renderPipeline: GPURenderPipeline | null = null;

  constructor(canvas: HTMLCanvasElement, device: GPUDevice) {
    this._canvas = canvas;
    this._device = device;
  }

  async initialize(): Promise<void> {
    this._context = this._canvas.getContext('webgpu')!;
    const format = navigator.gpu.getPreferredCanvasFormat();

    this._context.configure({
      device: this._device,
      format,
      alphaMode: 'opaque',
    });

    // Create a simple render pipeline for mesh visualization
    const shaderModule = this._device.createShaderModule({
      code: VIEWER_SHADER,
      label: 'viewer3d_shader',
    });

    this._renderPipeline = this._device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
        buffers: [
          {
            arrayStride: 12, // 3 × f32
            attributes: [{ format: 'float32x3', offset: 0, shaderLocation: 0 }],
          },
        ],
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
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
      label: 'viewer3d_pipeline',
    });
  }

  /**
   * Render a frame with the given mesh vertices.
   */
  render(vertexBuffer: GPUBuffer, vertexCount: number): void {
    if (!this._context || !this._renderPipeline) return;

    const depthTexture = this._device.createTexture({
      size: [this._canvas.width, this._canvas.height],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    const encoder = this._device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this._context.getCurrentTexture().createView(),
          clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    pass.setPipeline(this._renderPipeline);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.draw(vertexCount);
    pass.end();

    this._device.queue.submit([encoder.finish()]);
    depthTexture.destroy();
  }
}

const VIEWER_SHADER = /* wgsl */ `
struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec3<f32>,
}

@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> VertexOutput {
  var out: VertexOutput;
  // Simple orthographic projection for visualization
  out.position = vec4<f32>(pos.x * 0.5, pos.y * 0.5, pos.z * 0.1 + 0.5, 1.0);
  // Color by depth
  out.color = vec3<f32>(0.3, 0.6, 1.0) * (1.0 - pos.z * 0.5);
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(in.color, 1.0);
}
`;
