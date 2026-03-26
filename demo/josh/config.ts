/** JOSH paper hyperparameters (exact values from paper) */
export const JOSH_CONFIG = {
  // Two-stage Adam optimization
  stage1: {
    iters: 500,
    lr: 0.07,
    w3D: 1,
    w2D: 0, // L_2D disabled in stage 1
    wc1: 1,
    wc2: 20,
    wp: 10,
    ws: 1,
  },
  stage2: {
    iters: 200,
    lr: 0.014,
    w3D: 1,
    w2D: 1, // L_2D enabled in stage 2
    wc1: 1,
    wc2: 20,
    wp: 10,
    ws: 1,
  },
  // Contact thresholds
  deltaC1: 0, // contact scale threshold (exact contact)
  deltaC2: 0.1, // contact static threshold (10cm)
  contactThreshold: 0.05, // 5cm for contact detection

  // Temporal chunking
  chunkSize: 100, // frames per chunk
  keyframeInterval: 0.2, // seconds (5 FPS → every frame is a keyframe)
  targetFps: 5, // video sampling rate

  // Camera (default for 384×384 frames)
  defaultFx: 300,
  defaultFy: 300,
  defaultCx: 192,
  defaultCy: 192,

  // Model dimensions
  smplVertexCount: 6890,
  smplJointCount: 24,
  smplPoseDim: 72,
  smplShapeDim: 10,
  paramDim: 89, // 72+10+3+3+1

  // Inference
  depthMapSize: 384,
  moveNetInputSize: 192,
  rompInputSize: 512,
} as const;

export type JoshConfig = typeof JOSH_CONFIG;
