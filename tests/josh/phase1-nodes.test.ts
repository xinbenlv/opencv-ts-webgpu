/**
 * Phase 1 node tests.
 *
 * All tests run in Node.js (via vitest) — no browser APIs are required.
 * Tests cover:
 *   1. JOSH_CONFIG structure and values
 *   2. COCO_TO_SMPL mapping validity
 *   3. SegmentationNode fallback (HSV skin-colour heuristic)
 *   4. FrameExtractor instantiation (type-level smoke test)
 */

import { describe, it, expect } from 'vitest';
import { JOSH_CONFIG } from '../../demo/josh/config.ts';
import { COCO_TO_SMPL, COCO_KEYPOINTS, Pose2DNode } from '../../demo/josh/nodes/pose-2d.node.ts';
import { SegmentationNode } from '../../demo/josh/nodes/segmentation.node.ts';
import { FrameExtractor } from '../../demo/josh/batch/frame-extractor.ts';

// ---------------------------------------------------------------------------
// 1. JOSH_CONFIG
// ---------------------------------------------------------------------------

describe('JOSH_CONFIG', () => {
  it('has all required top-level fields', () => {
    expect(JOSH_CONFIG).toHaveProperty('stage1');
    expect(JOSH_CONFIG).toHaveProperty('stage2');
    expect(JOSH_CONFIG).toHaveProperty('deltaC1');
    expect(JOSH_CONFIG).toHaveProperty('deltaC2');
    expect(JOSH_CONFIG).toHaveProperty('contactThreshold');
    expect(JOSH_CONFIG).toHaveProperty('chunkSize');
    expect(JOSH_CONFIG).toHaveProperty('targetFps');
    expect(JOSH_CONFIG).toHaveProperty('smplVertexCount');
    expect(JOSH_CONFIG).toHaveProperty('smplJointCount');
    expect(JOSH_CONFIG).toHaveProperty('paramDim');
  });

  it('stage1 and stage2 have all weight fields', () => {
    for (const stage of [JOSH_CONFIG.stage1, JOSH_CONFIG.stage2] as const) {
      expect(stage).toHaveProperty('iters');
      expect(stage).toHaveProperty('lr');
      expect(stage).toHaveProperty('w3D');
      expect(stage).toHaveProperty('w2D');
      expect(stage).toHaveProperty('wc1');
      expect(stage).toHaveProperty('wc2');
      expect(stage).toHaveProperty('wp');
      expect(stage).toHaveProperty('ws');
    }
  });

  it('stage1.iters + stage2.iters = 700', () => {
    expect(JOSH_CONFIG.stage1.iters + JOSH_CONFIG.stage2.iters).toBe(700);
  });

  it('deltaC2 = 0.1', () => {
    expect(JOSH_CONFIG.deltaC2).toBe(0.1);
  });

  it('w2D is 0 in stage1 and 1 in stage2', () => {
    expect(JOSH_CONFIG.stage1.w2D).toBe(0);
    expect(JOSH_CONFIG.stage2.w2D).toBe(1);
  });

  it('paramDim = smplPoseDim + smplShapeDim + 3 + 3 + 1', () => {
    const expected =
      JOSH_CONFIG.smplPoseDim +
      JOSH_CONFIG.smplShapeDim +
      3 + // global translation
      3 + // global rotation
      1; // scale
    expect(JOSH_CONFIG.paramDim).toBe(expected);
  });

  it('all numeric fields have type number', () => {
    // Spot-check a selection of fields
    expect(typeof JOSH_CONFIG.defaultFx).toBe('number');
    expect(typeof JOSH_CONFIG.depthMapSize).toBe('number');
    expect(typeof JOSH_CONFIG.moveNetInputSize).toBe('number');
    expect(typeof JOSH_CONFIG.contactThreshold).toBe('number');
  });
});

// ---------------------------------------------------------------------------
// 2. COCO_TO_SMPL mapping
// ---------------------------------------------------------------------------

describe('COCO_TO_SMPL mapping', () => {
  it('all SMPL joint indices are in range [0, 23]', () => {
    for (const [cocoIdx, smplIdx] of Object.entries(COCO_TO_SMPL)) {
      expect(smplIdx).toBeGreaterThanOrEqual(0);
      expect(smplIdx).toBeLessThanOrEqual(23);
      // cocoIdx must be a valid COCO keypoint index
      expect(Number(cocoIdx)).toBeGreaterThanOrEqual(0);
      expect(Number(cocoIdx)).toBeLessThan(COCO_KEYPOINTS.length);
    }
  });

  it('is a strict subset of 17 COCO keypoints', () => {
    const keys = Object.keys(COCO_TO_SMPL).map(Number);
    expect(keys.length).toBeLessThanOrEqual(COCO_KEYPOINTS.length);
    for (const k of keys) {
      expect(k).toBeGreaterThanOrEqual(0);
      expect(k).toBeLessThan(COCO_KEYPOINTS.length);
    }
  });

  it('maps nose (0) to SMPL head (15)', () => {
    expect(COCO_TO_SMPL[0]).toBe(15);
  });

  it('maps left_hip (11) to SMPL joint 1', () => {
    expect(COCO_TO_SMPL[11]).toBe(1);
  });

  it('maps right_hip (12) to SMPL joint 2', () => {
    expect(COCO_TO_SMPL[12]).toBe(2);
  });

  it('SMPL indices in mapping have no duplicates', () => {
    const smplIndices = Object.values(COCO_TO_SMPL);
    const unique = new Set(smplIndices);
    expect(unique.size).toBe(smplIndices.length);
  });
});

// ---------------------------------------------------------------------------
// 3. SegmentationNode — HSV skin-colour heuristic
// ---------------------------------------------------------------------------

/**
 * Build a synthetic ImageData for use in Node.js where the DOM ImageData
 * constructor is not available.  vitest's jsdom environment provides it, but
 * we implement a minimal polyfill just in case.
 */
function makeImageData(data: Uint8ClampedArray, width: number, height: number): ImageData {
  if (typeof ImageData !== 'undefined') {
    // Ensure we have a plain ArrayBuffer-backed Uint8ClampedArray (not SharedArrayBuffer)
    const safeCopy = new Uint8ClampedArray(new ArrayBuffer(data.byteLength));
    safeCopy.set(data);
    return new ImageData(safeCopy, width, height);
  }
  // Minimal structural polyfill for Node.js environments without jsdom
  return { data, width, height } as unknown as ImageData;
}

/**
 * Create a 16×16 ImageData:
 *  - Pixels in the central 4×4 area are a warm skin tone (R=220, G=160, B=100)
 *  - Surrounding pixels are neutral grey (R=128, G=128, B=128)
 */
function buildSkinTestImage(width = 16, height = 16): ImageData {
  const data = new Uint8ClampedArray(width * height * 4);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const inCenter = x >= 6 && x < 10 && y >= 6 && y < 10;
      if (inCenter) {
        // Warm skin tone — should pass the HSV heuristic
        data[idx] = 220;
        data[idx + 1] = 160;
        data[idx + 2] = 100;
        data[idx + 3] = 255;
      } else {
        // Neutral grey — should fail the heuristic (S ≈ 0)
        data[idx] = 128;
        data[idx + 1] = 128;
        data[idx + 2] = 128;
        data[idx + 3] = 255;
      }
    }
  }

  return makeImageData(data, width, height);
}

describe('SegmentationNode (HSV heuristic fallback)', () => {
  it('can be instantiated without arguments', () => {
    const node = new SegmentationNode();
    expect(node).toBeInstanceOf(SegmentationNode);
    node.dispose();
  });

  it('detects warm skin-tone pixels as foreground', async () => {
    const node = new SegmentationNode();
    // Do NOT call initialize() — that requires browser APIs.
    // The segment() method falls back to the skin-colour heuristic when
    // _usingMediaPipe is false (default state).

    const imageData = buildSkinTestImage(16, 16);
    const result = await node.segment(imageData);

    expect(result.width).toBe(16);
    expect(result.height).toBe(16);
    expect(result.mask).toHaveLength(16 * 16);

    // All central 4×4 pixels (indices 6..9 in both axes) should be foreground
    let skinCount = 0;
    let greyCount = 0;
    for (let y = 0; y < 16; y++) {
      for (let x = 0; x < 16; x++) {
        const pixelMask = result.mask[y * 16 + x];
        const inCenter = x >= 6 && x < 10 && y >= 6 && y < 10;
        if (inCenter) {
          skinCount += pixelMask!;
        } else {
          greyCount += pixelMask!;
        }
      }
    }

    // Skin-tone centre pixels should mostly be classified as foreground
    expect(skinCount).toBeGreaterThan(0);
    // Grey background pixels should be classified as background (mask = 0)
    expect(greyCount).toBe(0);

    node.dispose();
  });

  it('applyMask zeroes out background pixels', async () => {
    const node = new SegmentationNode();
    const imageData = buildSkinTestImage(4, 4);

    // Create a mask that marks only the first pixel as foreground
    const mask = new Uint8Array(4 * 4);
    mask[0] = 1;

    const masked = node.applyMask(imageData, mask);
    expect(masked.width).toBe(4);
    expect(masked.height).toBe(4);

    // First pixel should be unchanged (foreground)
    expect(masked.data[0]).toBe(imageData.data[0]);

    // Second pixel should be blacked out
    expect(masked.data[4]).toBe(0);
    expect(masked.data[5]).toBe(0);
    expect(masked.data[6]).toBe(0);
    expect(masked.data[7]).toBe(255); // alpha preserved

    node.dispose();
  });
});

// ---------------------------------------------------------------------------
// 4. Pose2DNode — type-level smoke test
// ---------------------------------------------------------------------------

describe('Pose2DNode', () => {
  it('can be instantiated', () => {
    const node = new Pose2DNode();
    expect(node).toBeInstanceOf(Pose2DNode);
    node.dispose();
  });

  it('COCO_KEYPOINTS has exactly 17 entries', () => {
    expect(COCO_KEYPOINTS.length).toBe(17);
  });
});

// ---------------------------------------------------------------------------
// 5. FrameExtractor — instantiation smoke test
// ---------------------------------------------------------------------------

describe('FrameExtractor', () => {
  it('can be instantiated', () => {
    const extractor = new FrameExtractor();
    expect(extractor).toBeInstanceOf(FrameExtractor);
  });

  it('exposes expected methods', () => {
    const extractor = new FrameExtractor();
    expect(typeof extractor.extractFromFile).toBe('function');
    expect(typeof extractor.extractFromUrl).toBe('function');
    expect(typeof extractor.getVideoInfo).toBe('function');
  });
});
