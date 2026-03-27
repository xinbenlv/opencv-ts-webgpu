/**
 * Phase 1F — Contact Detection
 *
 * Detects contact vertices (e.g. foot soles) by comparing SMPL vertex Z
 * positions against a scene depth map.
 *
 * For each candidate vertex:
 *   1. Project to image: (u, v) = (fx*X/Z + cx, fy*Y/Z + cy)
 *   2. Bilinearly sample the depth map at (u, v)
 *   3. If |vertex_Z - sampled_depth| < threshold → contact
 */

export interface ContactDetectionParams {
  /** Contact threshold in meters, e.g. 0.05 (5 cm) */
  contactThresholdMeters: number;
  /** Camera intrinsics */
  cameraFx: number;
  cameraFy: number;
  cameraCx: number;
  cameraCy: number;
}

export interface ContactResult {
  /** Per-candidate contact flag (same order as candidateVertexIndices) */
  isContact: boolean[];
  /** SMPL vertex indices that are in contact */
  contactVertexIndices: number[];
}

/**
 * Detect which candidate vertices are in contact with the scene surface.
 *
 * @param vertices               Flat [V*3] array of posed SMPL vertex XYZ
 * @param depthMap               Flat [H*W] depth map in metres (row-major)
 * @param depthWidth             Width of the depth map in pixels
 * @param depthHeight            Height of the depth map in pixels
 * @param candidateVertexIndices Vertex indices to test (e.g. foot sole verts)
 * @param params                 Camera intrinsics + threshold
 */
export function detectContacts(
  vertices: Float32Array,
  depthMap: Float32Array,
  depthWidth: number,
  depthHeight: number,
  candidateVertexIndices: number[],
  params: ContactDetectionParams,
): ContactResult {
  const { contactThresholdMeters, cameraFx, cameraFy, cameraCx, cameraCy } = params;

  const isContact: boolean[] = [];
  const contactVertexIndices: number[] = [];

  for (let ci = 0; ci < candidateVertexIndices.length; ci++) {
    const vi = candidateVertexIndices[ci]!;
    const X = vertices[vi * 3]!;
    const Y = vertices[vi * 3 + 1]!;
    const Z = vertices[vi * 3 + 2]!;

    // Vertex must be in front of the camera
    if (Z <= 0) {
      isContact.push(false);
      continue;
    }

    // Project to image coordinates
    const u = cameraFx * (X / Z) + cameraCx;
    const v = cameraFy * (Y / Z) + cameraCy;

    // Bilinearly sample the depth map
    const sampledDepth = sampleDepthBilinear(depthMap, depthWidth, depthHeight, u, v);

    if (sampledDepth === null) {
      // Projection is outside the image bounds
      isContact.push(false);
      continue;
    }

    const inContact = Math.abs(Z - sampledDepth) < contactThresholdMeters;
    isContact.push(inContact);
    if (inContact) {
      contactVertexIndices.push(vi);
    }
  }

  return { isContact, contactVertexIndices };
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/**
 * Bilinear interpolation of a depth map at fractional pixel coordinates.
 *
 * Returns null when (u, v) is outside the image.
 */
function sampleDepthBilinear(
  depthMap: Float32Array,
  W: number,
  H: number,
  u: number,
  v: number,
): number | null {
  // Clamp to valid sample region (need at least 1 full texel)
  if (u < 0 || v < 0 || u > W - 1 || v > H - 1) {
    return null;
  }

  const x0 = Math.floor(u);
  const y0 = Math.floor(v);
  const x1 = Math.min(x0 + 1, W - 1);
  const y1 = Math.min(y0 + 1, H - 1);

  const tx = u - x0;
  const ty = v - y0;

  const d00 = depthMap[y0 * W + x0]!;
  const d10 = depthMap[y0 * W + x1]!;
  const d01 = depthMap[y1 * W + x0]!;
  const d11 = depthMap[y1 * W + x1]!;

  // Bilinear interpolation
  return (1 - tx) * (1 - ty) * d00
       + tx       * (1 - ty) * d10
       + (1 - tx) *      ty  * d01
       + tx       *      ty  * d11;
}
