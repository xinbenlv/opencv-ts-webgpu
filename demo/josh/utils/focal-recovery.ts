/**
 * Phase 1D — Focal Length Recovery
 *
 * Port of recover_focal_shift() from the JOSH paper.
 *
 * Given a set of 3D points (from MASt3R) and their corresponding 2D
 * projections, solves for focal length f and an additive depth shift δ
 * using closed-form least-squares.
 *
 * Projection model: u = f * X/Z + cx, v = f * Y/Z + cy
 *
 * Focal least-squares:
 *   Minimize Σ_i [ (f*(X_i/Z_i) - (u_i - cx))² + (f*(Y_i/Z_i) - (v_i - cy))² ]
 *
 *   Differentiating wrt f and setting to zero:
 *     f = Σ_i (A_ix * b_ix + A_iy * b_iy) / Σ_i (A_ix² + A_iy²)
 *   where A_ix = X_i/Z_i, A_iy = Y_i/Z_i, b_ix = u_i - cx, b_iy = v_i - cy
 */

export interface FocalRecoveryResult {
  /** Recovered focal length in pixels */
  focalLength: number;
  /** Additive depth shift δ such that corrected depth = Z + δ */
  depthShift: number;
  /** RMS reprojection error in pixels */
  reprojectionError: number;
}

/**
 * Recover focal length from 3D–2D correspondences via closed-form
 * least-squares.
 *
 * @param points3D  Flat [N*3] array of XYZ world points (row-major)
 * @param points2D  Flat [N*2] array of UV pixel coordinates (row-major)
 * @param imageWidth   Image width in pixels (used to derive cx = W/2)
 * @param imageHeight  Image height in pixels (used to derive cy = H/2)
 */
export function recoverFocalLength(
  points3D: Float32Array,
  points2D: Float32Array,
  imageWidth: number,
  imageHeight: number,
): FocalRecoveryResult {
  const N = points3D.length / 3;
  if (N < 1) {
    throw new RangeError('recoverFocalLength: need at least 1 point');
  }

  const cx = imageWidth / 2;
  const cy = imageHeight / 2;

  // ── Pass 1: solve for f with Z as-is ────────────────────────────────────────
  let num = 0;
  let den = 0;
  for (let i = 0; i < N; i++) {
    const X = points3D[i * 3];
    const Y = points3D[i * 3 + 1];
    const Z = points3D[i * 3 + 2];
    if (Z <= 0) continue; // behind camera – skip

    const aX = X / Z;
    const aY = Y / Z;
    const bX = points2D[i * 2] - cx;
    const bY = points2D[i * 2 + 1] - cy;

    num += aX * bX + aY * bY;
    den += aX * aX + aY * aY;
  }

  const focalLength = den > 0 ? num / den : 1;

  // ── Pass 2: recover depth shift δ ─────────────────────────────────────────
  //   With Z' = Z + δ the projection becomes:
  //     u = f*X/(Z+δ) + cx ≈ f*X/Z * (1 - δ/Z) + cx   (first-order)
  //   For a full nonlinear solve we use a single Newton step over
  //   residual r(δ) = Σ [(f*X/(Z+δ) - (u-cx))² + (f*Y/(Z+δ) - (v-cy))²].
  //   Two Newton steps are sufficient for typical depth ranges.
  const depthShift = recoverDepthShift(points3D, points2D, focalLength, cx, cy);

  // ── Pass 3: compute RMS reprojection error ─────────────────────────────────
  let sse = 0;
  let count = 0;
  for (let i = 0; i < N; i++) {
    const X = points3D[i * 3];
    const Y = points3D[i * 3 + 1];
    const Z = points3D[i * 3 + 2] + depthShift;
    if (Z <= 0) continue;

    const uProj = focalLength * (X / Z) + cx;
    const vProj = focalLength * (Y / Z) + cy;
    const du = uProj - points2D[i * 2];
    const dv = vProj - points2D[i * 2 + 1];
    sse += du * du + dv * dv;
    count++;
  }
  const reprojectionError = count > 0 ? Math.sqrt(sse / count) : 0;

  return { focalLength, depthShift, reprojectionError };
}

/**
 * Recover additive depth shift δ such that Z_i + δ minimizes reprojection
 * error, given a known focal length.
 *
 * Uses two Newton–Raphson iterations on the sum-of-squared-residuals
 * objective.  Starting point δ = 0.
 *
 * @param points3D   Flat [N*3] XYZ array
 * @param points2D   Flat [N*2] UV array
 * @param focalLength  Known focal length in pixels
 * @param cx  Principal point x
 * @param cy  Principal point y
 * @returns   Optimal depth shift δ
 */
export function recoverFocalShift(
  points3D: Float32Array,
  points2D: Float32Array,
  focalLength: number,
  cx: number,
  cy: number,
): number {
  return recoverDepthShift(points3D, points2D, focalLength, cx, cy);
}

// ── Internal helper ──────────────────────────────────────────────────────────

/** Two Newton steps on the reprojection-error objective over δ. */
function recoverDepthShift(
  points3D: Float32Array,
  points2D: Float32Array,
  f: number,
  cx: number,
  cy: number,
): number {
  let delta = 0;
  const ITERS = 3;

  for (let iter = 0; iter < ITERS; iter++) {
    let grad = 0; // dL/dδ
    let hess = 0; // d²L/dδ²

    const N = points3D.length / 3;
    for (let i = 0; i < N; i++) {
      const X = points3D[i * 3];
      const Y = points3D[i * 3 + 1];
      const Zraw = points3D[i * 3 + 2];
      const Z = Zraw + delta;
      if (Z <= 0) continue;

      const Z2 = Z * Z;
      const Z3 = Z2 * Z;

      // residuals
      const rx = f * (X / Z) + cx - points2D[i * 2];
      const ry = f * (Y / Z) + cy - points2D[i * 2 + 1];

      // drx/dδ = -f*X/Z²
      const drxDd = -f * X / Z2;
      const dryDd = -f * Y / Z2;

      // d²rx/dδ² = 2f*X/Z³
      const d2rxDd2 = 2 * f * X / Z3;
      const d2ryDd2 = 2 * f * Y / Z3;

      // dL/dδ = 2*(rx*drx/dδ + ry*dry/dδ)
      grad += rx * drxDd + ry * dryDd;

      // d²L/dδ² ≈ 2*(drx/dδ)² + 2*(dry/dδ)²  [Gauss-Newton approx, positive-definite]
      hess += drxDd * drxDd + dryDd * dryDd;
    }

    if (Math.abs(hess) < 1e-12) break;
    delta -= grad / hess;
  }

  return delta;
}
