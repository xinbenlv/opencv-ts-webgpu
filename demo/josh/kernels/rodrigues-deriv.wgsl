// Rodrigues Derivative: analytical dR/dω for all pose parameters
//
// For each of 72 pose parameters (24 joints × 3 axes), computes the 3×3
// derivative of the rotation matrix w.r.t. that axis-angle component.
//
// Formula: R = I + s·K + c·K²
//   ds/dθ = (θcos(θ) - sin(θ)) / θ²
//   dc/dθ = (θsin(θ) - 2(1-cos(θ))) / θ³
//   dθ/dωi = ωi/θ
//
// dR/dωi = (ds/dθ · ωi/θ)·K + s·(dK/dωi) + (dc/dθ · ωi/θ)·K² + c·(dK²/dωi)
//
// Output is row-major [dR00,dR01,dR02, dR10,...,dR22] — 9 floats per parameter.
// Total output: [72, 9] floats.
//
// Used by the JVP kernel to backpropagate loss gradients through FK.

@group(0) @binding(0) var<storage, read> pose: array<f32>;     // [72]
@group(0) @binding(1) var<storage, read_write> dR: array<f32>; // [72, 9] row-major

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let param = gid.x;
  if (param >= 72u) { return; }

  let j = param / 3u;   // joint index (0..23)
  let a = param % 3u;   // axis: 0=x, 1=y, 2=z

  let wx = pose[j * 3u];
  let wy = pose[j * 3u + 1u];
  let wz = pose[j * 3u + 2u];

  let eps: f32 = 1e-12;
  let theta2 = wx*wx + wy*wy + wz*wz;
  let theta  = sqrt(theta2 + eps);

  let s = sin(theta) / (theta + eps);
  let c = (1.0 - cos(theta)) / (theta2 + eps);

  // Derivatives of sinc and cosc w.r.t. θ
  let ds_dt = (theta * cos(theta) - sin(theta)) / (theta2 + eps);
  let dc_dt = (theta * sin(theta) - 2.0 * (1.0 - cos(theta))) / (theta2 * theta + eps);

  // dθ/dωi = ωi/θ
  var dtheta: f32;
  if (a == 0u) { dtheta = wx / theta; }
  else if (a == 1u) { dtheta = wy / theta; }
  else { dtheta = wz / theta; }

  // Composite coefficients: d(sinc)/dωi = ds_dt * dθ/dωi, same for cosc
  let cs = ds_dt * dtheta;  // coefficient for K  term (sinc derivative)
  let cc = dc_dt * dtheta;  // coefficient for K² term (cosc derivative)

  // Fixed K and K² elements
  let K00 = 0.0; let K01 = -wz; let K02 = wy;
  let K10 = wz;  let K11 = 0.0; let K12 = -wx;
  let K20 = -wy; let K21 = wx;  let K22 = 0.0;

  let K2_00 = -(wy*wy + wz*wz); let K2_01 = wx*wy;              let K2_02 = wx*wz;
  let K2_10 = wx*wy;             let K2_11 = -(wx*wx + wz*wz);  let K2_12 = wy*wz;
  let K2_20 = wx*wz;             let K2_21 = wy*wz;             let K2_22 = -(wx*wx + wy*wy);

  // Axis-specific dK/dωi and dK²/dωi (row-major 3×3)
  var dK00: f32; var dK01: f32; var dK02: f32;
  var dK10: f32; var dK11: f32; var dK12: f32;
  var dK20: f32; var dK21: f32; var dK22: f32;

  var dK2_00: f32; var dK2_01: f32; var dK2_02: f32;
  var dK2_10: f32; var dK2_11: f32; var dK2_12: f32;
  var dK2_20: f32; var dK2_21: f32; var dK2_22: f32;

  if (a == 0u) {
    // dK/dwx: K[1,2] = -wx → d/dwx = -1; K[2,1] = wx → d/dwx = 1
    dK00=0.; dK01=0.; dK02=0.;
    dK10=0.; dK11=0.; dK12=-1.;
    dK20=0.; dK21=1.; dK22=0.;
    // dK²/dwx
    dK2_00=0.;      dK2_01=wy;     dK2_02=wz;
    dK2_10=wy;      dK2_11=-2.*wx; dK2_12=0.;
    dK2_20=wz;      dK2_21=0.;     dK2_22=-2.*wx;
  } else if (a == 1u) {
    // dK/dwy: K[0,2] = wy → d/dwy = 1; K[2,0] = -wy → d/dwy = -1
    dK00=0.; dK01=0.;  dK02=1.;
    dK10=0.; dK11=0.;  dK12=0.;
    dK20=-1.; dK21=0.; dK22=0.;
    // dK²/dwy
    dK2_00=-2.*wy; dK2_01=wx; dK2_02=0.;
    dK2_10=wx;     dK2_11=0.; dK2_12=wz;
    dK2_20=0.;     dK2_21=wz; dK2_22=-2.*wy;
  } else {
    // dK/dwz: K[0,1] = -wz → d/dwz = -1; K[1,0] = wz → d/dwz = 1
    dK00=0.;  dK01=-1.; dK02=0.;
    dK10=1.;  dK11=0.;  dK12=0.;
    dK20=0.;  dK21=0.;  dK22=0.;
    // dK²/dwz
    dK2_00=-2.*wz; dK2_01=0.;  dK2_02=wx;
    dK2_10=0.;     dK2_11=-2.*wz; dK2_12=wy;
    dK2_20=wx;     dK2_21=wy;  dK2_22=0.;
  }

  // dR/dωi = cs·K + s·(dK/dωi) + cc·K² + c·(dK²/dωi)  [row-major]
  let out = param * 9u;
  dR[out + 0u] = cs*K00 + s*dK00 + cc*K2_00 + c*dK2_00;  // dR[0,0]
  dR[out + 1u] = cs*K01 + s*dK01 + cc*K2_01 + c*dK2_01;  // dR[0,1]
  dR[out + 2u] = cs*K02 + s*dK02 + cc*K2_02 + c*dK2_02;  // dR[0,2]
  dR[out + 3u] = cs*K10 + s*dK10 + cc*K2_10 + c*dK2_10;  // dR[1,0]
  dR[out + 4u] = cs*K11 + s*dK11 + cc*K2_11 + c*dK2_11;  // dR[1,1]
  dR[out + 5u] = cs*K12 + s*dK12 + cc*K2_12 + c*dK2_12;  // dR[1,2]
  dR[out + 6u] = cs*K20 + s*dK20 + cc*K2_20 + c*dK2_20;  // dR[2,0]
  dR[out + 7u] = cs*K21 + s*dK21 + cc*K2_21 + c*dK2_21;  // dR[2,1]
  dR[out + 8u] = cs*K22 + s*dK22 + cc*K2_22 + c*dK2_22;  // dR[2,2]
}
