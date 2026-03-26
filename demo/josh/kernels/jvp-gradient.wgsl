// JVP Gradient Kernel: dL/dv → dL/dpose via FK chain rule
//
// Converts per-vertex loss gradient dL/dv [V,3] into per-pose-parameter
// gradient dL/dpose [72] using the analytical SMPL Jacobian.
//
// Formula: dL/dθ_{k,a} = Σ_v (dL/dv_v) · (R_parent_k^g × dR_k^local/dθ_{k,a}) × B_{k,v}
//
// where B_{k,v} = Σ_{j: k is ancestor-or-equal of j} w_{v,j} × R_k^{gT} × (R_j^g × v_shaped_v + t_j^g - t_k^g)
//
// This expresses the shaped vertex v_shaped in the local frame of joint k,
// rotated by all descendant joints, weighted by skinning weights.
//
// Requires joint_transforms [24,16] from the FK forward pass (column-major 4×4).
// One thread per pose parameter (72 threads).

struct Params {
  vertex_count: u32,
  joint_count: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<storage, read> dl_dv: array<f32>;            // [V*3] per-vertex gradient
@group(0) @binding(1) var<storage, read> dR_buf: array<f32>;           // [72*9] Rodrigues derivatives (row-major)
@group(0) @binding(2) var<storage, read> joint_transforms: array<f32>; // [24*16] col-major 4×4
@group(0) @binding(3) var<storage, read> shaped_vertices: array<f32>;  // [V*3]
@group(0) @binding(4) var<storage, read> skin_weights: array<f32>;     // [V*4]
@group(0) @binding(5) var<storage, read> skin_indices: array<u32>;     // [V*4]
@group(0) @binding(6) var<storage, read> parent_indices: array<i32>;   // [24]
@group(0) @binding(7) var<storage, read_write> gradient: array<f32>;   // [89+] full gradient buffer (pose part [0..71])
@group(0) @binding(8) var<uniform> params: Params;

// Check if joint `anc` is an ancestor of (or equal to) `desc` in the kinematic tree.
fn is_ancestor(anc: u32, desc: u32) -> bool {
  var cur: i32 = i32(desc);
  for (var i = 0; i < 24; i++) {
    if (u32(cur) == anc) { return true; }
    let p = parent_indices[u32(cur)];
    if (p < 0) { break; }
    cur = p;
  }
  return false;
}

// Extract global 3×3 rotation from a column-major 4×4 joint transform.
// Returns a WGSL mat3x3 (column-major convention: args are col0, col1, col2).
fn get_global_rot(j: u32) -> mat3x3<f32> {
  let b = j * 16u;
  // joint_transforms is column-major 4×4: col0=[b+0,b+1,b+2,b+3], col1=[b+4..], etc.
  return mat3x3<f32>(
    joint_transforms[b + 0u], joint_transforms[b + 1u], joint_transforms[b + 2u],   // col 0
    joint_transforms[b + 4u], joint_transforms[b + 5u], joint_transforms[b + 6u],   // col 1
    joint_transforms[b + 8u], joint_transforms[b + 9u], joint_transforms[b + 10u],  // col 2
  );
}

// Extract global translation (joint world position) from a 4×4 transform.
fn get_global_pos(j: u32) -> vec3<f32> {
  let b = j * 16u;
  return vec3<f32>(joint_transforms[b + 12u], joint_transforms[b + 13u], joint_transforms[b + 14u]);
}

// Load a row-major 3×3 matrix from dR_buf as a WGSL mat3x3 (column-major).
// Row-major [R00,R01,R02, R10,R11,R12, R20,R21,R22] → col-major mat3x3.
fn get_dR_mat(param: u32) -> mat3x3<f32> {
  let b = param * 9u;
  // Row-major dR[i,j] → dR_buf[i*3+j]
  // WGSL mat3x3 col-major: col0=[dR00,dR10,dR20], col1=[dR01,dR11,dR21], col2=[dR02,dR12,dR22]
  return mat3x3<f32>(
    dR_buf[b + 0u], dR_buf[b + 3u], dR_buf[b + 6u],  // col 0
    dR_buf[b + 1u], dR_buf[b + 4u], dR_buf[b + 7u],  // col 1
    dR_buf[b + 2u], dR_buf[b + 5u], dR_buf[b + 8u],  // col 2
  );
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let param = gid.x;
  if (param >= 72u) { return; }

  let k = param / 3u;  // joint index (0..23)

  // R_parent(k)^global — identity if k is root
  let parent_k = parent_indices[k];
  var R_par: mat3x3<f32>;
  if (parent_k < 0) {
    R_par = mat3x3<f32>(1.,0.,0., 0.,1.,0., 0.,0.,1.);
  } else {
    R_par = get_global_rot(u32(parent_k));
  }

  // A = R_par × dR_k  [3×3]
  // This maps the Rodrigues derivative into the parent's global frame.
  let dR_k = get_dR_mat(param);
  let A = R_par * dR_k;

  // R_k^global^T and t_k^global (for projecting into k's local frame)
  let R_k_gT = transpose(get_global_rot(k));
  let t_k_g  = get_global_pos(k);

  var grad = 0.0;

  for (var v = 0u; v < params.vertex_count; v++) {
    let dl = vec3<f32>(dl_dv[v*3u], dl_dv[v*3u+1u], dl_dv[v*3u+2u]);

    // Skip vertices where the loss gradient is negligible
    if (abs(dl.x) + abs(dl.y) + abs(dl.z) < 1e-10) { continue; }

    let vs = vec3<f32>(shaped_vertices[v*3u], shaped_vertices[v*3u+1u], shaped_vertices[v*3u+2u]);

    // B_{k,v} = Σ_{j: k is ancestor of j} w_{v,j} × R_k^{gT} × (R_j^g × vs + t_j^g - t_k^g)
    var B = vec3<f32>(0.0);
    for (var s = 0u; s < 4u; s++) {
      let j = skin_indices[v * 4u + s];
      let w = skin_weights[v * 4u + s];
      if (w < 1e-4) { continue; }
      if (!is_ancestor(k, j)) { continue; }

      let R_j_g = get_global_rot(j);
      let t_j_g = get_global_pos(j);
      // Position of shaped vertex under joint j's global transform
      let v_world = R_j_g * vs + t_j_g;
      // Express in joint k's frame (removes k's global rotation and translation)
      let v_in_k  = R_k_gT * (v_world - t_k_g);
      B += w * v_in_k;
    }

    // JVP: (dL/dv_v)ᵀ × (A × B)
    grad += dot(dl, A * B);
  }

  // Accumulate into the pose part of the full gradient buffer
  gradient[param] += grad;
}
