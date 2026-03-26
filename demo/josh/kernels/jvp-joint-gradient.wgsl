// JVP Joint Gradient: dL/djoint → dL/dpose via FK chain rule
//
// Joint positions j_k depend on pose params the same way as vertices,
// but the skinning is trivial (weight = 1 for the influencing joint only).
//
// For SMPL joint k's position: j_k = t_k^global (the translation column
// of the global transform T_k^global). Its derivative w.r.t. pose param
// (anc, a) where anc is an ancestor of k:
//
//   dj_k/d(theta_{anc,a}) = (R_parent(anc)^g × dR_anc/d(theta_{anc,a}) × R_{anc}^{gT} × t_k^g)
//
// This is the same chain-rule formula as for vertex positions, but since
// joint k has "skinning weight 1 to itself", B_{anc,k} simplifies to
// R_{anc}^{gT} × (t_k^g - t_{anc}^g).
//
// dL/d(theta_{anc,a}) += Σ_{k: anc is ancestor of k} dl_djoint_k · A_{anc,a} × B_{anc,k}
//
// Output writes additively to gradient[0..71].

struct Params {
  joint_count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> dl_djoint: array<f32>;        // [24*3]
@group(0) @binding(1) var<storage, read> dR_buf: array<f32>;           // [72*9] row-major
@group(0) @binding(2) var<storage, read> joint_transforms: array<f32>; // [24*16] col-major 4×4
@group(0) @binding(3) var<storage, read> parent_indices: array<i32>;   // [24]
@group(0) @binding(4) var<storage, read_write> gradient: array<f32>;   // [89+]
@group(0) @binding(5) var<uniform> p: Params;

fn is_ancestor(anc: u32, desc: u32) -> bool {
  var cur: i32 = i32(desc);
  for (var i = 0; i < 24; i++) {
    if (u32(cur) == anc) { return true; }
    let par = parent_indices[u32(cur)];
    if (par < 0) { break; }
    cur = par;
  }
  return false;
}

fn get_global_rot(j: u32) -> mat3x3<f32> {
  let b = j * 16u;
  return mat3x3<f32>(
    joint_transforms[b + 0u], joint_transforms[b + 1u], joint_transforms[b + 2u],
    joint_transforms[b + 4u], joint_transforms[b + 5u], joint_transforms[b + 6u],
    joint_transforms[b + 8u], joint_transforms[b + 9u], joint_transforms[b + 10u],
  );
}

fn get_global_pos(j: u32) -> vec3<f32> {
  let b = j * 16u;
  return vec3<f32>(joint_transforms[b + 12u], joint_transforms[b + 13u], joint_transforms[b + 14u]);
}

fn get_dR_mat(param: u32) -> mat3x3<f32> {
  let b = param * 9u;
  return mat3x3<f32>(
    dR_buf[b + 0u], dR_buf[b + 3u], dR_buf[b + 6u],
    dR_buf[b + 1u], dR_buf[b + 4u], dR_buf[b + 7u],
    dR_buf[b + 2u], dR_buf[b + 5u], dR_buf[b + 8u],
  );
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let param = gid.x;
  if (param >= 72u) { return; }

  let anc = param / 3u;  // ancestor joint index

  // R_parent(anc)^global
  let parent_anc = parent_indices[anc];
  var R_par: mat3x3<f32>;
  if (parent_anc < 0) {
    R_par = mat3x3<f32>(1.,0.,0., 0.,1.,0., 0.,0.,1.);
  } else {
    R_par = get_global_rot(u32(parent_anc));
  }

  let dR_anc = get_dR_mat(param);
  let A = R_par * dR_anc;

  let R_anc_gT = transpose(get_global_rot(anc));
  let t_anc    = get_global_pos(anc);

  var grad = 0.0;

  for (var k = 0u; k < 24u; k++) {
    if (!is_ancestor(anc, k)) { continue; }

    let dl = vec3<f32>(dl_djoint[k*3u], dl_djoint[k*3u+1u], dl_djoint[k*3u+2u]);
    if (abs(dl.x) + abs(dl.y) + abs(dl.z) < 1e-10) { continue; }

    // B = R_{anc}^{gT} × (t_k^g - t_{anc}^g)
    let t_k = get_global_pos(k);
    let B = R_anc_gT * (t_k - t_anc);

    grad += dot(dl, A * B);
  }

  gradient[param] += grad;
}
