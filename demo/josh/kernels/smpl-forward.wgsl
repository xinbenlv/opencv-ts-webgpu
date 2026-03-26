// SMPL Forward Pass — Linear Blend Skinning (LBS) Compute Shader
//
// Computes posed mesh vertices from SMPL parameters:
// 1. Shape blend: V_shaped = meanTemplate + shapeBlendShapes * beta
// 2. Joint regression: J = jointRegressor * V_shaped
// 3. Forward kinematics: global transforms from local rotations
// 4. LBS: V_posed = sum_j(w_j * T_j * V_shaped_j)
//
// This shader handles step 1 (blend shapes) and step 4 (LBS).
// Joint regression and FK are done in a separate pass.

struct Params {
  vertex_count: u32,
  joint_count: u32,
  shape_dim: u32,
  _pad: u32,
}

// Bind group 0: mesh data
@group(0) @binding(0) var<storage, read> mean_template: array<f32>;      // [V, 3]
@group(0) @binding(1) var<storage, read> shape_blend_shapes: array<f32>; // [V, 3, B]
@group(0) @binding(2) var<storage, read> skinning_weights: array<f32>;   // [V, 4]
@group(0) @binding(3) var<storage, read> skinning_indices: array<u32>;   // [V, 4]
@group(0) @binding(4) var<storage, read> joint_transforms: array<f32>;   // [J, 16] (4x4 matrices)
@group(0) @binding(5) var<storage, read> shape_params: array<f32>;       // [B]
@group(0) @binding(6) var<storage, read_write> output_vertices: array<f32>; // [V, 3]
@group(0) @binding(7) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let vid = gid.x;
  if (vid >= params.vertex_count) {
    return;
  }

  let base3 = vid * 3u;

  // Step 1: Shape blend — V_shaped = meanTemplate + sum(beta_i * blendShape_i)
  var vx = mean_template[base3];
  var vy = mean_template[base3 + 1u];
  var vz = mean_template[base3 + 2u];

  for (var b = 0u; b < params.shape_dim; b++) {
    let bs_base = (vid * 3u * params.shape_dim) + b;
    let beta = shape_params[b];
    vx += beta * shape_blend_shapes[bs_base];
    vy += beta * shape_blend_shapes[bs_base + params.shape_dim];
    vz += beta * shape_blend_shapes[bs_base + params.shape_dim * 2u];
  }

  // Step 4: Linear Blend Skinning
  var result = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  let v_homo = vec4<f32>(vx, vy, vz, 1.0);

  for (var k = 0u; k < 4u; k++) {
    let w = skinning_weights[vid * 4u + k];
    if (w < 0.001) {
      continue;
    }

    let joint_idx = skinning_indices[vid * 4u + k];
    let t_base = joint_idx * 16u;

    // 4x4 matrix * vertex (column-major)
    let tx = joint_transforms[t_base]      * v_homo.x +
             joint_transforms[t_base + 4u]  * v_homo.y +
             joint_transforms[t_base + 8u]  * v_homo.z +
             joint_transforms[t_base + 12u] * v_homo.w;

    let ty = joint_transforms[t_base + 1u]  * v_homo.x +
             joint_transforms[t_base + 5u]  * v_homo.y +
             joint_transforms[t_base + 9u]  * v_homo.z +
             joint_transforms[t_base + 13u] * v_homo.w;

    let tz = joint_transforms[t_base + 2u]  * v_homo.x +
             joint_transforms[t_base + 6u]  * v_homo.y +
             joint_transforms[t_base + 10u] * v_homo.z +
             joint_transforms[t_base + 14u] * v_homo.w;

    result += w * vec4<f32>(tx, ty, tz, 0.0);
  }

  output_vertices[base3] = result.x;
  output_vertices[base3 + 1u] = result.y;
  output_vertices[base3 + 2u] = result.z;
}
