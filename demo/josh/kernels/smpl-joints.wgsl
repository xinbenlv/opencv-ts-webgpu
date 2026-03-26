// SMPL Joint Regression + Forward Kinematics
//
// 1. Joint regression: J = jointRegressor * V_shaped (sparse)
// 2. Forward kinematics: compose parent transforms down kinematic tree
//
// This runs as a single workgroup (24 joints) since FK is sequential.

struct Params {
  vertex_count: u32,
  joint_count: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<storage, read> shaped_vertices: array<f32>;    // [V, 3]
@group(0) @binding(1) var<storage, read> joint_regressor: array<f32>;    // [J, V] sparse weights
@group(0) @binding(2) var<storage, read> local_rotations: array<f32>;    // [J, 9] rotation matrices
@group(0) @binding(3) var<storage, read> parent_indices: array<i32>;     // [J]
@group(0) @binding(4) var<storage, read_write> joint_transforms: array<f32>; // [J, 16]
@group(0) @binding(5) var<storage, read_write> joint_positions: array<f32>;  // [J, 3]
@group(0) @binding(6) var<uniform> params: Params;

// Multiply two 4x4 matrices (column-major)
fn mat4_mul(a_base: u32, b: array<f32, 16>) -> array<f32, 16> {
  var result: array<f32, 16>;
  for (var col = 0u; col < 4u; col++) {
    for (var row = 0u; row < 4u; row++) {
      var sum = 0.0;
      for (var k = 0u; k < 4u; k++) {
        sum += joint_transforms[a_base + k * 4u + row] * b[col * 4u + k];
      }
      result[col * 4u + row] = sum;
    }
  }
  return result;
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // Step 1: Joint regression — J_j = sum_v(regressor[j,v] * V_shaped[v])
  for (var j = 0u; j < params.joint_count; j++) {
    var jx = 0.0;
    var jy = 0.0;
    var jz = 0.0;

    for (var v = 0u; v < params.vertex_count; v++) {
      let w = joint_regressor[j * params.vertex_count + v];
      if (w != 0.0) {
        jx += w * shaped_vertices[v * 3u];
        jy += w * shaped_vertices[v * 3u + 1u];
        jz += w * shaped_vertices[v * 3u + 2u];
      }
    }

    joint_positions[j * 3u] = jx;
    joint_positions[j * 3u + 1u] = jy;
    joint_positions[j * 3u + 2u] = jz;
  }

  // Step 2: Forward kinematics — compose transforms from root to leaves
  for (var j = 0u; j < params.joint_count; j++) {
    let t_base = j * 16u;
    let r_base = j * 9u;

    // Build local 4x4 transform: [R | t; 0 0 0 1]
    // Rotation from local_rotations (3x3 row-major → 4x4 column-major)
    var local_mat: array<f32, 16>;
    local_mat[0]  = local_rotations[r_base];     // col0.x
    local_mat[1]  = local_rotations[r_base + 3u]; // col0.y
    local_mat[2]  = local_rotations[r_base + 6u]; // col0.z
    local_mat[3]  = 0.0;
    local_mat[4]  = local_rotations[r_base + 1u]; // col1.x
    local_mat[5]  = local_rotations[r_base + 4u]; // col1.y
    local_mat[6]  = local_rotations[r_base + 7u]; // col1.z
    local_mat[7]  = 0.0;
    local_mat[8]  = local_rotations[r_base + 2u]; // col2.x
    local_mat[9]  = local_rotations[r_base + 5u]; // col2.y
    local_mat[10] = local_rotations[r_base + 8u]; // col2.z
    local_mat[11] = 0.0;
    // Translation = joint position (relative to parent for non-root)
    local_mat[12] = joint_positions[j * 3u];
    local_mat[13] = joint_positions[j * 3u + 1u];
    local_mat[14] = joint_positions[j * 3u + 2u];
    local_mat[15] = 1.0;

    let parent = parent_indices[j];

    if (parent < 0) {
      // Root joint — local transform IS global transform
      for (var i = 0u; i < 16u; i++) {
        joint_transforms[t_base + i] = local_mat[i];
      }
    } else {
      // Compose with parent: T_global = T_parent * T_local
      let p_base = u32(parent) * 16u;

      // Adjust translation: relative to parent joint
      local_mat[12] -= joint_positions[u32(parent) * 3u];
      local_mat[13] -= joint_positions[u32(parent) * 3u + 1u];
      local_mat[14] -= joint_positions[u32(parent) * 3u + 2u];

      let result = mat4_mul(p_base, local_mat);
      for (var i = 0u; i < 16u; i++) {
        joint_transforms[t_base + i] = result[i];
      }
    }

    // Update joint positions from global transform translation
    joint_positions[j * 3u] = joint_transforms[t_base + 12u];
    joint_positions[j * 3u + 1u] = joint_transforms[t_base + 13u];
    joint_positions[j * 3u + 2u] = joint_transforms[t_base + 14u];
  }
}
