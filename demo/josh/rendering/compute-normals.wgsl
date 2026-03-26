// Compute per-vertex normals from a triangle mesh.
// Strategy: single workgroup (1,1,1), sequential loop over all faces.
// Each face contributes its face-normal to each of its three vertices.
// A second pass normalises every vertex normal.
// This is a simple offline approach — sufficient for static or slowly-changing meshes.

@group(0) @binding(0) var<storage, read>       vertices: array<f32>; // [V * 3]
@group(0) @binding(1) var<storage, read>       faces:    array<u32>; // [F * 3]
@group(0) @binding(2) var<storage, read_write> normals:  array<f32>; // [V * 3]

struct Params {
  face_count:   u32,
  vertex_count: u32,
  _p0:          u32,
  _p1:          u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(1, 1, 1)
fn main() {
  // --- Zero normals ---
  for (var v: u32 = 0u; v < params.vertex_count; v++) {
    normals[v * 3u + 0u] = 0.0;
    normals[v * 3u + 1u] = 0.0;
    normals[v * 3u + 2u] = 0.0;
  }

  // --- Accumulate face normals ---
  for (var f: u32 = 0u; f < params.face_count; f++) {
    let i0 = faces[f * 3u + 0u];
    let i1 = faces[f * 3u + 1u];
    let i2 = faces[f * 3u + 2u];

    let v0 = vec3<f32>(
      vertices[i0 * 3u + 0u],
      vertices[i0 * 3u + 1u],
      vertices[i0 * 3u + 2u],
    );
    let v1 = vec3<f32>(
      vertices[i1 * 3u + 0u],
      vertices[i1 * 3u + 1u],
      vertices[i1 * 3u + 2u],
    );
    let v2 = vec3<f32>(
      vertices[i2 * 3u + 0u],
      vertices[i2 * 3u + 1u],
      vertices[i2 * 3u + 2u],
    );

    let face_normal = normalize(cross(v1 - v0, v2 - v0));

    // Add to each vertex (no atomics needed — single thread).
    for (var k: u32 = 0u; k < 3u; k++) {
      let idx = array<u32, 3>(i0, i1, i2)[k];
      normals[idx * 3u + 0u] += face_normal.x;
      normals[idx * 3u + 1u] += face_normal.y;
      normals[idx * 3u + 2u] += face_normal.z;
    }
  }

  // --- Normalise ---
  for (var v: u32 = 0u; v < params.vertex_count; v++) {
    let nx = normals[v * 3u + 0u];
    let ny = normals[v * 3u + 1u];
    let nz = normals[v * 3u + 2u];
    let len = sqrt(nx * nx + ny * ny + nz * nz);
    if (len > 0.0001) {
      normals[v * 3u + 0u] = nx / len;
      normals[v * 3u + 1u] = ny / len;
      normals[v * 3u + 2u] = nz / len;
    } else {
      normals[v * 3u + 0u] = 0.0;
      normals[v * 3u + 1u] = 1.0;
      normals[v * 3u + 2u] = 0.0;
    }
  }
}
