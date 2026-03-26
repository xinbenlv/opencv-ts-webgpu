// Counter Increment — GPU-native Adam timestep tracker
//
// Increments a u32 counter stored in a storage buffer by 1.
// Used to track the Adam optimizer timestep (t) across iterations
// within a single command encoder (no CPU involvement in the loop).
//
// The counter starts at 0; after the first Adam step it becomes 1, etc.
// Adam bias correction uses t = counter[0].

@group(0) @binding(0) var<storage, read_write> counter: array<u32>; // [1]

@compute @workgroup_size(1, 1, 1)
fn main() {
  counter[0] = counter[0] + 1u;
}
