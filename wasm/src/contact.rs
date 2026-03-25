use wasm_bindgen::prelude::*;

/// Contact constraint evaluator for the JOSH solver.
///
/// Evaluates whether human mesh vertices (feet, hands) are in contact
/// with scene surfaces, and computes the corresponding loss and gradient
/// for the joint optimization.
#[wasm_bindgen]
pub struct ContactConstraintEvaluator {
    /// Gravity direction (normalized)
    gravity: [f64; 3],
    /// Contact distance threshold in meters
    contact_threshold: f64,
    /// Indices of SMPL vertices that are potential contact points
    /// (feet: 3216, 3217, 6617, 6618; hands: 1961, 5361)
    contact_vertex_indices: Vec<usize>,
}

#[wasm_bindgen]
impl ContactConstraintEvaluator {
    #[wasm_bindgen(constructor)]
    pub fn new(contact_threshold: f64) -> Self {
        Self {
            gravity: [0.0, -1.0, 0.0],
            contact_threshold,
            // Default SMPL contact vertices (feet soles)
            contact_vertex_indices: vec![
                3216, 3217, 3218, 3219, 3220, // left foot sole
                6617, 6618, 6619, 6620, 6621, // right foot sole
            ],
        }
    }

    /// Set custom gravity direction.
    #[wasm_bindgen(js_name = setGravity)]
    pub fn set_gravity(&mut self, gx: f64, gy: f64, gz: f64) {
        let norm = (gx * gx + gy * gy + gz * gz).sqrt();
        if norm > 1e-8 {
            self.gravity = [gx / norm, gy / norm, gz / norm];
        }
    }

    /// Set contact vertex indices from a JS Uint32Array.
    #[wasm_bindgen(js_name = setContactVertices)]
    pub fn set_contact_vertices(&mut self, indices: &[u32]) {
        self.contact_vertex_indices = indices.iter().map(|&i| i as usize).collect();
    }

    /// Evaluate contact loss and gradient.
    ///
    /// # Arguments
    /// * `vertices` - SMPL mesh vertices, flat f64 array of shape [6890, 3]
    /// * `depth_map` - Scene depth map, flat f64 array of shape [H * W]
    /// * `width` - Depth map width
    /// * `height` - Depth map height
    /// * `fx` - Camera focal length x
    /// * `fy` - Camera focal length y
    /// * `cx` - Camera principal point x
    /// * `cy` - Camera principal point y
    ///
    /// Returns a flat array: [loss, grad_v0_x, grad_v0_y, grad_v0_z, ...]
    #[wasm_bindgen(js_name = evaluateContact)]
    pub fn evaluate_contact(
        &self,
        vertices: &[f64],
        depth_map: &[f64],
        width: u32,
        height: u32,
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
    ) -> Vec<f64> {
        let num_contacts = self.contact_vertex_indices.len();
        // Output: 1 (loss) + num_contacts * 3 (gradients)
        let mut result = vec![0.0; 1 + num_contacts * 3];

        let mut total_loss = 0.0;

        for (ci, &vi) in self.contact_vertex_indices.iter().enumerate() {
            if vi * 3 + 2 >= vertices.len() {
                continue;
            }

            let vx = vertices[vi * 3];
            let vy = vertices[vi * 3 + 1];
            let vz = vertices[vi * 3 + 2];

            // Skip vertices behind camera
            if vz <= 0.0 {
                continue;
            }

            // Project vertex to image plane
            let u = (fx * vx / vz + cx).round() as i32;
            let v = (fy * vy / vz + cy).round() as i32;

            if u < 0 || u >= width as i32 || v < 0 || v >= height as i32 {
                continue;
            }

            let depth_idx = (v as u32 * width + u as u32) as usize;
            if depth_idx >= depth_map.len() {
                continue;
            }

            let scene_depth = depth_map[depth_idx];
            let vertex_depth = vz;

            // Contact: vertex is close to scene surface
            let penetration = vertex_depth - scene_depth;

            if penetration.abs() < self.contact_threshold {
                // Soft contact loss: penalize deviation from surface
                let loss = penetration * penetration;
                total_loss += loss;

                // Gradient w.r.t. vertex position (chain rule through projection)
                let grad_z = 2.0 * penetration;
                result[1 + ci * 3 + 2] = grad_z;

                // Check surface normal alignment with gravity
                let normal_alignment = self.gravity[1]; // simplified
                if normal_alignment < -0.7 {
                    // Surface is roughly horizontal (floor) — stronger constraint
                    result[1 + ci * 3 + 1] = 2.0 * penetration * 0.5;
                }
            }
        }

        result[0] = total_loss;
        result
    }
}
