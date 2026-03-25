use wasm_bindgen::prelude::*;

/// L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) optimizer.
///
/// Used by the JOSH solver for joint optimization of human motion
/// and scene geometry. Runs in WASM for numerical stability while
/// gradient computation happens on the GPU.
#[wasm_bindgen]
pub struct LBFGSOptimizer {
    /// Number of corrections to approximate inverse Hessian
    m: usize,
    /// Parameter vector length
    n: usize,
    /// Current parameter values (shared via SharedArrayBuffer from JS)
    x: Vec<f64>,
    /// Previous gradient
    prev_grad: Vec<f64>,
    /// History of position differences (s_k = x_{k+1} - x_k)
    s_history: Vec<Vec<f64>>,
    /// History of gradient differences (y_k = g_{k+1} - g_k)
    y_history: Vec<Vec<f64>>,
    /// History of rho_k = 1 / (y_k^T s_k)
    rho_history: Vec<f64>,
    /// Current iteration
    iteration: usize,
    /// Convergence tolerance
    tolerance: f64,
}

#[wasm_bindgen]
impl LBFGSOptimizer {
    /// Create a new L-BFGS optimizer.
    ///
    /// # Arguments
    /// * `n` - Dimension of the parameter space
    /// * `m` - Number of correction pairs to store (typically 5-10)
    /// * `tolerance` - Convergence tolerance for gradient norm
    #[wasm_bindgen(constructor)]
    pub fn new(n: usize, m: usize, tolerance: f64) -> Self {
        Self {
            m,
            n,
            x: vec![0.0; n],
            prev_grad: vec![0.0; n],
            s_history: Vec::with_capacity(m),
            y_history: Vec::with_capacity(m),
            rho_history: Vec::with_capacity(m),
            iteration: 0,
            tolerance,
        }
    }

    /// Set the current parameter values from a JS Float64Array.
    #[wasm_bindgen(js_name = setParameters)]
    pub fn set_parameters(&mut self, params: &[f64]) {
        self.x.copy_from_slice(params);
    }

    /// Get the current parameter values.
    #[wasm_bindgen(js_name = getParameters)]
    pub fn get_parameters(&self) -> Vec<f64> {
        self.x.clone()
    }

    /// Perform one L-BFGS step given the current gradient.
    ///
    /// Returns the search direction (negated L-BFGS direction).
    /// The caller should evaluate the objective along this direction
    /// and call `update` with the new gradient.
    #[wasm_bindgen]
    pub fn step(&mut self, gradient: &[f64]) -> Vec<f64> {
        let mut q = gradient.to_vec();
        let k = self.s_history.len();

        // L-BFGS two-loop recursion
        let mut alpha = vec![0.0; k];

        // First loop: backward through history
        for i in (0..k).rev() {
            alpha[i] = self.rho_history[i] * dot(&self.s_history[i], &q);
            for j in 0..self.n {
                q[j] -= alpha[i] * self.y_history[i][j];
            }
        }

        // Initial Hessian approximation: H_0 = gamma * I
        let gamma = if k > 0 {
            let last = k - 1;
            dot(&self.s_history[last], &self.y_history[last])
                / dot(&self.y_history[last], &self.y_history[last])
        } else {
            1.0
        };

        let mut r: Vec<f64> = q.iter().map(|&qi| gamma * qi).collect();

        // Second loop: forward through history
        for i in 0..k {
            let beta = self.rho_history[i] * dot(&self.y_history[i], &r);
            for j in 0..self.n {
                r[j] += self.s_history[i][j] * (alpha[i] - beta);
            }
        }

        // Negate for descent direction
        for v in &mut r {
            *v = -*v;
        }

        r
    }

    /// Update optimizer state after a line search step.
    ///
    /// # Arguments
    /// * `new_x` - New parameter values after line search
    /// * `new_gradient` - Gradient at the new point
    ///
    /// Returns true if converged.
    #[wasm_bindgen]
    pub fn update(&mut self, new_x: &[f64], new_gradient: &[f64]) -> bool {
        // Compute s_k and y_k
        let s: Vec<f64> = new_x.iter().zip(self.x.iter()).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = new_gradient
            .iter()
            .zip(self.prev_grad.iter())
            .map(|(a, b)| a - b)
            .collect();

        let ys = dot(&y, &s);

        // Skip update if curvature condition is not satisfied
        if ys > 1e-10 {
            if self.s_history.len() >= self.m {
                self.s_history.remove(0);
                self.y_history.remove(0);
                self.rho_history.remove(0);
            }

            self.rho_history.push(1.0 / ys);
            self.s_history.push(s);
            self.y_history.push(y);
        }

        self.x.copy_from_slice(new_x);
        self.prev_grad.copy_from_slice(new_gradient);
        self.iteration += 1;

        // Check convergence
        let grad_norm: f64 = new_gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        grad_norm < self.tolerance
    }

    /// Get the current iteration number.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Reset the optimizer state.
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.x.fill(0.0);
        self.prev_grad.fill(0.0);
        self.s_history.clear();
        self.y_history.clear();
        self.rho_history.clear();
        self.iteration = 0;
    }
}

/// Dot product of two vectors.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}
