mod contact;
mod lbfgs;

use wasm_bindgen::prelude::*;

/// Initialize the WASM module (called once from JS).
#[wasm_bindgen(start)]
pub fn init() {
    // Set up panic hook for better error messages in browser console
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

// Re-export public API
pub use contact::ContactConstraintEvaluator;
pub use lbfgs::LBFGSOptimizer;
