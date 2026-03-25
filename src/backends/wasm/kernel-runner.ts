import { WasmError } from '../../core/errors.ts';
import { loadWasmModule, WasmHandle, type WasmLBFGS, type WasmContact } from './runtime.ts';

/**
 * WASM kernel runner — provides typed wrappers around WASM-compiled functions.
 */
export class WasmKernelRunner {
  private _lbfgs: WasmHandle<WasmLBFGS> | null = null;
  private _contact: WasmHandle<WasmContact> | null = null;

  /**
   * Initialize the L-BFGS optimizer.
   */
  async createLBFGS(
    paramDim: number,
    historySize = 7,
    tolerance = 1e-5,
  ): Promise<WasmHandle<WasmLBFGS>> {
    const mod = await loadWasmModule();
    const lbfgs = new mod.LBFGSOptimizer(paramDim, historySize, tolerance);
    this._lbfgs = new WasmHandle(lbfgs);
    return this._lbfgs;
  }

  /**
   * Initialize the contact constraint evaluator.
   */
  async createContactEvaluator(
    contactThreshold = 0.05,
  ): Promise<WasmHandle<WasmContact>> {
    const mod = await loadWasmModule();
    const contact = new mod.ContactConstraintEvaluator(contactThreshold);
    this._contact = new WasmHandle(contact);
    return this._contact;
  }

  /**
   * Run one L-BFGS optimization step.
   *
   * @param gradient - Current gradient from GPU computation
   * @returns Search direction for line search
   */
  lbfgsStep(gradient: Float64Array): Float64Array {
    if (!this._lbfgs) {
      throw new WasmError('L-BFGS optimizer not initialized. Call createLBFGS first.');
    }
    return this._lbfgs.inner.step(gradient);
  }

  /**
   * Update L-BFGS state after line search.
   */
  lbfgsUpdate(newParams: Float64Array, newGradient: Float64Array): boolean {
    if (!this._lbfgs) {
      throw new WasmError('L-BFGS optimizer not initialized.');
    }
    return this._lbfgs.inner.update(newParams, newGradient);
  }

  dispose(): void {
    this._lbfgs?.dispose();
    this._contact?.dispose();
    this._lbfgs = null;
    this._contact = null;
  }
}
