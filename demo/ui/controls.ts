/**
 * Simple UI controls for the JOSH demo.
 */
export interface DemoControls {
  /** Toggle depth map visualization */
  showDepth: boolean;
  /** Toggle SMPL mesh wireframe */
  showMesh: boolean;
  /** Toggle solver output */
  showSolverOutput: boolean;
  /** Max L-BFGS iterations per frame */
  maxIterations: number;
  /** Contact constraint weight */
  contactWeight: number;
}

export function createDefaultControls(): DemoControls {
  return {
    showDepth: true,
    showMesh: true,
    showSolverOutput: true,
    maxIterations: 5,
    contactWeight: 1.0,
  };
}

/**
 * Bind keyboard shortcuts for the demo.
 */
export function bindKeyboard(controls: DemoControls): void {
  document.addEventListener('keydown', (e) => {
    switch (e.key) {
      case '1':
        controls.showDepth = !controls.showDepth;
        break;
      case '2':
        controls.showMesh = !controls.showMesh;
        break;
      case '3':
        controls.showSolverOutput = !controls.showSolverOutput;
        break;
      case '+':
      case '=':
        controls.maxIterations = Math.min(controls.maxIterations + 1, 20);
        break;
      case '-':
        controls.maxIterations = Math.max(controls.maxIterations - 1, 1);
        break;
    }
  });
}
