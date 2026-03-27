/**
 * Loading step tracker for the JOSH pipeline initialisation UI.
 *
 * `LoadingStatus` is framework-free and holds no DOM references until
 * `attachToDOM()` is called, making it independently testable.
 */

export type StepStatus = 'pending' | 'active' | 'done' | 'warn' | 'error';

const ICONS: Record<StepStatus, string> = {
  pending: '│', active: '▶', done: '✔', warn: '⚠', error: '✘',
};
const COLORS: Record<StepStatus, string> = {
  pending: '#555', active: '#60a5fa', done: '#4ade80', warn: '#fbbf24', error: '#f87171',
};

interface StepEntry { status: StepStatus; text: string }

export class LoadingStatus {
  private readonly _steps = new Map<string, StepEntry>();
  private _loadingEl: HTMLElement | null = null;
  private _modalEl: HTMLElement | null = null;

  setStep(id: string, status: StepStatus, text: string): void {
    this._steps.set(id, { status, text });
    this._flush();
  }

  /**
   * Returns the current steps as an HTML string.
   * Pure: depends only on internal state, no DOM access.
   */
  renderSteps(): string {
    return [...this._steps.entries()]
      .map(([, { status, text }]) =>
        `<div style="color:${COLORS[status]}">${ICONS[status]} ${text}</div>`,
      )
      .join('');
  }

  attachToDOM(
    loadingEl: HTMLElement,
    modalEl: HTMLElement,
    showBtn: HTMLElement,
    closeBtn: HTMLElement,
  ): void {
    this._loadingEl = loadingEl;
    this._modalEl   = modalEl;

    const modalStepsEl = modalEl.querySelector<HTMLElement>('#statusModalSteps');

    showBtn.addEventListener('click', () => {
      modalEl.style.display = 'flex';
      if (modalStepsEl) modalStepsEl.innerHTML = this.renderSteps();
    });
    closeBtn.addEventListener('click', () => { modalEl.style.display = 'none'; });
    modalEl.addEventListener('click', (e) => {
      if (e.target === modalEl) modalEl.style.display = 'none';
    });

    this._flush();
  }

  private _flush(): void {
    const html = this.renderSteps();
    if (this._loadingEl) this._loadingEl.innerHTML = html;
    const modalStepsEl = this._modalEl?.querySelector<HTMLElement>('#statusModalSteps');
    if (modalStepsEl) modalStepsEl.innerHTML = html;
  }
}
