/**
 * Manages the realtime video input source (camera vs. sample file).
 * Wraps the HTMLVideoElement + select dropdown interaction.
 */

export class VideoSource {
  private _stream: MediaStream | null = null;

  constructor(
    private readonly _videoEl: HTMLVideoElement,
    private readonly _selectEl: HTMLSelectElement,
    private readonly _statusEl: HTMLElement,
  ) {}

  /** Wire up the dropdown change handler and attempt camera on first call. */
  async init(): Promise<void> {
    this._selectEl.addEventListener('change', () => {
      if (this._selectEl.value === 'camera') {
        this.switchToCamera();
      } else {
        this.switchToSample();
      }
    });

    try {
      await this._acquireCamera();
      this._selectEl.value = 'camera';
    } catch {
      this._selectEl.value = './assets/josh-demo.mp4';
      await this.switchToSample();
    }
  }

  async switchToCamera(): Promise<void> {
    try {
      this._statusEl.textContent = 'Requesting camera access...';
      this._statusEl.style.display = '';
      if (this._videoEl.src) {
        this._videoEl.removeAttribute('src');
        this._videoEl.load();
      }
      await this._acquireCamera();
      this._statusEl.style.display = 'none';
    } catch {
      this._statusEl.textContent = 'Camera unavailable';
      this._selectEl.value = './assets/josh-demo.mp4';
      await this.switchToSample();
    }
  }

  async switchToSample(): Promise<void> {
    this._releaseCamera();
    this._videoEl.srcObject = null;
    this._videoEl.src = this._selectEl.value;
    this._videoEl.loop = true;
    this._videoEl.muted = true;
    await this._videoEl.play().catch(() => {});
    this._statusEl.style.display = 'none';
  }

  private async _acquireCamera(): Promise<void> {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
      audio: false,
    });
    this._releaseCamera();
    this._stream = stream;
    this._videoEl.srcObject = stream;
    await this._videoEl.play();
  }

  private _releaseCamera(): void {
    this._stream?.getTracks().forEach((t) => t.stop());
    this._stream = null;
  }
}
