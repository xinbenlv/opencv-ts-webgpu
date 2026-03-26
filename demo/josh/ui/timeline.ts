/**
 * Timeline — 2D canvas-based scrubber for JOSH frame playback.
 *
 * Renders onto a provided HTMLCanvasElement:
 *  - A scrubber bar showing all frames coloured by loss (green=low, red=high)
 *  - A draggable playhead indicator
 *  - Current frame / total frames text
 *  - Clickable play/pause/step buttons drawn directly on the canvas
 *  - Loss value display for the current frame
 *
 * Usage:
 *   const tl = new Timeline({ canvas, onFrameChange: (i) => renderFrame(i) });
 *   tl.setFrameCount(30);
 *   tl.setFrameResult(i, lossValue);  // call as frames are processed
 *   tl.render();
 */

export interface TimelineOptions {
  canvas: HTMLCanvasElement;
  onFrameChange?: (frameIndex: number) => void;
}

// ---------------------------------------------------------------------------
// Layout constants (pixels)
// ---------------------------------------------------------------------------

const PAD = 12;
const BAR_HEIGHT = 18;
const BAR_TOP_OFFSET = 46;   // y from top of canvas to top of frame bar
const BTN_SIZE = 28;
const BTN_Y_OFFSET = 8;      // y from top of canvas to button centre
const PLAYHEAD_HALF_W = 5;
const FONT_SMALL = '11px "Segoe UI", system-ui, sans-serif';
const FONT_LABEL = '12px "Segoe UI", system-ui, sans-serif';
const FONT_BOLD  = 'bold 12px "Segoe UI", system-ui, sans-serif';

// ---------------------------------------------------------------------------
// Colour helpers
// ---------------------------------------------------------------------------

function lossToColor(t: number): string {
  // t in [0,1]; 0 = green, 1 = red via HSL
  const hue = Math.round((1 - t) * 120); // 120 = green, 0 = red
  return `hsl(${hue}, 80%, 50%)`;
}

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.replace('#', ''), 16);
  return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff];
}
void hexToRgb; // suppress unused warning (utility for future use)

// ---------------------------------------------------------------------------
// Button descriptor
// ---------------------------------------------------------------------------

interface Button {
  id: 'prev' | 'play' | 'next';
  cx: number;   // centre x (computed during render)
  cy: number;   // centre y (computed during render)
}

// ---------------------------------------------------------------------------
// Timeline class
// ---------------------------------------------------------------------------

export class Timeline {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly onFrameChange?: (frameIndex: number) => void;

  private frameCount = 0;
  private currentFrame = 0;
  private playbackFps = 5;
  private playing = false;
  private playIntervalId: ReturnType<typeof setInterval> | null = null;

  // Per-frame loss data.  NaN means not yet set.
  private losses: Float32Array = new Float32Array(0);
  private minLoss = Infinity;
  private maxLoss = -Infinity;

  // Cached button positions (updated each render call)
  private buttons: Button[] = [
    { id: 'prev', cx: 0, cy: 0 },
    { id: 'play', cx: 0, cy: 0 },
    { id: 'next', cx: 0, cy: 0 },
  ];

  // Drag state for the scrubber
  private isDragging = false;

  // ---------------------------------------------------------------------------
  // Constructor
  // ---------------------------------------------------------------------------

  constructor(opts: TimelineOptions) {
    this.canvas = opts.canvas;
    this.onFrameChange = opts.onFrameChange;

    const ctx = opts.canvas.getContext('2d');
    if (!ctx) throw new Error('Timeline: could not get 2D context from canvas');
    this.ctx = ctx;

    this.canvas.addEventListener('mousedown', this._onMouseDown);
    this.canvas.addEventListener('mousemove', this._onMouseMove);
    window.addEventListener('mouseup', this._onMouseUp);
    this.canvas.addEventListener('touchstart', this._onTouchStart, { passive: true });
    this.canvas.addEventListener('touchmove', this._onTouchMove, { passive: true });
    window.addEventListener('touchend', this._onTouchEnd);
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  setFrameCount(count: number): void {
    this.frameCount = count;
    this.losses = new Float32Array(count).fill(NaN);
    this.minLoss = Infinity;
    this.maxLoss = -Infinity;
    this.currentFrame = 0;
    this.render();
  }

  /** Record the loss for a processed frame; triggers a re-render. */
  setFrameResult(index: number, loss: number): void {
    if (index < 0 || index >= this.frameCount) return;
    this.losses[index] = loss;
    if (loss < this.minLoss) this.minLoss = loss;
    if (loss > this.maxLoss) this.maxLoss = loss;
    this.render();
  }

  getCurrentFrame(): number {
    return this.currentFrame;
  }

  play(): void {
    if (this.playing || this.frameCount === 0) return;
    this.playing = true;
    const ms = 1000 / this.playbackFps;
    this.playIntervalId = setInterval(() => {
      const next = (this.currentFrame + 1) % this.frameCount;
      this._seekInternal(next);
      this.render();
    }, ms);
    this.render();
  }

  pause(): void {
    if (!this.playing) return;
    this.playing = false;
    if (this.playIntervalId !== null) {
      clearInterval(this.playIntervalId);
      this.playIntervalId = null;
    }
    this.render();
  }

  stepForward(): void {
    this.pause();
    if (this.frameCount === 0) return;
    this._seekInternal((this.currentFrame + 1) % this.frameCount);
    this.render();
  }

  stepBackward(): void {
    this.pause();
    if (this.frameCount === 0) return;
    const prev = this.currentFrame === 0 ? this.frameCount - 1 : this.currentFrame - 1;
    this._seekInternal(prev);
    this.render();
  }

  seekTo(frameIndex: number): void {
    this.pause();
    this._seekInternal(Math.max(0, Math.min(frameIndex, this.frameCount - 1)));
    this.render();
  }

  setPlaybackFPS(fps: number): void {
    this.playbackFps = Math.max(1, fps);
    if (this.playing) {
      this.pause();
      this.play();
    }
  }

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  render(): void {
    const { canvas, ctx } = this;
    const W = canvas.width;
    const H = canvas.height;

    // Clear
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0f0f1a';
    ctx.fillRect(0, 0, W, H);

    // Top separator
    ctx.fillStyle = '#1e1e30';
    ctx.fillRect(0, 0, W, 1);

    this._drawButtons(W, H);
    this._drawFrameBar(W, H);
    this._drawInfo(W, H);
  }

  dispose(): void {
    this.pause();
    this.canvas.removeEventListener('mousedown', this._onMouseDown);
    this.canvas.removeEventListener('mousemove', this._onMouseMove);
    window.removeEventListener('mouseup', this._onMouseUp);
    this.canvas.removeEventListener('touchstart', this._onTouchStart);
    this.canvas.removeEventListener('touchmove', this._onTouchMove);
    window.removeEventListener('touchend', this._onTouchEnd);
  }

  // ---------------------------------------------------------------------------
  // Drawing helpers
  // ---------------------------------------------------------------------------

  private _drawButtons(W: number, H: number): void {
    const btnY = BTN_Y_OFFSET + BTN_SIZE / 2;
    const spacing = BTN_SIZE + 8;
    const groupW = spacing * 3 - 8;
    let bx = Math.floor(W / 2 - groupW / 2 + BTN_SIZE / 2);

    for (const btn of this.buttons) {
      btn.cx = bx;
      btn.cy = btnY;
      bx += spacing;
    }

    const { ctx } = this;

    // Draw each button
    for (const btn of this.buttons) {
      const isHover = false; // hover state is not tracked (kept simple)
      ctx.fillStyle = isHover ? '#2a2a44' : '#181828';
      ctx.strokeStyle = '#3a3a5a';
      ctx.lineWidth = 1;
      this._roundRect(btn.cx - BTN_SIZE / 2, btn.cy - BTN_SIZE / 2, BTN_SIZE, BTN_SIZE, 5);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = '#8090b8';

      if (btn.id === 'prev') {
        // Step-back: |◀
        this._drawStepBackIcon(btn.cx, btn.cy);
      } else if (btn.id === 'next') {
        // Step-forward: ▶|
        this._drawStepForwardIcon(btn.cx, btn.cy);
      } else {
        // Play/pause
        if (this.playing) {
          this._drawPauseIcon(btn.cx, btn.cy);
        } else {
          this._drawPlayIcon(btn.cx, btn.cy);
        }
      }
    }
  }

  private _drawPlayIcon(cx: number, cy: number): void {
    const { ctx } = this;
    const s = 8;
    ctx.fillStyle = '#7abfff';
    ctx.beginPath();
    ctx.moveTo(cx - s * 0.4, cy - s * 0.6);
    ctx.lineTo(cx + s * 0.7, cy);
    ctx.lineTo(cx - s * 0.4, cy + s * 0.6);
    ctx.closePath();
    ctx.fill();
  }

  private _drawPauseIcon(cx: number, cy: number): void {
    const { ctx } = this;
    ctx.fillStyle = '#7abfff';
    const w = 4, h = 11, gap = 4;
    ctx.fillRect(cx - gap / 2 - w, cy - h / 2, w, h);
    ctx.fillRect(cx + gap / 2, cy - h / 2, w, h);
  }

  private _drawStepBackIcon(cx: number, cy: number): void {
    const { ctx } = this;
    ctx.fillStyle = '#8090b8';
    // bar
    ctx.fillRect(cx - 7, cy - 5, 2, 10);
    // triangle pointing left
    ctx.beginPath();
    ctx.moveTo(cx + 4, cy - 5);
    ctx.lineTo(cx - 4, cy);
    ctx.lineTo(cx + 4, cy + 5);
    ctx.closePath();
    ctx.fill();
  }

  private _drawStepForwardIcon(cx: number, cy: number): void {
    const { ctx } = this;
    ctx.fillStyle = '#8090b8';
    // bar
    ctx.fillRect(cx + 5, cy - 5, 2, 10);
    // triangle pointing right
    ctx.beginPath();
    ctx.moveTo(cx - 4, cy - 5);
    ctx.lineTo(cx + 4, cy);
    ctx.lineTo(cx - 4, cy + 5);
    ctx.closePath();
    ctx.fill();
  }

  private _drawFrameBar(W: number, _H: number): void {
    const { ctx } = this;
    const barX = PAD;
    const barW = W - PAD * 2;
    const barY = BAR_TOP_OFFSET;

    // Background track
    ctx.fillStyle = '#1a1a2e';
    this._roundRect(barX, barY, barW, BAR_HEIGHT, 4);
    ctx.fill();

    if (this.frameCount === 0) return;

    const frameW = barW / this.frameCount;

    // Draw per-frame colour segments
    const lossRange = this.maxLoss - this.minLoss;

    for (let i = 0; i < this.frameCount; i++) {
      const loss = this.losses[i];
      const segX = barX + i * frameW;
      const segW = Math.max(1, frameW - 0.5);

      if (isNaN(loss)) {
        ctx.fillStyle = '#2a2a40';
      } else {
        const t = lossRange > 0 ? (loss - this.minLoss) / lossRange : 0;
        ctx.fillStyle = lossToColor(t);
      }
      ctx.fillRect(segX, barY, segW, BAR_HEIGHT);
    }

    // Subtle border radius clip via re-drawing track as clip path would be heavy;
    // instead just overdraw rounded corners with background colour.
    ctx.fillStyle = '#0f0f1a';
    this._roundRect(barX, barY, barW, BAR_HEIGHT, 4);
    ctx.stroke(); // transparent — just using for rounded look via strokeStyle below
    ctx.strokeStyle = '#2a2a44';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Playhead
    const playheadX = barX + (this.currentFrame + 0.5) * frameW;
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.moveTo(playheadX - PLAYHEAD_HALF_W, barY - 4);
    ctx.lineTo(playheadX + PLAYHEAD_HALF_W, barY - 4);
    ctx.lineTo(playheadX + PLAYHEAD_HALF_W, barY + BAR_HEIGHT + 4);
    ctx.lineTo(playheadX - PLAYHEAD_HALF_W, barY + BAR_HEIGHT + 4);
    ctx.closePath();
    ctx.fill();

    // Thin bright line in playhead centre
    ctx.fillStyle = '#7abfff';
    ctx.fillRect(playheadX - 1, barY - 4, 2, BAR_HEIGHT + 8);
  }

  private _drawInfo(W: number, H: number): void {
    const { ctx } = this;

    // Frame counter (left)
    ctx.font = FONT_BOLD;
    ctx.fillStyle = '#a0b4cc';
    const frameText = this.frameCount > 0
      ? `Frame ${this.currentFrame + 1} / ${this.frameCount}`
      : 'No frames';
    ctx.fillText(frameText, PAD, H - 10);

    // Loss value (right)
    const loss = this.frameCount > 0 ? this.losses[this.currentFrame] : NaN;
    ctx.font = FONT_LABEL;
    ctx.fillStyle = '#7a8aa0';
    const lossText = isNaN(loss) ? 'Loss: —' : `Loss: ${loss.toFixed(4)}`;
    const lossW = ctx.measureText(lossText).width;
    ctx.fillText(lossText, W - PAD - lossW, H - 10);

    // FPS label (centre)
    ctx.font = FONT_SMALL;
    ctx.fillStyle = '#505868';
    const fpsText = `${this.playbackFps} FPS`;
    const fpsW = ctx.measureText(fpsText).width;
    ctx.fillText(fpsText, W / 2 - fpsW / 2, H - 10);
  }

  // ---------------------------------------------------------------------------
  // Hit-testing helpers
  // ---------------------------------------------------------------------------

  private _barRect(): { x: number; y: number; w: number; h: number } {
    return {
      x: PAD,
      y: BAR_TOP_OFFSET,
      w: this.canvas.width - PAD * 2,
      h: BAR_HEIGHT,
    };
  }

  private _frameIndexFromX(clientX: number): number {
    const rect = this.canvas.getBoundingClientRect();
    const canvasX = (clientX - rect.left) * (this.canvas.width / rect.width);
    const bar = this._barRect();
    const t = (canvasX - bar.x) / bar.w;
    return Math.max(0, Math.min(this.frameCount - 1, Math.floor(t * this.frameCount)));
  }

  private _hitButton(clientX: number, clientY: number): Button | null {
    const rect = this.canvas.getBoundingClientRect();
    const cx = (clientX - rect.left) * (this.canvas.width / rect.width);
    const cy = (clientY - rect.top)  * (this.canvas.height / rect.height);
    const r  = BTN_SIZE / 2;
    for (const btn of this.buttons) {
      if (Math.abs(cx - btn.cx) <= r && Math.abs(cy - btn.cy) <= r) return btn;
    }
    return null;
  }

  private _hitBar(clientX: number, clientY: number): boolean {
    const rect = this.canvas.getBoundingClientRect();
    const cx = (clientX - rect.left) * (this.canvas.width / rect.width);
    const cy = (clientY - rect.top)  * (this.canvas.height / rect.height);
    const bar = this._barRect();
    return cx >= bar.x && cx <= bar.x + bar.w && cy >= bar.y - 6 && cy <= bar.y + bar.h + 6;
  }

  // ---------------------------------------------------------------------------
  // Event handlers
  // ---------------------------------------------------------------------------

  private _onMouseDown = (e: MouseEvent): void => {
    const btn = this._hitButton(e.clientX, e.clientY);
    if (btn) {
      this._handleButtonClick(btn);
      return;
    }
    if (this._hitBar(e.clientX, e.clientY) && this.frameCount > 0) {
      this.isDragging = true;
      const fi = this._frameIndexFromX(e.clientX);
      this.pause();
      this._seekInternal(fi);
      this.render();
    }
  };

  private _onMouseMove = (e: MouseEvent): void => {
    if (!this.isDragging || this.frameCount === 0) return;
    const fi = this._frameIndexFromX(e.clientX);
    this._seekInternal(fi);
    this.render();
  };

  private _onMouseUp = (): void => {
    this.isDragging = false;
  };

  private _onTouchStart = (e: TouchEvent): void => {
    const t = e.touches[0];
    if (!t) return;
    const btn = this._hitButton(t.clientX, t.clientY);
    if (btn) { this._handleButtonClick(btn); return; }
    if (this._hitBar(t.clientX, t.clientY) && this.frameCount > 0) {
      this.isDragging = true;
      this.pause();
      this._seekInternal(this._frameIndexFromX(t.clientX));
      this.render();
    }
  };

  private _onTouchMove = (e: TouchEvent): void => {
    const t = e.touches[0];
    if (!t || !this.isDragging) return;
    this._seekInternal(this._frameIndexFromX(t.clientX));
    this.render();
  };

  private _onTouchEnd = (): void => {
    this.isDragging = false;
  };

  private _handleButtonClick(btn: Button): void {
    if (btn.id === 'play') {
      this.playing ? this.pause() : this.play();
    } else if (btn.id === 'next') {
      this.stepForward();
    } else {
      this.stepBackward();
    }
  }

  // ---------------------------------------------------------------------------
  // Internal seek (no render)
  // ---------------------------------------------------------------------------

  private _seekInternal(frameIndex: number): void {
    const clamped = Math.max(0, Math.min(frameIndex, this.frameCount - 1));
    if (clamped === this.currentFrame) return;
    this.currentFrame = clamped;
    this.onFrameChange?.(clamped);
  }

  // ---------------------------------------------------------------------------
  // Canvas utility
  // ---------------------------------------------------------------------------

  private _roundRect(x: number, y: number, w: number, h: number, r: number): void {
    const { ctx } = this;
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.arcTo(x + w, y, x + w, y + r, r);
    ctx.lineTo(x + w, y + h - r);
    ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
    ctx.lineTo(x + r, y + h);
    ctx.arcTo(x, y + h, x, y + h - r, r);
    ctx.lineTo(x, y + r);
    ctx.arcTo(x, y, x + r, y, r);
    ctx.closePath();
  }
}
