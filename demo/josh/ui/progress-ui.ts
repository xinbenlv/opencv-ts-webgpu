/**
 * ProgressUI — Phase 2F
 *
 * Renders a live progress dashboard for the JOSH batch pipeline:
 *  1. Phase table  (9 phases, per-phase progress bar + status badge)
 *  2. Loss curve   (canvas 2D, incremental, auto-scaled)
 *  3. ETA display  (rolling average of phase completion times)
 *  4. Stats bar    (frame / chunk / iteration / loss)
 *
 * Usage:
 *   const ui = new ProgressUI({ container: document.getElementById('root') });
 *   ui.show();
 *   ui.updatePhase('extract', 12, 50, 'active');
 *   ui.addLossPoint({ frameIndex: 12, iterIndex: 450, loss: 0.0342 });
 */

export interface PhaseStatus {
  name: string;
  label: string;
  current: number;
  total: number;
  status: 'pending' | 'active' | 'done' | 'error';
  etaMs?: number;
}

export interface LossPoint {
  frameIndex: number;
  iterIndex: number;
  loss: number;
}

export interface ProgressUIOptions {
  container: HTMLElement;
  /** Phase names to display (default: all 9 in pipeline order) */
  phases?: string[];
}

// ---------------------------------------------------------------------------
// Phase metadata
// ---------------------------------------------------------------------------

interface PhaseMeta {
  name: string;
  label: string;
  icon: string;
  color: string;
}

const ALL_PHASES: PhaseMeta[] = [
  { name: 'extract',     label: 'Frame Extraction',   icon: '🎞️',  color: '#4CAF50' },
  { name: 'segment',     label: 'Person Segmentation', icon: '✂️',  color: '#66BB6A' },
  { name: 'mast3r',      label: 'MASt3R Depth',        icon: '🔭',  color: '#42A5F5' },
  { name: 'focal',       label: 'Focal Estimation',    icon: '📐',  color: '#26C6DA' },
  { name: 'romp',        label: 'ROMP Detection',      icon: '🧍',  color: '#AB47BC' },
  { name: 'pose2d',      label: '2D Pose (MoveNet)',   icon: '🦴',  color: '#EC407A' },
  { name: 'contact',     label: 'Contact Detection',   icon: '👣',  color: '#FFA726' },
  { name: 'optimize',    label: 'JOSH Optimisation',   icon: '⚙️',  color: '#EF5350' },
  { name: 'interpolate', label: 'SE3 Interpolation',   icon: '🔗',  color: '#26A69A' },
];

// ---------------------------------------------------------------------------
// Inline CSS injected once into <head>
// ---------------------------------------------------------------------------

const CSS_ID = 'progress-ui-styles';

function injectStyles(): void {
  if (document.getElementById(CSS_ID)) return;
  const style = document.createElement('style');
  style.id = CSS_ID;
  style.textContent = `
/* ===== ProgressUI root ===== */
.pui-root {
  display: none;
  flex-direction: column;
  gap: 12px;
  background: #1a1a2e;
  border: 1px solid #2a2a4a;
  border-radius: 8px;
  padding: 16px;
  font-family: 'Segoe UI', system-ui, sans-serif;
  font-size: 13px;
  color: #e0e0f0;
  min-width: 420px;
}
.pui-root.pui-visible { display: flex; }

/* ===== Stats bar ===== */
.pui-stats {
  display: flex;
  gap: 16px;
  background: #12122a;
  border: 1px solid #2a2a4a;
  border-radius: 4px;
  padding: 7px 12px;
  font-family: monospace;
  font-size: 12px;
  flex-wrap: wrap;
  color: #a0a8d0;
}
.pui-stats span { color: #dde; }

/* ===== ETA ===== */
.pui-eta {
  font-size: 12px;
  color: #8888aa;
  text-align: right;
  font-family: monospace;
}
.pui-eta b { color: #aaddff; }

/* ===== Phase table ===== */
.pui-phase-table {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.pui-phase-row {
  display: grid;
  grid-template-columns: 24px 1fr auto auto;
  align-items: center;
  gap: 8px;
}

.pui-phase-icon { font-size: 14px; text-align: center; line-height: 1; }

.pui-phase-label { font-size: 12px; color: #ccd; white-space: nowrap; }

.pui-phase-bar-wrap {
  position: relative;
  height: 8px;
  background: #22224a;
  border-radius: 4px;
  overflow: hidden;
  min-width: 80px;
  flex: 1 1 80px;
}

/* Override grid: make bar-wrap + counter sit inside a sub-flex */
.pui-phase-row {
  grid-template-columns: 24px 130px 1fr auto;
}

.pui-phase-bar {
  position: absolute;
  inset: 0 auto 0 0;
  border-radius: 4px;
  transition: width 0.25s ease;
  background: linear-gradient(90deg, #4CAF50, #2196F3);
}

.pui-phase-bar.pui-active {
  animation: pui-pulse 1.2s ease-in-out infinite;
}

@keyframes pui-pulse {
  0%   { opacity: 1; }
  50%  { opacity: 0.65; }
  100% { opacity: 1; }
}

.pui-phase-counter {
  font-family: monospace;
  font-size: 11px;
  color: #7788aa;
  white-space: nowrap;
  min-width: 52px;
  text-align: right;
}

.pui-badge {
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 10px;
  font-weight: 600;
  white-space: nowrap;
}
.pui-badge-pending  { background: #222244; color: #6666aa; }
.pui-badge-active   { background: #1a3a1a; color: #66cc66; border: 1px solid #44aa44; }
.pui-badge-done     { background: #123312; color: #44ee44; }
.pui-badge-error    { background: #3a1212; color: #ee4444; }

/* ===== Loss canvas ===== */
.pui-loss-wrap {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.pui-loss-title { font-size: 11px; color: #7788aa; }
.pui-loss-canvas {
  display: block;
  background: #0d0d1e;
  border: 1px solid #2a2a4a;
  border-radius: 4px;
  width: 100%;
  height: 200px;
}
  `;
  document.head.appendChild(style);
}

// ---------------------------------------------------------------------------
// ProgressUI class
// ---------------------------------------------------------------------------

export class ProgressUI {
  private readonly _root: HTMLDivElement;
  private readonly _phases: PhaseMeta[];

  // Per-phase DOM references
  private readonly _phaseState = new Map<
    string,
    {
      bar: HTMLDivElement;
      counter: HTMLSpanElement;
      badge: HTMLSpanElement;
      current: number;
      total: number;
      status: PhaseStatus['status'];
      completedAt?: number;
    }
  >();

  // ETA tracking: rolling list of (phaseName -> completionTime ms)
  private readonly _phaseStartTimes = new Map<string, number>();
  private readonly _phaseCompletionTimes: number[] = [];

  // Loss data
  private readonly _lossPoints: LossPoint[] = [];
  private readonly _canvas: HTMLCanvasElement;
  private readonly _ctx: CanvasRenderingContext2D;
  private _currentFrameIndex = 0;

  // Stats
  private _statsFrame = 0;
  private _statsTotalFrames = 0;
  private _statsChunk = 1;
  private _statsTotalChunks = 1;
  private _statsIter = 0;
  private _statsTotalIter = 700;
  private _statsLoss = 0;

  // Stats bar DOM
  private readonly _statsFrameEl: HTMLSpanElement;
  private readonly _statsChunkEl: HTMLSpanElement;
  private readonly _statsIterEl: HTMLSpanElement;
  private readonly _statsLossEl: HTMLSpanElement;
  private readonly _etaEl: HTMLSpanElement;

  constructor(opts: ProgressUIOptions) {
    injectStyles();

    const phaseNames = opts.phases ?? ALL_PHASES.map((p) => p.name);
    this._phases = ALL_PHASES.filter((p) => phaseNames.includes(p.name));

    // ---- Build DOM ----
    this._root = document.createElement('div');
    this._root.className = 'pui-root';

    // Stats bar
    const statsBar = document.createElement('div');
    statsBar.className = 'pui-stats';
    this._statsFrameEl = document.createElement('span');
    this._statsChunkEl = document.createElement('span');
    this._statsIterEl  = document.createElement('span');
    this._statsLossEl  = document.createElement('span');
    statsBar.append(
      document.createTextNode('Frame '),
      this._statsFrameEl,
      document.createTextNode(' | Chunk '),
      this._statsChunkEl,
      document.createTextNode(' | Iteration '),
      this._statsIterEl,
      document.createTextNode(' | Loss: '),
      this._statsLossEl,
    );

    // ETA
    const etaRow = document.createElement('div');
    etaRow.className = 'pui-eta';
    etaRow.append(document.createTextNode('ETA: '));
    this._etaEl = document.createElement('b');
    this._etaEl.textContent = '—';
    etaRow.appendChild(this._etaEl);

    // Phase table
    const table = document.createElement('div');
    table.className = 'pui-phase-table';

    for (const ph of this._phases) {
      const row = document.createElement('div');
      row.className = 'pui-phase-row';

      const icon = document.createElement('div');
      icon.className = 'pui-phase-icon';
      icon.textContent = ph.icon;

      const label = document.createElement('div');
      label.className = 'pui-phase-label';
      label.textContent = ph.label;

      const barWrap = document.createElement('div');
      barWrap.className = 'pui-phase-bar-wrap';

      const bar = document.createElement('div');
      bar.className = 'pui-phase-bar';
      bar.style.width = '0%';
      bar.style.background = `linear-gradient(90deg, ${ph.color}, #2196F3)`;
      barWrap.appendChild(bar);

      const counter = document.createElement('span');
      counter.className = 'pui-phase-counter';
      counter.textContent = '0/0';

      const badge = document.createElement('span');
      badge.className = 'pui-badge pui-badge-pending';
      badge.textContent = 'pending';

      row.appendChild(icon);
      row.appendChild(label);
      row.appendChild(barWrap);
      row.appendChild(counter);
      row.appendChild(badge);
      table.appendChild(row);

      this._phaseState.set(ph.name, { bar, counter, badge, current: 0, total: 0, status: 'pending' });
    }

    // Loss curve
    const lossWrap = document.createElement('div');
    lossWrap.className = 'pui-loss-wrap';
    const lossTitle = document.createElement('div');
    lossTitle.className = 'pui-loss-title';
    lossTitle.textContent = 'Optimisation Loss Curve';
    this._canvas = document.createElement('canvas');
    this._canvas.className = 'pui-loss-canvas';
    this._canvas.width = 600;
    this._canvas.height = 200;
    this._ctx = this._canvas.getContext('2d')!;
    lossWrap.appendChild(lossTitle);
    lossWrap.appendChild(this._canvas);

    // Assemble
    this._root.appendChild(statsBar);
    this._root.appendChild(etaRow);
    this._root.appendChild(table);
    this._root.appendChild(lossWrap);

    opts.container.appendChild(this._root);

    this._updateStats();
    this._drawLoss();
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  updatePhase(
    name: string,
    current: number,
    total: number,
    status: PhaseStatus['status'] = 'active',
  ): void {
    const state = this._phaseState.get(name);
    if (!state) return;

    const prev = state.status;

    // Track start time for ETA
    if (prev !== 'active' && status === 'active') {
      this._phaseStartTimes.set(name, performance.now());
    }

    // Track completion time
    if (prev === 'active' && (status === 'done' || status === 'error')) {
      const start = this._phaseStartTimes.get(name);
      if (start !== undefined) {
        this._phaseCompletionTimes.push(performance.now() - start);
        this._updateEta();
      }
    }

    state.current = current;
    state.total   = total;
    state.status  = status;

    const pct = total > 0 ? Math.min(100, (current / total) * 100) : 0;
    state.bar.style.width = pct + '%';

    state.bar.classList.toggle('pui-active', status === 'active');

    state.counter.textContent = total > 0 ? `${current}/${total}` : '—';

    // Badge
    state.badge.className = `pui-badge pui-badge-${status}`;
    const labels: Record<PhaseStatus['status'], string> = {
      pending: 'pending',
      active:  'running',
      done:    '✓ done',
      error:   '✗ error',
    };
    state.badge.textContent = labels[status];

    // Sync stats frame from extract phase
    if (name === 'extract' || name === 'optimize') {
      this._statsFrame = current;
      this._statsTotalFrames = total;
      this._updateStats();
    }
  }

  addLossPoint(point: LossPoint): void {
    this._lossPoints.push(point);
    this._currentFrameIndex = point.frameIndex;
    this._statsIter   = point.iterIndex;
    this._statsLoss   = point.loss;
    this._updateStats();
    this._drawLoss();
  }

  reset(): void {
    for (const [_name, state] of this._phaseState) {
      state.current = 0;
      state.total   = 0;
      state.status  = 'pending';
      state.bar.style.width = '0%';
      state.bar.classList.remove('pui-active');
      state.counter.textContent = '0/0';
      state.badge.className = 'pui-badge pui-badge-pending';
      state.badge.textContent = 'pending';
    }
    this._lossPoints.length = 0;
    this._phaseStartTimes.clear();
    this._phaseCompletionTimes.length = 0;
    this._statsFrame = 0;
    this._statsTotalFrames = 0;
    this._statsChunk = 1;
    this._statsTotalChunks = 1;
    this._statsIter = 0;
    this._statsLoss = 0;
    this._etaEl.textContent = '—';
    this._updateStats();
    this._drawLoss();
  }

  show(): void {
    this._root.classList.add('pui-visible');
  }

  hide(): void {
    this._root.classList.remove('pui-visible');
  }

  getElement(): HTMLElement {
    return this._root;
  }

  dispose(): void {
    this._root.remove();
  }

  // -------------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------------

  private _updateStats(): void {
    this._statsFrameEl.textContent =
      this._statsTotalFrames > 0
        ? `${this._statsFrame}/${this._statsTotalFrames}`
        : `${this._statsFrame}`;
    this._statsChunkEl.textContent = `${this._statsChunk}/${this._statsTotalChunks}`;
    this._statsIterEl.textContent  =
      this._statsTotalIter > 0
        ? `${this._statsIter}/${this._statsTotalIter}`
        : `${this._statsIter}`;
    this._statsLossEl.textContent  = this._statsLoss > 0 ? this._statsLoss.toFixed(4) : '—';
  }

  private _updateEta(): void {
    const times = this._phaseCompletionTimes;
    if (times.length === 0) { this._etaEl.textContent = '—'; return; }

    // Rolling average of last 3 phase durations
    const window = times.slice(-3);
    const avgMs  = window.reduce((a, b) => a + b, 0) / window.length;

    // Remaining phases = those still pending or active
    const remaining = [...this._phaseState.values()].filter(
      (s) => s.status === 'pending' || s.status === 'active',
    ).length;

    const totalMs = avgMs * remaining;
    if (totalMs <= 0) { this._etaEl.textContent = 'almost done'; return; }

    const s = Math.round(totalMs / 1000);
    const m = Math.floor(s / 60);
    const sec = s % 60;
    this._etaEl.textContent =
      m > 0 ? `~${m}m ${sec}s remaining` : `~${sec}s remaining`;
  }

  private _drawLoss(): void {
    const canvas = this._canvas;
    const ctx    = this._ctx;
    const W = canvas.width;
    const H = canvas.height;
    const PAD = { top: 12, right: 12, bottom: 28, left: 44 };
    const plotW = W - PAD.left - PAD.right;
    const plotH = H - PAD.top  - PAD.bottom;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0d0d1e';
    ctx.fillRect(0, 0, W, H);

    // Axes
    ctx.strokeStyle = '#2a2a5a';
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.moveTo(PAD.left, PAD.top);
    ctx.lineTo(PAD.left, PAD.top + plotH);
    ctx.lineTo(PAD.left + plotW, PAD.top + plotH);
    ctx.stroke();

    const pts = this._lossPoints;
    if (pts.length < 2) {
      // No data message
      ctx.fillStyle = '#444466';
      ctx.font = '12px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('no data yet', PAD.left + plotW / 2, PAD.top + plotH / 2);
      ctx.textAlign = 'left';
      return;
    }

    // Auto-scale
    const minLoss = Math.min(...pts.map((p) => p.loss));
    const maxLoss = Math.max(...pts.map((p) => p.loss));
    const minFrame = pts[0]!.frameIndex;
    const maxFrame = pts[pts.length - 1]!.frameIndex;

    const lossRange  = maxLoss - minLoss || 1;
    const frameRange = maxFrame - minFrame || 1;

    const toX = (f: number) => PAD.left + ((f - minFrame) / frameRange) * plotW;
    const toY = (l: number) => PAD.top + plotH - ((l - minLoss) / lossRange) * plotH;

    // Grid lines (4 horizontal)
    ctx.strokeStyle = '#1e1e3a';
    ctx.lineWidth   = 1;
    for (let i = 0; i <= 4; i++) {
      const y = PAD.top + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(PAD.left, y);
      ctx.lineTo(PAD.left + plotW, y);
      ctx.stroke();
      const lv = maxLoss - (lossRange / 4) * i;
      ctx.fillStyle = '#555577';
      ctx.font = '10px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(lv.toFixed(3), PAD.left - 4, y + 3);
    }
    ctx.textAlign = 'left';

    // Current frame vertical line
    if (pts.length > 0) {
      const cfx = toX(this._currentFrameIndex);
      ctx.strokeStyle = 'rgba(255, 200, 60, 0.35)';
      ctx.lineWidth   = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(cfx, PAD.top);
      ctx.lineTo(cfx, PAD.top + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Loss line
    ctx.beginPath();
    ctx.moveTo(toX(pts[0]!.frameIndex), toY(pts[0]!.loss));
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(toX(pts[i]!.frameIndex), toY(pts[i]!.loss));
    }
    const grad = ctx.createLinearGradient(PAD.left, 0, PAD.left + plotW, 0);
    grad.addColorStop(0, '#4CAF50');
    grad.addColorStop(1, '#2196F3');
    ctx.strokeStyle = grad;
    ctx.lineWidth   = 1.5;
    ctx.stroke();

    // Dots at each point (if few enough)
    if (pts.length <= 80) {
      ctx.fillStyle = '#4CAF50';
      for (const pt of pts) {
        ctx.beginPath();
        ctx.arc(toX(pt.frameIndex), toY(pt.loss), 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // X axis labels
    ctx.fillStyle = '#555577';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    const labelCount = Math.min(5, pts.length);
    for (let i = 0; i < labelCount; i++) {
      const idx = Math.round((i / (labelCount - 1)) * (pts.length - 1));
      const pt  = pts[idx]!;
      const x   = toX(pt.frameIndex);
      ctx.fillText(`f${pt.frameIndex}`, x, PAD.top + plotH + 14);
    }
    ctx.textAlign = 'left';
  }
}
