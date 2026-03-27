# Agent Guidelines for opencv-ts-webgpu

> Patterns and conventions for using Claude Code agents effectively in this project.

## Project Context

OpenCV.ts 5.0 — TypeScript-first, WebGPU-native Computer Vision framework.
JOSH algorithm reimplementation running entirely in-browser (no Python).

## Workflow Conventions

### After Code Changes
1. Run `npm run typecheck` — fix all errors before proceeding
2. Run `npm run build` — ensure clean build
3. Bump `version` in `package.json` (patch for fixes, minor for features)
4. Commit with conventional commit message including 6-char hash reference
5. Start `npm run dev` so user can test immediately
6. Show all Vite URLs including network IPs (Tailscale access)

### Deployment
- Always push unless explicitly told not to
- "committed" or similar past tense = "please commit for me"
- Never skip version bumps

## Parallel Agent Orchestration

When spawning multiple sub-agents (e.g., parallel crawl/research tasks):

- **Sub-agents must NOT run any git commands** — no `git add`, `git commit`, `git push`
- Sub-agents write results to temp files or unique output paths
- The **orchestrator** (parent agent) handles all git operations **sequentially**
- Use `TaskUpdate` to track agent progress
- If an agent fails, log it and continue — don't block others

This prevents `.git/index.lock` deadlocks that have caused partial failures in past sessions.

## File Size Discipline

- **600-line limit:** If a file exceeds 600 lines, either split it into smaller focused modules, or — if splitting is not yet warranted — add a comment at the earliest sensible location in the file explaining why it is large (e.g. co-located rendering logic, single-entry bootstrap, generated output).
- Apply this check whenever creating or significantly extending a file.

## Code Change Discipline

- Focus on the exact symptom/feature described — no unrequested changes
- If you find pre-existing issues, mention them but don't fix unless asked
- All changes must pass `tsc --noEmit` before committing
- Fix any type errors you introduce — never dismiss them
- For image/CV processing: prefer conservative thresholds, test with real data

## Key Build Commands

```bash
npm run dev          # Start Vite dev server (port 5182, HTTPS)
npm run build        # TypeScript check + Vite build
npm run typecheck    # tsc --noEmit only
npm run test         # Vitest run
npm run wasm:build   # Rust → WASM (requires wasm-pack)
npm run lint         # Biome check
```

## Architecture Quick Reference

- **Core framework:** `src/` (tensor types, graph engine, backends, kernels)
- **Demo/JOSH pipeline:** `demo/` (batch pipeline, WGSL shaders, nodes, rendering)
- **WASM backend:** `wasm/` (Rust source, compiled to `src/backends/wasm/pkg/`)
- **Tests:** `tests/` (72/100 passing, 28 missing)
- **Critical blocker:** `stubJoshOptimize` in `batch-pipeline.ts` needs tf.js wiring

## Dev Environment

- Mac Mini development machine, accessed via Tailscale from another machine
- SSH tunnel needed for HTTPS (WebGPU + SharedArrayBuffer require secure context)
- SMPL model requires separate academic license download (not in repo)
