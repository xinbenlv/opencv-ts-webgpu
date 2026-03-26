import { defineConfig } from 'vite';
import { resolve } from 'node:path';
import { copyFileSync, mkdirSync, existsSync } from 'node:fs';
import basicSsl from '@vitejs/plugin-basic-ssl';

// Copy ORT WASM files to a place Vite can serve them without transformation.
// These go into demo/assets/ort/ (not demo/public/) so Vite doesn't try to
// transform them as source modules.
function copyOrtWasmFiles() {
  const ortDist = resolve(__dirname, 'node_modules/onnxruntime-web/dist');
  const dest = resolve(__dirname, 'demo/assets/ort');
  if (!existsSync(dest)) mkdirSync(dest, { recursive: true });
  for (const f of [
    'ort-wasm-simd-threaded.jsep.wasm',
    'ort-wasm-simd-threaded.wasm',
  ]) {
    const src = resolve(ortDist, f);
    if (existsSync(src)) copyFileSync(src, resolve(dest, f));
  }
}

copyOrtWasmFiles();

export default defineConfig({
  plugins: [basicSsl()],

  root: 'demo',

  resolve: {
    alias: {
      '@core': resolve(__dirname, 'src/core'),
      '@graph': resolve(__dirname, 'src/graph'),
      '@backends': resolve(__dirname, 'src/backends'),
      '@kernels': resolve(__dirname, 'src/kernels'),
    },
  },

  // HTTPS + Cross-Origin Isolation headers for SharedArrayBuffer + WebGPU
  server: {
    host: '0.0.0.0',
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },

  build: {
    target: 'es2022',
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      formats: ['es'],
      fileName: 'opencv-ts',
    },
    rollupOptions: {
      external: ['onnxruntime-web'],
    },
  },

  // WGSL files imported as raw strings
  assetsInclude: ['**/*.wgsl'],
});
