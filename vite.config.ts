import { defineConfig, type Plugin } from 'vite';
import { resolve } from 'node:path';
import { copyFileSync, mkdirSync, existsSync, readFileSync } from 'node:fs';
import basicSsl from '@vitejs/plugin-basic-ssl';

// Copy ORT WASM binary to demo/assets/ort/ (only the .wasm, not .mjs)
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

/**
 * Vite plugin that serves ORT's .mjs worker file directly from node_modules.
 * ORT dynamically imports this file for the WebGPU backend, but Vite's
 * dep optimizer can't handle it (it's a worker, not a regular module).
 */
function ortWorkerPlugin(): Plugin {
  const ortDist = resolve(__dirname, 'node_modules/onnxruntime-web/dist');
  return {
    name: 'ort-worker-serve',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        // Intercept requests for ORT .mjs files
        if (req.url && req.url.includes('ort-wasm-simd-threaded') && req.url.endsWith('.mjs')) {
          // Extract just the filename
          const filename = req.url.split('/').pop()!.split('?')[0]!;
          const filePath = resolve(ortDist, filename);
          if (existsSync(filePath)) {
            const content = readFileSync(filePath);
            res.setHeader('Content-Type', 'text/javascript');
            res.setHeader('Access-Control-Allow-Origin', '*');
            res.end(content);
            return;
          }
        }
        next();
      });
    },
  };
}

export default defineConfig({
  plugins: [basicSsl(), ortWorkerPlugin()],

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

  // Don't let Vite rewrite ORT's internal dynamic imports
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
});
