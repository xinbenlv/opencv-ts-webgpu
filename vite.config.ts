import { defineConfig, type Plugin } from 'vite';
import { resolve } from 'node:path';
import { copyFileSync, mkdirSync, existsSync, readFileSync } from 'node:fs';
import basicSsl from '@vitejs/plugin-basic-ssl';

// Copy ORT WASM binary to demo/assets/ort/
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
 * Vite plugin that intercepts ORT's .mjs worker and .wasm requests.
 *
 * ORT dynamically imports ort-wasm-simd-threaded.jsep.mjs for its WASM backend.
 * When ORT is excluded from optimizeDeps, Vite serves it via @fs/ and appends
 * ?import to dynamic imports, which can break resolution. This middleware
 * intercepts ALL requests containing the ORT worker filename and serves
 * the file directly from node_modules.
 */
function ortWorkerPlugin(): Plugin {
  const ortDist = resolve(__dirname, 'node_modules/onnxruntime-web/dist');
  return {
    name: 'ort-worker-serve',
    enforce: 'pre',
    configureServer(server) {
      // Use server.middlewares.use with a path prefix that runs BEFORE Vite's transform
      server.middlewares.use((req, res, next) => {
        if (!req.url) return next();
        const url = req.url;

        // Add long-lived cache headers for model files (.onnx, .wasm, .bin)
        if (/\.(onnx|wasm|bin|pkl)(\?|$)/.test(url)) {
          res.setHeader('Cache-Control', 'public, max-age=604800, immutable'); // 7 days
        }

        // Match any request for ORT's worker .mjs (with or without ?import, @fs prefix, etc)
        if (url.includes('ort-wasm-simd-threaded') && url.includes('.mjs')) {
          // Extract the filename (strip query params and path)
          const match = url.match(/(ort-wasm-simd-threaded[^/?]*\.mjs)/);
          if (match) {
            const filePath = resolve(ortDist, match[1]!);
            if (existsSync(filePath)) {
              const content = readFileSync(filePath);
              res.setHeader('Content-Type', 'text/javascript');
              res.setHeader('Access-Control-Allow-Origin', '*');
              res.end(content);
              return;
            }
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

  server: {
    host: '0.0.0.0',
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'credentialless',
      'Cross-Origin-Resource-Policy': 'same-origin',
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

  assetsInclude: ['**/*.wgsl'],

  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
});
