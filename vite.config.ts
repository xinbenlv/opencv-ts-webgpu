import { defineConfig } from 'vite';
import { resolve } from 'node:path';
import basicSsl from '@vitejs/plugin-basic-ssl';

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
    host: '0.0.0.0', // Allow access from other machines (e.g. Tailscale)
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

  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
});
