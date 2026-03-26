import { defineConfig } from 'vite';
import { resolve } from 'node:path';

/**
 * Vite config for building the demo as a deployable web app (GitHub Pages).
 * Differs from the main config: no lib mode, no SSL plugin, includes WASM assets.
 */
export default defineConfig({
  root: 'demo',

  resolve: {
    alias: {
      '@core': resolve(__dirname, 'src/core'),
      '@graph': resolve(__dirname, 'src/graph'),
      '@backends': resolve(__dirname, 'src/backends'),
      '@kernels': resolve(__dirname, 'src/kernels'),
    },
  },

  build: {
    target: 'es2022',
    outDir: resolve(__dirname, 'dist'),
    emptyOutDir: true,
  },

  // WGSL files imported as raw strings
  assetsInclude: ['**/*.wgsl'],
});
