# OpenCV.ts 5.0 — WebGPU

Real-time JOSH 4D human reconstruction pipeline running entirely in the browser using WebGPU and WASM.

## Requirements

### Secure Context (HTTPS)

This application requires a [secure context](https://w3c.github.io/webappsec-secure-contexts/) to function. This is because the JOSH solver uses `SharedArrayBuffer` for zero-copy data sharing between WASM and WebGPU, and browsers restrict `SharedArrayBuffer` to secure contexts with Cross-Origin Isolation headers as a mitigation against [Spectre-class attacks](https://www.w3.org/TR/post-spectre-webdev/#shared-array-buffer).

The required headers are set automatically by the Vite dev server:
- `Cross-Origin-Opener-Policy: same-origin`
- `Cross-Origin-Embedder-Policy: require-corp`

**You must access the app via one of:**
- `https://localhost:5173` (local development)
- `https://localhost:5173` via SSH tunnel if accessing from another machine:
  ```
  ssh -L 5173:localhost:5173 user@<host-ip>
  ```
- Any HTTPS deployment with the above headers

Accessing via plain `http://` or a non-localhost hostname without a valid TLS certificate will cause `SharedArrayBuffer` to be unavailable and the pipeline will fail to initialize.

References:
- [W3C Secure Contexts](https://w3c.github.io/webappsec-secure-contexts/)
- [W3C Post-Spectre Web Development](https://www.w3.org/TR/post-spectre-webdev/#shared-array-buffer)
- [MDN SharedArrayBuffer Security Requirements](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer#security_requirements)

## Development

```bash
npm install
npm run wasm:build   # Requires Rust + wasm-pack
npm run dev
```
