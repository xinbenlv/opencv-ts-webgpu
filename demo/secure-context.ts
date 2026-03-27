/**
 * Shows a dismissible warning banner when the app is loaded without a
 * secure context (HTTPS / localhost), which prevents SharedArrayBuffer.
 */
export function showSecureContextWarning(): void {
  const modal = document.createElement('div');
  modal.style.cssText = `
    position: fixed; bottom: 1rem; right: 1rem; z-index: 9999;
    max-width: 420px; padding: 1rem 1.25rem;
    background: #1a0000; border: 1px solid #ff4444; border-radius: 8px;
    color: #ff8888; font-family: system-ui, sans-serif; font-size: 0.85rem;
    line-height: 1.5; box-shadow: 0 4px 24px rgba(255,0,0,0.2);
  `;
  modal.innerHTML = `
    <strong style="color:#ff4444;">Secure Context Required</strong><br>
    <code>SharedArrayBuffer</code> is not available. This app requires a
    <a href="https://w3c.github.io/webappsec-secure-contexts/"
       target="_blank" rel="noopener" style="color:#ff6666;">secure context (HTTPS)</a>
    to protect against
    <a href="https://www.w3.org/TR/post-spectre-webdev/#shared-array-buffer"
       target="_blank" rel="noopener" style="color:#ff6666;">Spectre-class attacks</a>.<br><br>
    Access via <code>https://localhost:5173</code> or use an SSH tunnel for remote access.
  `;
  document.body.appendChild(modal);
}
