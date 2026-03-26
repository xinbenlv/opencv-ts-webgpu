"""
ROMP / BEV → ONNX export script
================================
Exports a ROMP or BEV (Bird's-Eye View) model to ONNX so that the JOSH
browser pipeline can run single-image SMPL parameter estimation.

Expected ONNX interface
-----------------------
  Input  : "image"    [1, 3, 512, 512]  normalised RGB (float32)
  Outputs: "pose"     [1, 72]           axis-angle pose  (24 joints × 3)
           "betas"    [1, 10]           SMPL shape coefficients
           "cam"      [1, 3]            weak-perspective camera (scale, tx, ty)

Output file: public/models/romp-bev-fp32.onnx  (~150 MB)

Usage
-----
    python scripts/export-romp.py [--out public/models/romp-bev-fp32.onnx]

Requirements (installed automatically if missing):
    torch torchvision onnx romp (or cloned from github.com/Arthur151/ROMP)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export ROMP/BEV to ONNX")
    p.add_argument(
        "--out",
        default="public/models/romp-bev-fp32.onnx",
        help="Destination .onnx path (default: public/models/romp-bev-fp32.onnx)",
    )
    p.add_argument(
        "--romp-dir",
        default="./romp",
        help="Path to ROMP source repo (cloned if absent)",
    )
    p.add_argument(
        "--checkpoint",
        default="",
        help="Explicit path to BEV.pkl or ROMP checkpoint (auto-downloaded if blank)",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    p.add_argument(
        "--model",
        choices=["bev", "romp"],
        default="bev",
        help="Which model variant to export (default: bev)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------

def pip_install(*packages: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", *packages]
    )


def ensure_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        print("[export-romp] Installing torch …")
        pip_install("torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu")


def ensure_romp_package() -> bool:
    """Try to install the 'romp' pip package. Returns True on success."""
    try:
        import romp  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        print("[export-romp] Installing romp package …")
        pip_install("romp")
        import romp  # noqa: F401
        return True
    except Exception as exc:
        print(f"[export-romp] pip install romp failed: {exc}")
        return False


def clone_romp_repo(romp_dir: str) -> bool:
    """Clone ROMP from GitHub if the directory does not exist."""
    if Path(romp_dir).exists():
        print(f"[export-romp] ROMP repo already present at {romp_dir}")
        return True
    try:
        print("[export-romp] Cloning ROMP repo …")
        subprocess.check_call(
            ["git", "clone", "--depth", "1",
             "https://github.com/Arthur151/ROMP.git", romp_dir]
        )
        return True
    except Exception as exc:
        print(f"[export-romp] git clone failed: {exc}")
        return False


def download_checkpoint(model: str, romp_dir: str) -> str:
    """
    Download the BEV or ROMP checkpoint and return its local path.
    URLs sourced from the ROMP / BEV release pages on GitHub.
    """
    import urllib.request

    checkpoints: dict[str, tuple[str, str]] = {
        "bev": (
            "https://github.com/Arthur151/ROMP/releases/download/V2.0/BEV.pkl",
            os.path.join(romp_dir, "trained_models", "BEV.pkl"),
        ),
        "romp": (
            "https://github.com/Arthur151/ROMP/releases/download/V1.1/ROMP.pkl",
            os.path.join(romp_dir, "trained_models", "ROMP.pkl"),
        ),
    }

    url, dest = checkpoints[model]
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        print(f"[export-romp] Checkpoint already present at {dest}")
        return str(dest_path)

    print(f"[export-romp] Downloading checkpoint from {url} …")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"[export-romp] Saved to {dest}")
        return str(dest_path)
    except Exception as exc:
        raise RuntimeError(f"Checkpoint download failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Export wrappers
# ---------------------------------------------------------------------------

def _build_bev_wrapper(checkpoint_path: str):
    """
    Load BEV model and return a torch.nn.Module wrapper that maps
    image [1,3,512,512] → (pose [1,72], betas [1,10], cam [1,3]).
    """
    import torch
    import torch.nn as nn

    # Add the ROMP source to sys.path so we can import internal modules.
    # BEV lives under romp/simple_romp/
    romp_root = Path(checkpoint_path).parent.parent
    for candidate in [
        romp_root / "simple_romp",
        romp_root / "romp",
        romp_root,
    ]:
        p = str(candidate)
        if p not in sys.path and (candidate / "__init__.py").exists() or candidate.is_dir():
            sys.path.insert(0, p)

    # Try the BEV import paths used across different ROMP releases
    bev_model = None
    tried: list[str] = []

    # Attempt 1: simple_romp.bev
    try:
        from simple_romp.bev import BEV  # type: ignore
        bev_model = BEV(checkpoint_path=checkpoint_path)
        print("[export-romp] Loaded BEV via simple_romp.bev")
    except Exception as exc:
        tried.append(f"simple_romp.bev: {exc}")

    # Attempt 2: romp.bev
    if bev_model is None:
        try:
            from romp.bev import BEV  # type: ignore
            bev_model = BEV(checkpoint_path=checkpoint_path)
            print("[export-romp] Loaded BEV via romp.bev")
        except Exception as exc:
            tried.append(f"romp.bev: {exc}")

    if bev_model is None:
        raise ImportError(
            "Could not import BEV model. Tried:\n  " + "\n  ".join(tried)
        )

    class BEVExportWrapper(nn.Module):
        """Thin wrapper: image → (pose, betas, cam)."""

        MAP_SIZE = 64  # BEV / ROMP output map is 64×64

        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, image: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            # image: [1, 3, 512, 512] normalised float32
            outputs = self.model(image)

            # BEV returns a dict with keys that vary by version; handle both.
            if isinstance(outputs, dict):
                center_maps = outputs.get("center_map", outputs.get("heatmap"))
                params_maps = outputs.get("params_map", outputs.get("param_map", outputs.get("params")))
            elif isinstance(outputs, (list, tuple)):
                center_maps, params_maps = outputs[0], outputs[1]
            else:
                raise ValueError(f"Unexpected model output type: {type(outputs)}")

            # center_maps: [1, 1, 64, 64] or [1, 64, 64]
            # params_maps: [1, C, 64, 64]  C = 3(cam)+72(pose)+10(betas) = 85
            if center_maps.dim() == 4:
                center_flat = center_maps[:, 0].reshape(1, -1)   # [1, 4096]
            else:
                center_flat = center_maps.reshape(1, -1)

            best_idx = center_flat.argmax(dim=1, keepdim=True)  # [1,1]

            ms = self.MAP_SIZE
            # params_maps: [1, 85, 64, 64] → [1, 85, 4096]
            pmap = params_maps.reshape(1, -1, ms * ms)           # [1, 85, 4096]

            idx_exp = best_idx.unsqueeze(1).expand(1, pmap.shape[1], 1)  # [1,85,1]
            params = pmap.gather(2, idx_exp).squeeze(2)          # [1, 85]

            cam   = params[:, 0:3]                               # [1, 3]
            pose  = params[:, 3:75]                              # [1, 72]
            betas = params[:, 75:85]                             # [1, 10]

            return pose, betas, cam

    wrapper = BEVExportWrapper(bev_model).eval()
    return wrapper


def _build_romp_v1_wrapper(checkpoint_path: str):
    """
    Fallback: load ROMP v1 model.
    Same output contract as BEVExportWrapper.
    """
    import torch
    import torch.nn as nn

    romp_root = Path(checkpoint_path).parent.parent
    for candidate in [romp_root / "romp", romp_root]:
        p = str(candidate)
        if p not in sys.path:
            sys.path.insert(0, p)

    from romp.model import ROMP  # type: ignore

    model = ROMP()
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()

    class ROMPv1ExportWrapper(nn.Module):
        MAP_SIZE = 64

        def __init__(self, m: nn.Module) -> None:
            super().__init__()
            self.model = m

        def forward(self, image: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            outputs = self.model(image)
            if isinstance(outputs, dict):
                center_maps = outputs["center_map"]
                params_maps = outputs["params_map"]
            else:
                center_maps, params_maps = outputs[0], outputs[1]

            ms = self.MAP_SIZE
            center_flat = center_maps[:, 0].reshape(1, -1)
            best_idx = center_flat.argmax(dim=1, keepdim=True)

            pmap = params_maps.reshape(1, -1, ms * ms)
            idx_exp = best_idx.unsqueeze(1).expand(1, pmap.shape[1], 1)
            params = pmap.gather(2, idx_exp).squeeze(2)

            cam   = params[:, 0:3]
            pose  = params[:, 3:75]
            betas = params[:, 75:85]
            return pose, betas, cam

    return ROMPv1ExportWrapper(model).eval()


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_to_onnx(wrapper, out_path: str, opset: int) -> None:
    import torch

    print(f"[export-romp] Exporting to ONNX (opset {opset}) → {out_path} …")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.zeros(1, 3, 512, 512, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy_input,
        out_path,
        opset_version=opset,
        input_names=["image"],
        output_names=["pose", "betas", "cam"],
        dynamic_axes={
            "image": {0: "batch"},
            "pose":  {0: "batch"},
            "betas": {0: "batch"},
            "cam":   {0: "batch"},
        },
        do_constant_folding=True,
        verbose=False,
    )
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"[export-romp] Done — {out_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_onnx(out_path: str) -> None:
    try:
        import onnx
    except ImportError:
        pip_install("onnx")
        import onnx

    print("[export-romp] Verifying ONNX model …")
    model = onnx.load(out_path)
    onnx.checker.check_model(model)

    # Print I/O shapes
    for inp in model.graph.input:
        shape = [d.dim_value or "?" for d in inp.type.tensor_type.shape.dim]
        print(f"  input  {inp.name}: {shape}")
    for out in model.graph.output:
        shape = [d.dim_value or "?" for d in out.type.tensor_type.shape.dim]
        print(f"  output {out.name}: {shape}")
    print("[export-romp] Model is valid.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # 1. Ensure torch is available
    ensure_torch()
    import torch
    print(f"[export-romp] torch {torch.__version__}")

    # 2. Resolve checkpoint path
    checkpoint = args.checkpoint
    model_variant = args.model

    if not checkpoint:
        romp_pkg_available = ensure_romp_package()
        cloned = clone_romp_repo(args.romp_dir)

        if not romp_pkg_available and not cloned:
            print("[export-romp] ERROR: Cannot obtain ROMP — neither pip install nor git clone succeeded.")
            sys.exit(1)

        checkpoint = download_checkpoint(model_variant, args.romp_dir)

    # 3. Build export wrapper — try BEV first, fall back to ROMP v1
    wrapper = None
    errors: list[str] = []

    if model_variant == "bev":
        try:
            wrapper = _build_bev_wrapper(checkpoint)
        except Exception as exc:
            errors.append(f"BEV export failed: {exc}")
            print(f"[export-romp] BEV wrapper failed: {exc}")
            print("[export-romp] Trying ROMP v1 fallback …")
            # Re-download ROMP v1 checkpoint if we don't have it
            try:
                v1_ckpt = download_checkpoint("romp", args.romp_dir)
                wrapper = _build_romp_v1_wrapper(v1_ckpt)
                print("[export-romp] Loaded ROMP v1 as fallback")
            except Exception as exc2:
                errors.append(f"ROMP v1 fallback failed: {exc2}")
    else:
        try:
            wrapper = _build_romp_v1_wrapper(checkpoint)
        except Exception as exc:
            errors.append(f"ROMP v1 export failed: {exc}")

    if wrapper is None:
        print("[export-romp] All model loading attempts failed:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)

    # 4. Export
    try:
        export_to_onnx(wrapper, args.out, args.opset)
    except Exception as exc:
        print(f"[export-romp] ONNX export failed: {exc}")
        raise

    # 5. Verify
    try:
        verify_onnx(args.out)
    except Exception as exc:
        print(f"[export-romp] Verification warning (non-fatal): {exc}")

    print(f"\n[export-romp] Export complete: {args.out}")
    print("Place the file at public/models/romp-bev-fp32.onnx and serve with Vite.")


if __name__ == "__main__":
    main()
