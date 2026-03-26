#!/usr/bin/env python3
"""
Export MASt3R (ViT-Large) to ONNX for browser inference.

Usage:
    python scripts/export-mast3r.py [--output public/models/mast3r-vit-large-fp32.onnx]

Requirements:
    torch>=2.0
    onnx>=1.14
    onnxruntime>=1.16
    huggingface_hub>=0.20

Steps performed:
    1. Clone MASt3R repo if not present at ./mast3r
    2. Install MASt3R requirements
    3. Download checkpoint from HuggingFace
    4. Wrap model to accept flat image tensors
    5. Export to ONNX (opset 17)
    6. Verify with onnxruntime and compare outputs to PyTorch
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Export MASt3R to ONNX")
parser.add_argument(
    "--output",
    default="public/models/mast3r-vit-large-fp32.onnx",
    help="Output ONNX file path (default: public/models/mast3r-vit-large-fp32.onnx)",
)
parser.add_argument(
    "--checkpoint-dir",
    default="checkpoints",
    help="Directory to store the downloaded checkpoint (default: checkpoints/)",
)
args = parser.parse_args()

OUTPUT_ONNX = Path(args.output)
CHECKPOINT_DIR = Path(args.checkpoint_dir)
MAST3R_REPO = Path("mast3r")
MAST3R_GIT = "https://github.com/naver/mast3r"
HF_REPO = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
CHECKPOINT_FILENAME = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

INPUT_SIZE = 512  # MASt3R canonical input resolution

# ---------------------------------------------------------------------------
# Step 1: Clone MASt3R if not present
# ---------------------------------------------------------------------------

if not MAST3R_REPO.exists():
    print(f"[export-mast3r] Cloning MASt3R from {MAST3R_GIT} ...")
    subprocess.check_call(
        ["git", "clone", "--recurse-submodules", MAST3R_GIT, str(MAST3R_REPO)]
    )
else:
    print(f"[export-mast3r] MASt3R repo already present at {MAST3R_REPO}")

# Add mast3r to sys.path so its modules are importable
sys.path.insert(0, str(MAST3R_REPO.resolve()))
sys.path.insert(0, str((MAST3R_REPO / "dust3r").resolve()))

# ---------------------------------------------------------------------------
# Step 2: Install dependencies
# ---------------------------------------------------------------------------

req_file = MAST3R_REPO / "requirements.txt"
if req_file.exists():
    print(f"[export-mast3r] Installing MASt3R requirements from {req_file} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
else:
    print(f"[export-mast3r] WARNING: {req_file} not found; skipping pip install")

# Ensure huggingface_hub is available for the download step
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.20"]
)

# ---------------------------------------------------------------------------
# Step 3: Download checkpoint
# ---------------------------------------------------------------------------

import torch  # noqa: E402  (must come after pip install)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_FILENAME

if not checkpoint_path.exists():
    print(f"[export-mast3r] Downloading checkpoint from HuggingFace: {HF_REPO} ...")
    from huggingface_hub import hf_hub_download  # noqa: E402

    downloaded = hf_hub_download(
        repo_id=HF_REPO,
        filename=CHECKPOINT_FILENAME,
        local_dir=str(CHECKPOINT_DIR),
    )
    print(f"[export-mast3r] Checkpoint saved to {downloaded}")
else:
    print(f"[export-mast3r] Checkpoint already present at {checkpoint_path}")

# ---------------------------------------------------------------------------
# Step 4: Load the model
# ---------------------------------------------------------------------------

# MASt3R model class lives in mast3r.model
from mast3r.model import AsymmetricMASt3R  # noqa: E402

print("[export-mast3r] Loading AsymmetricMASt3R from checkpoint ...")
model = AsymmetricMASt3R.from_pretrained(str(checkpoint_path))
model.eval()

# Determine device — use CPU for export to avoid device-specific ops in the graph
export_device = "cpu"
model = model.to(export_device)
print(f"[export-mast3r] Model on device: {export_device}")

# ---------------------------------------------------------------------------
# Step 5: Define the export wrapper
# ---------------------------------------------------------------------------

import torch.nn as nn  # noqa: E402


class MASt3RExportWrapper(nn.Module):
    """
    Thin wrapper that accepts flat image tensors instead of dicts.

    MASt3R's forward(view1, view2) expects each view to be a dict with
    at least the key 'img'.  This wrapper builds those dicts internally
    so that torch.onnx.export sees only plain tensor I/O.

    Inputs:
        image1: [B, 3, H, W] float32 in [-1, 1]
        image2: [B, 3, H, W] float32 in [-1, 1]

    Outputs (tuple):
        pts3d_1: [B, H, W, 3]   — 3D point map for image 1
        pts3d_2: [B, H, W, 3]   — 3D point map for image 2
        conf_1:  [B, H, W]      — confidence for image 1
        conf_2:  [B, H, W]      — confidence for image 2
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
    ):
        view1 = {"img": image1}
        view2 = {"img": image2}

        # MASt3R forward returns a list/tuple of dicts, one per view pair.
        # The exact keys depend on the decoder head; we extract pts3d and conf.
        res = self.model(view1, view2)

        # res is typically (pred1, pred2) where each is a dict
        pred1, pred2 = res[0], res[1]

        pts3d_1 = pred1["pts3d"]           # [B, H, W, 3]
        pts3d_2 = pred2["pts3d_in_other_view"]  # [B, H, W, 3]
        conf_1 = pred1["conf"]             # [B, H, W]
        conf_2 = pred2["conf"]             # [B, H, W]

        return pts3d_1, pts3d_2, conf_1, conf_2


wrapped = MASt3RExportWrapper(model)
wrapped.eval()

# ---------------------------------------------------------------------------
# Step 6: Create dummy inputs
# ---------------------------------------------------------------------------

B = 1
dummy_image1 = torch.zeros(B, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)
dummy_image2 = torch.zeros(B, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)

print("[export-mast3r] Running forward pass to validate wrapper ...")
with torch.no_grad():
    torch_out = wrapped(dummy_image1, dummy_image2)

pts3d_1_pt, pts3d_2_pt, conf_1_pt, conf_2_pt = torch_out
print(f"[export-mast3r] pts3d_1 shape: {pts3d_1_pt.shape}")
print(f"[export-mast3r] pts3d_2 shape: {pts3d_2_pt.shape}")
print(f"[export-mast3r] conf_1  shape: {conf_1_pt.shape}")
print(f"[export-mast3r] conf_2  shape: {conf_2_pt.shape}")

# ---------------------------------------------------------------------------
# Step 7: Export to ONNX
# ---------------------------------------------------------------------------

OUTPUT_ONNX.parent.mkdir(parents=True, exist_ok=True)

print(f"[export-mast3r] Exporting to ONNX: {OUTPUT_ONNX} ...")

torch.onnx.export(
    wrapped,
    (dummy_image1, dummy_image2),
    str(OUTPUT_ONNX),
    opset_version=17,
    input_names=["image1", "image2"],
    output_names=["pts3d_1", "pts3d_2", "conf_1", "conf_2"],
    dynamic_axes={
        "image1": {0: "batch"},
        "image2": {0: "batch"},
    },
    do_constant_folding=True,
    verbose=False,
)

file_size_mb = OUTPUT_ONNX.stat().st_size / 1e6
print(f"[export-mast3r] ONNX export complete: {OUTPUT_ONNX} ({file_size_mb:.1f} MB)")

# ---------------------------------------------------------------------------
# Step 8: Verify with onnxruntime — compare to PyTorch outputs
# ---------------------------------------------------------------------------

try:
    import onnxruntime as ort  # noqa: E402
    import numpy as np  # noqa: E402

    print("[export-mast3r] Verifying ONNX outputs with onnxruntime ...")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(OUTPUT_ONNX), sess_options)

    np_image1 = dummy_image1.numpy()
    np_image2 = dummy_image2.numpy()

    ort_outputs = session.run(
        None,
        {"image1": np_image1, "image2": np_image2},
    )

    names = ["pts3d_1", "pts3d_2", "conf_1", "conf_2"]
    torch_tensors = [t.detach().numpy() for t in torch_out]

    all_ok = True
    for name, ort_out, pt_out in zip(names, ort_outputs, torch_tensors):
        max_diff = float(np.max(np.abs(ort_out - pt_out)))
        status = "OK" if max_diff < 1e-3 else "MISMATCH"
        if status != "OK":
            all_ok = False
        print(f"  {name}: max_abs_diff={max_diff:.2e}  [{status}]")

    if all_ok:
        print("[export-mast3r] Verification PASSED — ONNX matches PyTorch within 1e-3")
    else:
        print("[export-mast3r] WARNING: Some outputs differ by more than 1e-3")

except ImportError:
    print(
        "[export-mast3r] onnxruntime not installed — skipping verification.\n"
        "  Install with: pip install onnxruntime>=1.16"
    )

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("Export complete!")
print(f"  Output : {OUTPUT_ONNX.resolve()}")
print(f"  Size   : {file_size_mb:.1f} MB")
print()
print("Place the file at:  public/models/mast3r-vit-large-fp32.onnx")
print("The browser node will load it from:  /models/mast3r-vit-large-fp32.onnx")
print("=" * 70)
