# Model Export Scripts

This directory contains Python scripts that export pre-trained neural networks
to ONNX format for use with ONNX Runtime Web in the browser.

---

## Python Requirements

All scripts require Python 3.10+ and:

```
torch>=2.0
onnx>=1.14
onnxruntime>=1.16
huggingface_hub>=0.20
```

Install everything at once:

```bash
pip install torch onnx onnxruntime huggingface_hub
```

---

## MASt3R (ViT-Large, 512×512)

### What it does

Exports the NAVER MASt3R model — a Matching and Stereo 3D Reconstruction
network — to ONNX.  The model takes two images and produces dense 3D point
maps and confidence scores for each, enabling metric-scale depth and camera
pose estimation without calibration data.

### How to run

From the repository root:

```bash
python scripts/export-mast3r.py
```

Optional flags:

```
--output PATH         Output ONNX file path
                      (default: public/models/mast3r-vit-large-fp32.onnx)
--checkpoint-dir DIR  Directory to cache the HuggingFace checkpoint
                      (default: checkpoints/)
```

The script will:
1. Clone `https://github.com/naver/mast3r` into `./mast3r/` (first run only)
2. `pip install` MASt3R's requirements
3. Download the ViT-Large checkpoint from HuggingFace
   (`naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`)
4. Wrap the model for flat-tensor ONNX I/O
5. Export with opset 17, batch-dynamic axes
6. Verify outputs against onnxruntime (max abs diff reported per output)

### Output file location

Place (or symlink) the exported file at:

```
public/models/mast3r-vit-large-fp32.onnx
```

The browser node (`demo/josh/nodes/mast3r.node.ts`) loads it from the URL
`/models/mast3r-vit-large-fp32.onnx`.

### Expected file size

| Model                      | Size (approx.) |
|----------------------------|----------------|
| mast3r-vit-large-fp32.onnx | ~500–700 MB    |

The file is large because ViT-Large has ~307 M parameters. The browser node
caches it in the Cache API after the first download, so subsequent page loads
are instant.

### ONNX I/O specification

**Inputs**

| Name    | Shape         | Dtype   | Range  | Description              |
|---------|---------------|---------|--------|--------------------------|
| image1  | [B, 3, 512, 512] | float32 | [-1, 1] | First image (NCHW)  |
| image2  | [B, 3, 512, 512] | float32 | [-1, 1] | Second image (NCHW) |

**Outputs**

| Name     | Shape           | Dtype   | Description                              |
|----------|-----------------|---------|------------------------------------------|
| pts3d_1  | [B, 512, 512, 3] | float32 | 3D point map for image 1 (XYZ per pixel) |
| pts3d_2  | [B, 512, 512, 3] | float32 | 3D point map for image 2 (XYZ per pixel) |
| conf_1   | [B, 512, 512]   | float32 | Confidence map for image 1               |
| conf_2   | [B, 512, 512]   | float32 | Confidence map for image 2               |

Batch dimension `B` is dynamic; the browser node always uses `B=1`.

### Preprocessing (handled by the browser node)

- Resize to 512×512 (bilinear)
- Normalize RGB from [0, 255] to [-1, 1]:  `pixel = (pixel / 127.5) - 1.0`
- Rearrange HWC → NCHW

---

## Adding more models

Follow this pattern when adding a new export script:

1. Create `scripts/export-<name>.py`
2. Output to `public/models/<name>.onnx`
3. Add a corresponding section to this README
4. Create `demo/josh/nodes/<name>.node.ts` following the `MASt3RNode` pattern
