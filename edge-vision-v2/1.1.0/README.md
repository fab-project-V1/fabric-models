# edge-vision-v2

## Overview

This directory contains the artifacts and scripts for the **edge-vision-v2** semantic‐segmentation model:

- `inference/`
  - `weights_fp32.onnx`, `weights_fp16.onnx`  
  - `weights_1_?_?_quant.onnx`  
  - `predictions/` (output masks)
- `scripts/`
  - `run_inference.py` – batch inference CLI
- `data/` – sample images & masks
- `checkpoints/` – raw PyTorch checkpoints

Two versions:
- **v1.0.0** – baseline ViT‐7B FP32 (92% COCO mIoU)
- **v1.1.0** – FP16+INT8 quantized (93% COCO mIoU)

---

## Prerequisites

1. **Python 3.10+** (we recommend 3.13 as used here)  
2. A virtual-env with:
   ```bash
   pip install torch torchvision transformers onnx onnxruntime onnxruntime-gpu numpy Pillow matplotlib
(Optional GPU) NVIDIA drivers ≥ 525, CUDA 12.x, cuDNN 9.x (for ONNX-Runtime GPU/TensorRT EP).

Quick Inference
bash
Copy
Edit
# activate your venv
cd edge-vision-v2/1.0.0   # or 1.1.0
python -m scripts.run_inference.py \
   data/v1.0/images/  \
   inference/predictions/
Results land in inference/predictions/.

Exporting & Quantizing
If you ever need to regenerate the ONNX or quantize:

bash
Copy
Edit
# Export FP32
python - << 'EOF'
import torch
from transformers import SegformerConfig, SegformerForSemanticSegmentation
ckpt = torch.load("checkpoints/edge_v2_1_?_?_epoch50.pt", map_location="cpu")
cfg = SegformerConfig(
  num_labels=21, encoder_hidden_size=1024,
  encoder_layers=24, encoder_attention_heads=16
)
model = SegformerForSemanticSegmentation(cfg)
model.load_state_dict(ckpt["model_state"])
model.eval()
torch.onnx.export(
  model, torch.randn(1,3,224,224),
  "inference/weights_fp32.onnx",
  opset_version=14,
  input_names=["pixel_values"],
  output_names=["logits"],
  do_constant_folding=True
)
print("✅ weights_fp32.onnx")
EOF

# Quantize to INT8
python -m onnxruntime.tools.quantization \
  --input inference/weights_fp32.onnx \
  --output inference/weights_1_?_?_quant.onnx \
  --mode QLinearOps
Packaging & Registration
Model manifest
Create model.yaml alongside:

yaml
Copy
Edit
name: edge-vision-v2
version: "1.0.0"         # or "1.1.0"
description: "7B‐parameter ViT semantic‐segmentation on edge"
framework: "ONNX-Runtime"
format: "onnx"
download_url: "./inference/weights_fp32.onnx"  # or the quantized ONNX
input_schema:
  type: object
  properties:
    pixel_values:
      type: array
      items: number
      description: "[1,3,224,224] normalized float tensor"
output_schema:
  type: object
  properties:
    logits:
      type: array
      items: number
      description: "[1,21,56,56] raw class scores"
parameter_count: 7000000000
tags: ["vision","semantic-segmentation","vit"]
author: "Shawn Blackmore"
license: "Apache-2.0"
provenance:
  training_commit: "commitid…"
  training_config: "training/config_?.yaml"
performance:
  metrics:
    - name: mIoU
      value: 0.92  # or 0.93
      dataset: COCO-2017
signatures:
  artifact_sig: "<sha256…>"
  manifest_sig: "<sha256…>"
Register via CLI

bash
Copy
Edit
fab model upload edge-vision-v2 1.0.0 \
  --weights inference/weights_fp32.onnx \
  --manifest model.yaml
Or point at the quantized ONNX / TensorRT engine if you prefer.

Tips & Troubleshooting
GPU support
If ONNX-Runtime can’t load CUDA/Trt EP, it’ll auto-fallback to CPU. Ensure your PATH includes:

powershell
Copy
Edit
$env:PATH = "<CUDA-12.9-zip-bin>;<TENSORRT-10.10-gz-lib>;$env:PATH"
Operators Not Implemented
If you hit a ConvInteger or similar error on CPU, regenerate the ONNX without QLinearOps, or fallback to FP32/FP16.

Dev Loop

(Re)train → 2. export → 3. quantize → 4. infer → 5. register.