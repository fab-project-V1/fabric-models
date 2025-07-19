#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse

import numpy as np
import onnxruntime as ort
from PIL import Image

# Locate the ONNX model (FP16 or FP32) under inference/
def find_onnx(base: Path):
    for name in ("weights_fp16.onnx", "weights_fp32.onnx"):  # supported ONNX formats
        p = base / "inference" / name
        if p.exists():
            return p
    return None

# Preprocess an image into [1,3,224,224] float tensor
def preprocess(img_path: Path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = ((arr - mean) / std).transpose(2,0,1)[None,...]
    return inp

# Save a per-pixel mask (raw output shape [1,Nc,H,W]) as grayscale PNG
def save_mask(raw: np.ndarray, out_path: Path):
    # raw: [B, C, H, W]
    pred = raw.argmax(axis=1)       # [B, H, W]
    mask = pred[0].astype(np.uint8) # [H, W]
    Image.fromarray(mask).save(out_path)

# CLI argument parsing
def parse_args():
    p = argparse.ArgumentParser(description="Run semantic segmentation on image(s)")
    p.add_argument("input",  help="path to image file or directory of images")
    p.add_argument("output", help="directory to save prediction masks")
    return p.parse_args()

# Main execution
def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    onnx_path = find_onnx(root)
    if not onnx_path:
        sys.exit("❌ No ONNX model found under inference/")

    # Pick execution providers
    providers = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    print("✅ Session providers:", sess.get_providers())

    # Determine input dtype from model file name
    dtype = np.float16 if onnx_path.name.endswith("fp16.onnx") else np.float32

    src = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather images
    if src.is_dir():
        imgs = sorted(src.glob("*.jpg")) + sorted(src.glob("*.png"))
    else:
        imgs = [src]

    for img_path in imgs:
        inp = preprocess(img_path).astype(dtype)
        raw = sess.run(None, {sess.get_inputs()[0].name: inp})[0]
        out_path = out_dir / f"{img_path.stem}_mask.png"
        save_mask(raw, out_path)
        print(f"→ Saved: {out_path}")

    print("✅ All done.")

if __name__ == "__main__":
    main()

