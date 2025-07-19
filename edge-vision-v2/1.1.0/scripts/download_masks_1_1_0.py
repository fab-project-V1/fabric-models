#!/usr/bin/env python3
"""
scripts/generate_masks_1_1_0.py

Generate pseudo-segmentation masks for Fabric edge-vision-v2 v1.1.0 dataset
using a pretrained DeepLabV3 ResNet101 model.

Defaults assume this script lives under models/edge-vision-v2/1.1.0/scripts/
so data directories resolve to:
  models/edge-vision-v2/1.1.0/data/v1.1/images
  models/edge-vision-v2/1.1.0/data/v1.1/masks

Usage (PowerShell):

# Install dependencies:
# pip install torch torchvision pillow
#
# Single-line (uses defaults):
# python .\scripts\generate_masks_1_1_0.py --device cuda
#
# Override directories if needed:
# python .\scripts\generate_masks_1_1_0.py `
#   --images_dir ".\data\v1.1\images" `
#   --masks_dir  ".\data\v1.1\masks" `
#   --device     cuda
"""
import os
import argparse
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101

def parse_args():
    # Determine default paths relative to this script
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_images = project_root / 'data' / 'v1.1' / 'images'
    default_masks  = project_root / 'data' / 'v1.1' / 'masks'

    parser = argparse.ArgumentParser(
        description="Generate pseudo-masks using DeepLabV3 ResNet101 for v1.1.0"
    )
    parser.add_argument(
        "--images_dir", default=str(default_images),
        help=f"Directory containing input images (default: {default_images})"
    )
    parser.add_argument(
        "--masks_dir", default=str(default_masks),
        help=f"Directory where masks will be saved (default: {default_masks})"
    )
    parser.add_argument(
        "--device", default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run the model on"
    )
    parser.add_argument(
        "--resize", type=int, nargs=2, default=[224, 224],
        metavar=("H", "W"),
        help="Resize images to H W before inference"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    masks_dir  = Path(args.masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = deeplabv3_resnet101(pretrained=True)
    model.to(device)
    model.eval()

    # Preprocessing transform
    preprocess = transforms.Compose([
        transforms.Resize(tuple(args.resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Generate masks
    for img_file in sorted(images_dir.iterdir()):
        if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        base = img_file.stem
        img_path = img_file
        mask_path = masks_dir / f"{base}.png"

        img = Image.open(img_path).convert("RGB")
        input_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)['out']  # [1,21,H,W]
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().byte()

        mask_img = Image.fromarray(pred.numpy())
        mask_img.save(mask_path)
        print(f"Saved mask: {mask_path}")


if __name__ == "__main__":
    main()
