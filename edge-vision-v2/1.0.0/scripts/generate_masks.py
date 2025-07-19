#!/usr/bin/env python3
"""
scripts/generate_masks.py

Generate pseudo-segmentation masks for Fabric edge-vision-v2 datasets
using a pretrained DeepLabV3 ResNet101 model.

Usage (PowerShell):

# Install dependencies:
# pip install torch torchvision pillow
#
# Single-line:
# python .\scripts\generate_masks.py --images_dir ".\data\v1.0\images" --masks_dir ".\data\v1.0\masks" --device cuda
#
# Multi-line with backticks:
# python .\scripts\generate_masks.py `
#   --images_dir ".\data\v1.0\images" `
#   --masks_dir ".\data\v1.0\masks" `
#   --device cuda
"""
import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate pseudo-masks using DeepLabV3 ResNet101"
    )
    parser.add_argument(
        "--images_dir", required=True,
        help="Directory containing input images"
    )
    parser.add_argument(
        "--masks_dir", required=True,
        help="Directory where masks will be saved"
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
    os.makedirs(args.masks_dir, exist_ok=True)

    # Load pretrained model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = deeplabv3_resnet101(pretrained=True)
    model.to(device)
    model.eval()

    # Transforms
    preprocess = transforms.Compose([
        transforms.Resize(tuple(args.resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Process each image
    for fname in sorted(os.listdir(args.images_dir)):
        base, ext = os.path.splitext(fname)
        if ext.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        img_path = os.path.join(args.images_dir, fname)
        mask_path = os.path.join(args.masks_dir, base + ".png")

        # Load and preprocess
        img = Image.open(img_path).convert("RGB")
        input_tensor = preprocess(img).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            output = model(input_tensor)["out"]  # [1, 21, H, W]
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().byte()

        # Save mask
        mask_img = Image.fromarray(pred.numpy())
        mask_img.save(mask_path)
        print(f"Saved mask: {mask_path}")

if __name__ == "__main__":
    main()
