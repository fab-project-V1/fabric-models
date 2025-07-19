#!/usr/bin/env python3
"""
training/generate_annotations.py

Collects image–mask pairs into a JSON for Fabric edge-vision-v2 v1.1.0.

Defaults assume this lives under:
  models/edge-vision-v2/1.1.0/training/
so data dirs resolve to:
  models/edge-vision-v2/1.1.0/data/v1.1/images
  models/edge-vision-v2/1.1.0/data/v1.1/masks
and output JSON to:
  models/edge-vision-v2/1.1.0/training/annotations.json
"""

import argparse
import json
from pathlib import Path

def parse_args():
    script_dir     = Path(__file__).resolve().parent
    project_root   = script_dir.parent
    default_images = project_root / 'data' / 'v1.1' / 'images'
    default_masks  = project_root / 'data' / 'v1.1' / 'masks'
    default_output = script_dir / 'annotations.json'

    p = argparse.ArgumentParser(
        description="Generate annotations JSON from images + masks (v1.1.0)"
    )
    p.add_argument(
        '--images_dir',
        type=Path,
        default=default_images,
        help=f"Directory of input images (default: {default_images})"
    )
    p.add_argument(
        '--masks_dir',
        type=Path,
        default=default_masks,
        help=f"Directory for masks output (default: {default_masks})"
    )
    p.add_argument(
        '--output',
        type=Path,
        default=default_output,
        help=f"Where to write the JSON (default: {default_output})"
    )
    return p.parse_args()

def main():
    args = parse_args()

    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {args.images_dir}")
    if not args.masks_dir.exists():
        raise FileNotFoundError(f"Masks dir not found:  {args.masks_dir}")

    imgs = {
        p.stem: p
        for p in args.images_dir.iterdir()
        if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
    }
    msks = {
        p.stem: p
        for p in args.masks_dir.iterdir()
        if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}
    }

    records = []
    for stem, img_path in sorted(imgs.items()):
        mask_path = msks.get(stem)
        if mask_path:
            records.append({
                'image': str(img_path),
                'mask':  str(mask_path)
            })
        else:
            print(f"Warning: no mask for image '{stem}', skipping")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(records, f, indent=2)
    print(f"Collected {len(records)} pairs → wrote {args.output}")

if __name__ == '__main__':
    main()

