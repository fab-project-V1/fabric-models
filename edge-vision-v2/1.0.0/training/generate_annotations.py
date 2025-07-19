#!/usr/bin/env python3
"""
training/generate_annotations.py

Generate a JSON file pairing each image with its segmentation mask.

Usage:
  python training/generate_annotations.py \
    --images_dir path/to/images \
    --masks_dir  path/to/masks \
    --output     path/to/annotations.json
"""
import argparse
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate annotations JSON for image/mask pairs."
    )
    parser.add_argument(
        "--images_dir", required=True,
        help="Directory containing input images"
    )
    parser.add_argument(
        "--masks_dir", required=True,
        help="Directory containing segmentation masks"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to write the annotations JSON"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)

    # Collect all image files (.jpg, .jpeg, .png)
    images = {f.stem: str(f) for f in images_dir.iterdir()
              if f.suffix.lower() in ['.jpg', '.jpeg', '.png']}

    # Collect all mask files (.png)
    masks = {f.stem: str(f) for f in masks_dir.iterdir()
             if f.suffix.lower() in ['.png', '.jpg', '.jpeg']}

    # Find common base filenames
    common_keys = sorted(images.keys() & masks.keys())

    annotations = []
    for key in common_keys:
        annotations.append({
            'image': images[key],
            'mask':  masks[key]
        })

    # Write out JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f"Collected {len(annotations)} annotations")
    print(f"Wrote annotations to {args.output}")


if __name__ == '__main__':
    main()

