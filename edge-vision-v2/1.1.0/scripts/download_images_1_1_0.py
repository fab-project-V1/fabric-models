#!/usr/bin/env python3
"""
scripts/download_images_1_1_0.py

Download images from Bing Image Search into the Fabric v1.1.0 dataset directory.

Usage (PowerShell):

# Install dependency:
#   pip install icrawler

# Single-line:
#   python .\scripts\download_images_1_1_0.py --query "urban street" --limit 500 --output_dir ".\data\v1.1\images"

# Multi-line with backticks:
#   python .\scripts\download_images_1_1_0.py `
#     --query "urban street" `
#     --limit 500 `
#     --output_dir ".\data\v1.1\images"
"""
import argparse
import os
from icrawler.builtin import BingImageCrawler

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download images from Bing into Fabric v1.1.0 data directory"
    )
    parser.add_argument(
        "--query", required=True,
        help="Search term, e.g. 'city street', 'indoor scene', etc."
    )
    parser.add_argument(
        "--limit", type=int, default=200,
        help="Maximum number of images to download"
    )
    parser.add_argument(
        "--output_dir", default="data/v1.1/images",
        help="Directory where images will be saved (default: data/v1.1/images)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Downloading up to {args.limit} images for '{args.query}' → {args.output_dir}")

    crawler = BingImageCrawler(
        feeder_threads=1,
        parser_threads=1,
        downloader_threads=4,
        storage={'root_dir': args.output_dir}
    )
    crawler.crawl(
        keyword=args.query,
        max_num=args.limit,
        min_size=(200, 200)
    )
    print("Download complete.")

if __name__ == "__main__":
    main()



