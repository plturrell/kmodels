"""Standalone script to pre-compute fractal dimensions for all images."""

import argparse
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

from .fractal import compute_fractal_dimension, WaveletFractalFeatures

def precompute_fractal_dimensions(image_dir: Path, output_csv: Path):
    """Computes fractal and wavelet features for all JPG images and saves to CSV."""
    image_files = list(image_dir.glob("**/*.jpg"))
    if not image_files:
        print(f"Warning: No JPG images found in {image_dir}")
        return

    wavelet_extractor = WaveletFractalFeatures()
    results = []
    for img_path in tqdm(image_files, desc="Computing Fractal and Wavelet Features"):
        try:
            with Image.open(img_path) as img:
                img_gray = img.convert("L")
                img_array = np.array(img_gray) / 255.0
                
                # 1. Compute single fractal dimension
                fd = compute_fractal_dimension(img_array)
                
                # 2. Compute wavelet features
                wavelet_features = wavelet_extractor.extract(img_array)
                
                record = {"image_path": img_path.name, "fractal_dimension": fd}
                for i, wf in enumerate(wavelet_features):
                    record[f"wavelet_feat_{i}"] = wf
                
                results.append(record)
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Successfully saved {len(df)} fractal dimensions to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Pre-compute fractal dimensions for images.")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing training images.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Path to save the output CSV file.")
    args = parser.parse_args()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    precompute_fractal_dimensions(args.image_dir, args.output_csv)

if __name__ == "__main__":
    main()
