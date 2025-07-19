# tests/test_inference_1_1_0.py

import os
import numpy as np
import pytest
from PIL import Image

from inference.infer import load_engine, infer

# Path to the TensorRT engine built for edge-vision-v2:1.1.0
ENGINE_PATH = os.path.abspath("inference/edge_vision_7b_1_1_0.engine")
# Directory containing your real test images and masks
TEST_DATA_DIR = os.path.join("tests", "data", "v1.1")

@pytest.mark.parametrize("idx", [1, 2])
def test_inference_output(idx):
    """
    Load the TensorRT engine, run inference on a real image,
    and verify that the output segmentation map matches
    the ground-truth mask shape and contains valid class IDs.
    """
    img_path  = os.path.join(TEST_DATA_DIR, f"img{idx}.jpg")
    mask_path = os.path.join(TEST_DATA_DIR, f"mask{idx}.png")

    # Ensure test assets exist
    assert os.path.isfile(img_path),  f"Test image not found: {img_path}"
    assert os.path.isfile(mask_path), f"Test mask not found: {mask_path}"
    assert os.path.isfile(ENGINE_PATH), f"Engine file not found: {ENGINE_PATH}"

    # Load and preprocess image
    img = np.array(Image.open(img_path).resize((224, 224)), dtype=np.uint8)
    # Load and resize mask
    mask = np.array(Image.open(mask_path).resize((224, 224)), dtype=np.int32)

    # Load the TensorRT engine
    engine = load_engine(ENGINE_PATH)
    assert engine is not None, "Failed to load TensorRT engine"

    # Run inference
    output = infer(img, engine)
    assert isinstance(output, np.ndarray), "Inference output is not an ndarray"
    assert output.shape == mask.shape, f"Output shape {output.shape} != mask shape {mask.shape}"

    # Check that all values are valid class IDs
    assert output.min() >= 0,   "Output contains negative class IDs"
    assert output.max() < 21,   "Output contains class IDs outside [0, 20]"

