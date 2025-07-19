import os
import subprocess
import numpy as np
import pytest
import torch
from PIL import Image

from inference.infer import load_engine, infer

# Paths for sample test data (you must provide these)
TEST_IMG_DIR  = os.path.join(os.path.dirname(__file__), "data")
IMG_PATHS     = [os.path.join(TEST_IMG_DIR, f"img{i}.jpg")   for i in (1,2)]
MASK_PATHS    = [os.path.join(TEST_IMG_DIR, f"mask{i}.png")  for i in (1,2)]
ENGINE_PATH   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../inference/edge_vision_7b.engine"))

@pytest.mark.parametrize("img_path,mask_path", zip(IMG_PATHS, MASK_PATHS))
def test_inference_engine_output(img_path, mask_path):
    """
    Load the TensorRT engine, run inference on a real image,
    and check output shape matches the ground-truth mask.
    """
    assert os.path.isfile(img_path), f"Test image not found: {img_path}"
    assert os.path.isfile(mask_path), f"Mask not found: {mask_path}"
    assert os.path.isfile(ENGINE_PATH), "Engine file missing"

    # Load engine
    engine = load_engine(ENGINE_PATH)
    assert engine is not None

    # Load image and mask
    img = np.array(Image.open(img_path).resize((224,224)), dtype=np.uint8)
    mask = np.array(Image.open(mask_path).resize((224,224)), dtype=np.int32)

    # Run inference
    out = infer(img, engine)
    # Should be a 2D array of ints
    assert isinstance(out, np.ndarray)
    assert out.ndim == 2
    assert out.shape == mask.shape

    # Values should be valid class IDs (0–20)
    assert out.min() >= 0 and out.max() < 21

@pytest.mark.timeout(600)  # allow up to 10 minutes
def test_training_smoke(tmp_path, monkeypatch):
    """
    Run a single-GPU, single-epoch training pass to smoke-test DDP setup.
    Uses a tiny synthetic dataset of 4 samples.
    """
    # Create a tiny synthetic annotations.json
    ann = []
    for i in range(4):
        img = tmp_path / f"img{i}.jpg"
        mask = tmp_path / f"mask{i}.png"
        # Create blank image/mask
        Image.new("RGB", (224,224), color=(i*20,i*20,i*20)).save(img)
        Image.new("L",   (224,224), color=0).save(mask)
        ann.append({"image": str(img), "mask": str(mask)})

    ann_file = tmp_path / "annotations.json"
    ann_file.write_text(json.dumps(ann, indent=2))

    # Create a minimal config.yaml
    cfg = {
        "distributed": {
            "backend": "nccl",
            "nnodes": 1,
            "nproc_per_node": 1
        },
        "hyperparameters": {
            "batch_size": 2,
            "learning_rate": 1e-4,
            "epochs": 1,
            "weight_decay": 1e-2,
            "checkpoint_interval": 1
        },
        "scheduler": {
            "type": "cosine",
            "warmup_fraction": 0.1
        }
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg))

    # Monkeypatch the paths inside training script
    monkeypatch.chdir(os.getcwd())  # ensure relative paths resolve
    env = os.environ.copy()
    env["LOCAL_RANK"] = "0"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29500"

    # Run training with torchrun for 1 proc
    cmd = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=1",
        "--rdzv_id=smoke_test",
        "--rdzv_backend=c10d",
        "training/train.py",
        "--config", str(cfg_file)
    ]
    # Inject our synthetic annotations path
    env["TRAINING_ANNOTATIONS"] = str(ann_file)
    # You may need to add logic in train.py to read os.environ["TRAINING_ANNOTATIONS"]
    # for this smoke-test; or copy the file to training/annotations.json here:
    os.makedirs("training", exist_ok=True)
    (tmp_path / "annotations.json").rename("training/annotations.json")

    subprocess.check_call(cmd, env=env)

    # Check that checkpoints and inference files were generated
    assert os.path.isdir("checkpoints")
    assert any(f.startswith("edge_v2_1_0_0_epoch1") for f in os.listdir("checkpoints"))
    assert os.path.isfile("weights.onnx")


