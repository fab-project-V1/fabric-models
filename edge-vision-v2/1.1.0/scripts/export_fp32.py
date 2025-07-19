#!/usr/bin/env python3
import torch
from transformers import SegformerConfig, SegformerForSemanticSegmentation
import os

# make sure the output folder exists
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "inference"), exist_ok=True)

# 1) Load your checkpoint
ckpt = torch.load(
    os.path.join(os.path.dirname(__file__), "..", "checkpoints", "edge_v2_1_1_0_epoch50.pt"),
    map_location="cpu"
)

# 2) Rebuild model skeleton
cfg = SegformerConfig(
    num_labels=21,
    encoder_hidden_size=1024,
    encoder_layers=24,
    encoder_attention_heads=16
)
model = SegformerForSemanticSegmentation(cfg)
model.load_state_dict(ckpt["model_state"])
model.eval()

# 3) Export to FP32 ONNX
dummy = torch.randn(1, 3, 224, 224)
outpath = os.path.join(os.path.dirname(__file__), "..", "inference", "weights_fp32.onnx")
torch.onnx.export(
    model,
    dummy,
    outpath,
    opset_version=14,
    input_names=["pixel_values"],
    output_names=["logits"],
    do_constant_folding=True,
)
print(f"✅ {outpath} written")
