#!/usr/bin/env python3
import torch
from transformers import SegformerConfig, SegformerForSemanticSegmentation

# 1) Load your checkpoint
ckpt = torch.load("checkpoints/edge_v2_1_1_0_epoch50.pt", map_location="cpu")

# 2) Rebuild the model
cfg = SegformerConfig(
    num_labels=21,
    encoder_hidden_size=1024,
    encoder_layers=24,
    encoder_attention_heads=16
)
model = SegformerForSemanticSegmentation(cfg)
model.load_state_dict(ckpt["model_state"])
model.eval()

# 3) Cast to fp16 then export
model.half()
dummy = torch.randn(1, 3, 224, 224).half()
torch.onnx.export(
    model,
    dummy,
    "inference/weights_fp16.onnx",
    opset_version=14,
    input_names=["pixel_values"],
    output_names=["logits"],
    do_constant_folding=True,
)
print("✅ inference/weights_fp16.onnx written")
