# export_onnx_1_0_0.py
import torch
from transformers import SegformerConfig, SegformerForSemanticSegmentation

# 1) Load your final checkpoint
ckpt = torch.load("checkpoints/edge_v2_1_0_0_epoch50.pt", map_location="cpu")

# 2) Rebuild the model skeleton
cfg = SegformerConfig(
    num_labels=21,
    encoder_hidden_size=1024,
    encoder_layers=24,
    encoder_attention_heads=16
)
model = SegformerForSemanticSegmentation(cfg)
model.load_state_dict(ckpt["model_state"])
model.eval()

# 3) Export to ONNX
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy,
    "weights_1_0_0.onnx",
    opset_version=14,
    input_names=["pixel_values"],
    output_names=["logits"]
)
print("✅ weights_1_0_0.onnx written")
