#!/usr/bin/env python3
import sys, os
from pathlib import Path

# (1) Locate the script, then the model’s root (two levels up: .../1.0.0/scripts → 1.0.0)
script_dir = Path(__file__).resolve().parent
model_root = script_dir.parent

# (2) Build the path to the quantized ONNX in the model’s inference folder
onnx_path = model_root / "inference" / "weights_1_0_0_quant.onnx"
if not onnx_path.exists():
    sys.exit(f"❌ Cannot find quantized ONNX at '{onnx_path}'")

print(f"🔍 Found ONNX at {onnx_path}")

# (3) Now do the usual TRT build
try:
    import tensorrt as trt
except ImportError:
    sys.exit("❌ TensorRT Python bindings not found—check that your PATH and your wheel install are correct")

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
builder    = trt.Builder(TRT_LOGGER)
network    = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser     = trt.OnnxParser(network, TRT_LOGGER)

# parse
with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        sys.exit("❌ ONNX parse failed")

# builder config
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1 GiB
config.set_flag(trt.BuilderFlag.FP16)
config.set_flag(trt.BuilderFlag.INT8)

# build
engine = builder.build_engine(network, config)
if engine is None:
    sys.exit("❌ Failed to build TensorRT engine")

# write out
out_dir  = model_root / "inference"
out_path = out_dir / "edge_vision_7b_1_0_0.engine"
out_dir.mkdir(exist_ok=True)
with open(out_path, "wb") as f:
    f.write(engine.serialize())

print(f"✅ Engine saved to {out_path}")

