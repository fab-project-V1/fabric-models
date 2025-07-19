#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# ─── CONFIGURE YOUR LOCAL DEPENDENCY PATHS ────────────────────────────────────
# Adjust these if you unpacked somewhere else:
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CUDA_DIR      = PROJECT_ROOT / "deps" / "cuda-12.9"
TENSORRT_DIR  = PROJECT_ROOT / "deps" / "tensorrt-10.10-ga"

# For Windows, DLL directories go on PATH. For Linux, use LD_LIBRARY_PATH.
if os.name == "nt":
    cuda_bin = str(CUDA_DIR / "bin")
    trt_lib  = str(TENSORRT_DIR / "lib")
    os.environ["PATH"] = os.pathsep.join([cuda_bin, trt_lib, os.environ.get("PATH", "")])
else:
    cuda_lib = str(CUDA_DIR / "lib")
    trt_lib  = str(TENSORRT_DIR / "lib")
    ldpaths  = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([cuda_lib, trt_lib, ldpaths])

# ─── NOW IMPORT TENSORRT ─────────────────────────────────────────────────────
try:
    import tensorrt as trt
except ImportError as e:
    sys.stderr.write(
        "\nERROR: could not import tensorrt. Make sure your local deps are in:\n"
        f"  {CUDA_DIR}/bin (or lib)\n"
        f"  {TENSORRT_DIR}/lib     \n\n"
        "If you're on Linux, check LD_LIBRARY_PATH; on Windows, check PATH.\n"
    )
    sys.exit(1)

# ─── BUILD ENGINE ─────────────────────────────────────────────────────────────
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
builder    = trt.Builder(TRT_LOGGER)
network    = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser     = trt.OnnxParser(network, TRT_LOGGER)

onnx_path = Path("inference") / "weights_1_0_0_quant.onnx"
if not onnx_path.exists():
    sys.exit(f"❌ Quantized ONNX not found at '{onnx_path}'")

with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        sys.exit("❌ Failed to parse ONNX model")

config = builder.create_builder_config()
config.max_workspace_size = 1 << 30   # 1 GiB
config.set_flag(trt.BuilderFlag.FP16)
config.set_flag(trt.BuilderFlag.INT8)

engine = builder.build_engine(network, config)
if engine is None:
    sys.exit("❌ TensorRT engine build failed!")

out_path = Path("inference") / "edge_vision_7b_1_0_0.engine"
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "wb") as out:
    out.write(engine.serialize())

print(f"✅ TensorRT engine saved to {out_path}")
