#!/usr/bin/env bash
#
# quantize_and_build.sh
# Quantize the ONNX model and build a TensorRT engine from the quantized graph.

set -e

# 1) Quantize FP32 ONNX to QLinearOps (INT8 + FP16 support)
python -m onnxruntime.quantization \
  --input ../weights.onnx \
  --output ../weights_quant.onnx \
  --quant_format QDQ              \
  --accelerate              \
  --mode QLinearOps

echo "✅ Quantized ONNX written to weights_quant.onnx"

# 2) Build TensorRT engine from quantized model
python - << 'PYCODE'
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
builder    = trt.Builder(TRT_LOGGER)
network    = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser     = trt.OnnxParser(network, TRT_LOGGER)

# Read the quantized ONNX
with open("../weights_quant.onnx", "rb") as f:
    if not parser.parse(f.read()):
        print("ERROR: Failed to parse ONNX")
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        exit(1)

# Builder configuration for FP16 + INT8
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30         # 1 GB scratch
config.set_flag(trt.BuilderFlag.FP16)       # enable FP16
config.set_flag(trt.BuilderFlag.INT8)       # enable INT8

# (Optional) attach an INT8 calibrator here if you have calibration data
# e.g. config.int8_calibrator = MyCalibrator(...)

engine = builder.build_engine(network, config)
with open("inference/edge_vision_7b.engine", "wb") as f:
    f.write(engine.serialize())
print("✅ TensorRT engine built at inference/edge_vision_7b.engine")
PYCODE
