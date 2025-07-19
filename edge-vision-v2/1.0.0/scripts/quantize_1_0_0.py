#!/usr/bin/env python3
# scripts/quantize_1_0_0.py

from onnxruntime.quantization import quantize_dynamic, QuantType

if __name__ == "__main__":
    # This will produce a QLinearOps model, quantizing weights to int8
    quantize_dynamic(
        model_input="weights_1_0_0.onnx",
        model_output="weights_1_0_0_quant.onnx",
        weight_type=QuantType.QInt8,
    )
    print("✅ weights_1_0_0_quant.onnx written")

