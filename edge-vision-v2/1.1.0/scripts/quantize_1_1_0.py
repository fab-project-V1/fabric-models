#!/usr/bin/env python3
# scripts/quantize_1_1_0.py

from onnxruntime.quantization import quantize_dynamic, QuantType

if __name__ == "__main__":
    # Quantize weights to int8, producing QLinearOps nodes
    quantize_dynamic(
        model_input="weights_1_1_0.onnx",
        model_output="weights_1_1_0_quant.onnx",
        weight_type=QuantType.QInt8,
    )
    print("✅ weights_1_1_0_quant.onnx written")
