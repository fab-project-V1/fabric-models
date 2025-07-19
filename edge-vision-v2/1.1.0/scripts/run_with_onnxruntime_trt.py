#!/usr/bin/env python3
import os
import sys
from pathlib import Path

def prepend_deps_to_env():
    # Adjust these folder names to match your unpack under deps/
    CUDA_FOLDER     = "cuda-12.9"
    TENSORRT_FOLDER = "tensorrt-10.10-ga"

    # locate project root
    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    deps_dir     = project_root / "deps"

    cuda_bin     = deps_dir / CUDA_FOLDER / "bin"
    tensorrt_lib = deps_dir / TENSORRT_FOLDER / "lib"

    if os.name == "nt":
        # Windows DLL search path
        prev = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{cuda_bin};{tensorrt_lib};{prev}"
    else:
        # Linux shared‐object search path
        prev = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_bin}:{tensorrt_lib}:{prev}"

# Prepend before importing onnxruntime
prepend_deps_to_env()

import numpy as np
import onnxruntime as ort

def main():
    onnx_path = Path("inference/weights_1_1_0_quant.onnx")
    if not onnx_path.exists():
        sys.exit(f"❌ Quantized ONNX not found at '{onnx_path}'")

    # create session options
    sess_opts = ort.SessionOptions()

    # Try TRT EP first, then CUDA EP
    providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
    try:
        sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=providers)
    except Exception as e:
        sys.exit(f"❌ Failed to create ORT session with TRT+CUDA providers:\n{e}")

    print("✅ ONNX Runtime session created with providers:", sess.get_providers())

    # dummy inference to force engine build & cache
    inp    = sess.get_inputs()[0]
    dummy  = np.random.randn(1, 3, 224, 224).astype(np.float32)
    try:
        outputs = sess.run(None, {inp.name: dummy})
    except Exception as e:
        sys.exit(f"❌ Dummy inference failed:\n{e}")

    print("✅ Dummy inference OK; output shapes =", [o.shape for o in outputs])
    print()
    print("✅ TensorRT engine has been built & cached by ONNX Runtime.")
    print("   Cache location:")
    if os.name == "nt":
        print("     %LOCALAPPDATA%\\onnxruntime\\tensorrt")
    else:
        print("     ~/.cache/onnxruntime/tensorrt")

if __name__ == "__main__":
    main()
