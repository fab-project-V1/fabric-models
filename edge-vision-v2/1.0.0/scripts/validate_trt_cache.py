#!/usr/bin/env python3
import os
import sys

import numpy as np
import onnxruntime as ort

def prepend_deps_to_env():
    # locate project root (two levels up from this script)
    script_dir   = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    deps_dir     = os.path.join(project_root, "deps")

    # your unpacked CUDA & TensorRT paths
    cuda_bin      = os.path.join(deps_dir, "cuda-12.9", "bin")
    tensorrt_lib  = os.path.join(deps_dir, "tensorrt-10.10-ga", "lib")

    # on Windows, augment PATH; on *nix, augment LD_LIBRARY_PATH
    if os.name == "nt":
        prior = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{cuda_bin};{tensorrt_lib};{prior}"
    else:
        prior = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_bin}:{tensorrt_lib}:{prior}"

def main():
    prepend_deps_to_env()

    onnx_path = "inference/weights_1_0_0_quant.onnx"
    if not os.path.exists(onnx_path):
        sys.exit(f"❌ ONNX not found at '{onnx_path}'")

    sess_opts = ort.SessionOptions()
    providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
    try:
        sess = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
    except Exception as e:
        sys.exit(f"❌ Failed to create ORT session with TRT+CUDA providers:\n{e}")

    print("✅ ONNX Runtime session created with providers:", sess.get_providers())

    # force engine build by dummy inference
    inp   = sess.get_inputs()[0]
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
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
