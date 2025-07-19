import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def load_engine(path):
    with open(path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def infer(image_array, engine):
    # stub: implement buffer allocation and inference
    return np.zeros((224,224), dtype=np.int32)

if __name__ == '__main__':
    import sys, cv2
    img = cv2.imread(sys.argv[1])
    engine = load_engine('inference/edge_vision_7b.engine')
    seg = infer(img, engine)
    print(seg.shape)
