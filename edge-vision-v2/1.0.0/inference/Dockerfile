FROM nvcr.io/nvidia/tensorrt:23.06-py3
WORKDIR /app
COPY edge_vision_7b.engine ./edge_vision_7b.engine
COPY infer.py ./
RUN pip install numpy pycuda opencv-python
ENTRYPOINT ["python", "infer.py"]
