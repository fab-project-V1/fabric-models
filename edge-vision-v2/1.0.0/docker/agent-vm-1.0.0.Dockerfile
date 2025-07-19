# docker/agent-vm-1.0.0.Dockerfile
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Install SGX DCAP and system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git python3.9 python3-pip \
    libsgx-dcap-ql libsgx-dcap-dev libsgx-dcap-default-qpl \
    && rm -rf /var/lib/apt/lists/*

# Symlink python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Copy and install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# (Optional) Install NVIDIA DCGM-SGX plugin for SGX–GPU attestation
# RUN apt-get update && apt-get install -y --no-install-recommends dcgm-sgx-plugin

WORKDIR /app
