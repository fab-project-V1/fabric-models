#!/usr/bin/env bash
# scripts/run_distributed.sh
# Launch distributed training for edge-vision-v2 v1.0.0

# Rendezvous settings
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# Number of nodes and procs per node
# Set NNODES=2 if you have two machines; otherwise keep 1
NNODES=${NNODES:-1}
PROC_PER_NODE=${PROC_PER_NODE:-4}

# Locate a Python interpreter with torch available
if python3 -c "import torch" &> /dev/null; then
  PYTHON_CMD="python3"
elif python -c "import torch" &> /dev/null; then
  PYTHON_CMD="python"
else
  echo "Error: neither 'python3' nor 'python' has torch installed."
  exit 1
fi

# Run with torch.distributed.run
$PYTHON_CMD -m torch.distributed.run \
  --nnodes $NNODES \
  --nproc_per_node $PROC_PER_NODE \
  --rdzv_id edge-v2-1.0.0 \
  --rdzv_backend c10d \
  training/train.py \
  --config training/config.yaml


