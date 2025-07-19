#!/usr/bin/env bash
#
# scripts/run_distributed_1_1_0.sh
# Launch 2 nodes × 4 GPUs per node for edge-vision-v2:1.1.0 training.

# Ensure these are set (or export them in your environment beforehand)
#!/usr/bin/env bash
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500


torchrun \
  --nnodes 2 \
  --nproc_per_node 4 \
  --rdzv_id edge-v2-1.1.0 \
  --rdzv_backend c10d \
  training/train.py \
  --config training/config_1_1_0.yaml
