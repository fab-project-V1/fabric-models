#!/usr/bin/env python3
import os
import sys
import json
import yaml
import argparse
from datetime import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
# make TensorBoard optional
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from PIL import Image
import numpy as np

from transformers import SegformerForSemanticSegmentation, SegformerConfig
from transformers import get_cosine_schedule_with_warmup

# --- Dataset Definition ---
class SegmentationDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        with open(annotations_file) as f:
            self.anns = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        item = self.anns[idx]
        image = Image.open(item['image']).convert('RGB')
        mask  = Image.open(item['mask']).convert('L')
        if self.transform:
            image = self.transform(image)
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        return {'pixel_values': image, 'labels': mask}

# --- Argument Parsing ---
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='Path to config.yaml')
    # For legacy torch.distributed.launch compatibility:
    p.add_argument('--local_rank', type=int, default=0, help='Local process rank')
    return p.parse_args()

# --- Distributed Setup ---
def setup_distributed(local_rank):
    use_cuda   = torch.cuda.is_available()
    is_windows = sys.platform.startswith('win')
    backend    = 'nccl' if use_cuda and not is_windows else 'gloo'

    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size > 1:
        dist.init_process_group(backend=backend)
        rank = int(os.environ.get('LOCAL_RANK', local_rank))
        if use_cuda:
            torch.cuda.set_device(rank)
        return rank, world_size, True
    else:
        return 0, 1, False

# --- Model Builder ---
def build_model(device):
    cfg = SegformerConfig(
        num_labels=21,
        encoder_hidden_size=1024,
        encoder_layers=24,
        encoder_attention_heads=16
    )
    model = SegformerForSemanticSegmentation(cfg)
    return model.to(device)

# --- Main Training & Export ---
def main():
    args = parse_args()
    rank, world_size, distributed = setup_distributed(args.local_rank)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    hp      = cfg['hyperparameters']
    sch_cfg = cfg['scheduler']

    # Cast hyperparameters to correct types
    hp['batch_size']          = int(hp['batch_size'])
    hp['learning_rate']       = float(hp['learning_rate'])
    hp['weight_decay']        = float(hp['weight_decay'])
    hp['epochs']              = int(hp['epochs'])
    hp['checkpoint_interval'] = int(hp['checkpoint_interval'])
    sch_cfg['warmup_fraction'] = float(sch_cfg['warmup_fraction'])

    # TensorBoard on rank 0 (if available)
    tb_writer = None
    if rank == 0 and SummaryWriter:
        logdir = f"runs/edge_v2_1_0_0_{datetime.now():%Y%m%d_%H%M%S}"
        tb_writer = SummaryWriter(logdir)

    # Data transforms & loader
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    ds = SegmentationDataset('training/annotations.json', transform)
    sampler = DistributedSampler(ds) if distributed else None
    loader = DataLoader(
        ds,
        batch_size=hp['batch_size'],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True,
    )

    # Build & wrap model
    model = build_model(device)
    if distributed:
        model = DDP(model, device_ids=[rank] if device.type=='cuda' else None)

    optimizer = AdamW(
        model.parameters(),
        lr=hp['learning_rate'],
        weight_decay=hp['weight_decay']
    )

    total_steps = len(loader) * hp['epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(sch_cfg['warmup_fraction'] * total_steps),
        num_training_steps=total_steps
    )
    criterion = CrossEntropyLoss(ignore_index=255)

    # Training loop
    for epoch in range(hp['epochs']):
        if distributed:
            sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(loader):
            imgs  = batch['pixel_values'].to(device, non_blocking=True)
            masks = batch['labels'].to(device, non_blocking=True)

            # Forward
            outputs = (model.module if distributed else model)(pixel_values=imgs).logits
            # Upsample logits to match mask resolution:
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(
                    outputs,
                    size=masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            loss = criterion(outputs, masks)

            # Backward + step
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            if rank == 0 and tb_writer and (batch_idx % 50 == 0):
                step = epoch * len(loader) + batch_idx
                tb_writer.add_scalar('train/loss', loss.item(), step)

        # Epoch end
        avg_loss = running_loss / len(loader)
        if rank == 0:
            print(f"[Epoch {epoch+1}/{hp['epochs']}] Loss: {avg_loss:.4f}")
            if tb_writer:
                tb_writer.add_scalar('train/epoch_loss', avg_loss, epoch)
            if (epoch+1) % hp['checkpoint_interval'] == 0:
                os.makedirs('checkpoints', exist_ok=True)
                ckpt = {
                    'epoch': epoch+1,
                    'model_state': model.module.state_dict() if distributed else model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict()
                }
                path = f"checkpoints/edge_v2_1_0_0_epoch{epoch+1}.pt"
                torch.save(ckpt, path)
                print(f"Saved checkpoint: {path}")

    # Final export & quantization on rank 0
    if rank == 0:
        os.makedirs('inference', exist_ok=True)
        sd = model.module.state_dict() if distributed else model.state_dict()
        torch.save(sd, 'inference/vit_7b.pth')

        dummy = torch.randn(1,3,224,224, device=device)
        torch.onnx.export(
            model.module if distributed else model,
            dummy,
            'weights.onnx',
            opset_version=14
        )
        print('✅ Exported weights.onnx')

        # Quantize
        os.system(
            'python -m onnxruntime.quantization '
            '--input weights.onnx '
            '--output weights_quant.onnx '
            '--quant_format QDQ '
            '--mode QLinearOps'
        )
        print('✅ Quantized to weights_quant.onnx')

        # TensorRT build
        import tensorrt as trt
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        builder    = trt.Builder(TRT_LOGGER)
        network    = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser     = trt.OnnxParser(network, TRT_LOGGER)
        with open('weights_quant.onnx','rb') as f:
            parser.parse(f.read())
        config_trt   = builder.create_builder_config()
        config_trt.max_workspace_size = 1<<30
        config_trt.set_flag(trt.BuilderFlag.FP16)
        config_trt.set_flag(trt.BuilderFlag.INT8)
        engine       = builder.build_engine(network, config_trt)
        with open('inference/edge_vision_7b.engine','wb') as f:
            f.write(engine.serialize())
        print('✅ Built TensorRT engine')

    if distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()







