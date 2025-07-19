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

# --- Dataset ---
class SegmentationDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        with open(annotations_file, encoding="utf-8") as f:
            self.anns = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        rec = self.anns[idx]
        img = Image.open(rec["image"]).convert("RGB")
        m   = Image.open(rec["mask"]).convert("L")
        if self.transform:
            img = self.transform(img)
        mask = torch.tensor(np.array(m), dtype=torch.long)
        return {"pixel_values": img, "labels": mask}

# --- Utility for mIoU ---
def compute_miou(preds: torch.Tensor, labels: torch.Tensor,
                 num_classes=21, ignore_index=255):
    per = []
    for cls in range(num_classes):
        if cls == ignore_index: continue
        p = (preds == cls)
        l = (labels == cls)
        inter = (p & l).sum().float()
        union = (p | l).sum().float()
        if union == 0: continue
        per.append(inter / union)
    return torch.mean(torch.stack(per)) if per else torch.tensor(0.0, device=preds.device)

# --- CLI args ---
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to config YAML")
    p.add_argument("--local_rank", type=int, default=0,
                   help="Local rank for torch.distributed")
    return p.parse_args()

# --- Distributed setup ---
def setup_distributed(local_rank):
    use_cuda   = torch.cuda.is_available()
    is_win     = sys.platform.startswith("win")
    backend    = "nccl" if (use_cuda and not is_win) else "gloo"
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend=backend)
        rank = int(os.environ.get("LOCAL_RANK", local_rank))
        if use_cuda:
            torch.cuda.set_device(rank)
        return rank, world_size, True
    return 0, 1, False

# --- Model builder ---
def build_model(device):
    cfg = SegformerConfig(
        num_labels=21,
        encoder_hidden_size=1024,
        encoder_layers=24,
        encoder_attention_heads=16
    )
    model = SegformerForSemanticSegmentation(cfg)
    return model.to(device)

# --- Main ---
def main():
    args = parse_args()
    rank, world_size, distributed = setup_distributed(args.local_rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load config as UTF-8
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # pick up annotation file (must be set in your config_1_1_0.yaml)
    ann_file = cfg.get("annotations_file", "training/annotations.json")
    assert os.path.exists(ann_file), f"Annotations not found: {ann_file}"

    # hyperparameters + scheduler defaults
    hp     = cfg.get("hyperparameters", {})
    sched0 = cfg.get("scheduler", {})
    hp["batch_size"]          = int(hp.get("batch_size", 8))
    hp["learning_rate"]       = float(hp.get("learning_rate", 1e-4))
    hp["weight_decay"]        = float(hp.get("weight_decay", 1e-2))
    hp["epochs"]              = int(hp.get("epochs", 50))
    hp["checkpoint_interval"] = int(hp.get("checkpoint_interval", 5))
    warmup_frac = float(sched0.get("warmup_fraction", 0.1))

    # TensorBoard
    tb = None
    if rank == 0 and SummaryWriter:
        tb = SummaryWriter(f"runs/edge_v2_1_1_0_{datetime.now():%Y%m%d_%H%M%S}")

    # data pipeline
    tfm = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    ds      = SegmentationDataset(ann_file, tfm)
    sampler = DistributedSampler(ds) if distributed else None
    loader  = DataLoader(ds,
                         batch_size=hp["batch_size"],
                         sampler=sampler,
                         shuffle=(sampler is None),
                         num_workers=4,
                         pin_memory=True)

    # model / optim / sched / loss
    model = build_model(device)
    if distributed:
        model = DDP(model, device_ids=[rank] if device.type=="cuda" else None)
    opt   = AdamW(model.parameters(),
                  lr=hp["learning_rate"],
                  weight_decay=hp["weight_decay"])
    total_steps = len(loader) * hp["epochs"]
    sched = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=int(warmup_frac * total_steps),
        num_training_steps=total_steps
    )
    crit = CrossEntropyLoss(ignore_index=255)

    # training loop
    for epoch in range(1, hp["epochs"]+1):
        if distributed:
            sampler.set_epoch(epoch)
        model.train()
        sum_loss = 0.0
        sum_miou = 0.0

        for bidx, batch in enumerate(loader):
            imgs  = batch["pixel_values"].to(device, non_blocking=True)
            masks = batch["labels"].to(device, non_blocking=True)

            # forward
            raw = (model(pixel_values=imgs).logits
                   if not distributed else
                   model.module(pixel_values=imgs).logits)
            # upsample to mask size
            logits = F.interpolate(raw, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            loss   = crit(logits, masks)
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()

            # mIoU
            miou = compute_miou(logits.argmax(1), masks)
            sum_loss += loss.item()
            sum_miou += miou.item()

            if rank==0 and bidx%50==0 and tb:
                step = (epoch-1)*len(loader) + bidx
                tb.add_scalar("train/batch_loss", loss.item(), step)
                tb.add_scalar("train/batch_mIoU", miou, step)

        avg_loss = sum_loss / len(loader)
        avg_miou = sum_miou / len(loader)
        if rank == 0:
            print(f"[Epoch {epoch}/{hp['epochs']}] Loss={avg_loss:.4f}, mIoU={avg_miou:.4f}")
            if tb:
                tb.add_scalar("train/epoch_loss", avg_loss, epoch)
                tb.add_scalar("train/epoch_mIoU", avg_miou, epoch)
            if epoch % hp["checkpoint_interval"] == 0:
                os.makedirs("checkpoints", exist_ok=True)
                path = f"checkpoints/edge_v2_1_1_0_epoch{epoch}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state": (model.module.state_dict()
                                    if distributed else model.state_dict()),
                    "optimizer_state": opt.state_dict(),
                    "scheduler_state": sched.state_dict()
                }, path)
                print(f"✔️  Checkpoint saved: {path}")

    # export & quantize (rank 0)
    if rank == 0:
        os.makedirs("inference", exist_ok=True)
        sd = model.module.state_dict() if distributed else model.state_dict()
        torch.save(sd, "inference/vit_7b.pth")

        dummy = torch.randn(1,3,224,224,device=device)
        torch.onnx.export(
            model.module if distributed else model,
            dummy,
            "inference/weights_1_1_0.onnx",
            opset_version=14
        )
        print("✅ weights_1_1_0.onnx exported")

        os.system(
            "python -m onnxruntime.quantization "
            "--input inference/weights_1_1_0.onnx "
            "--output inference/weights_1_1_0_quant.onnx "
            "--mode QLinearOps --per_channel"
        )
        print("✅ weights_1_1_0_quant.onnx generated")

        import tensorrt as trt
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        builder    = trt.Builder(TRT_LOGGER)
        net        = builder.create_network(
                         1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser     = trt.OnnxParser(net, TRT_LOGGER)
        with open("inference/weights_1_1_0_quant.onnx","rb") as f:
            parser.parse(f.read())
        cfg_trt = builder.create_builder_config()
        cfg_trt.max_workspace_size = 1<<30
        cfg_trt.set_flag(trt.BuilderFlag.FP16)
        cfg_trt.set_flag(trt.BuilderFlag.INT8)
        engine = builder.build_engine(net, cfg_trt)
        with open("inference/edge_vision_7b_1_1_0.engine","wb") as f:
            f.write(engine.serialize())
        print("✅ TensorRT engine → inference/edge_vision_7b_1_1_0.engine")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()






