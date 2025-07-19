# 🤝 Contributing to `fabric-models`

Thank you for your interest in contributing to the **Fabric Models Registry** — the official open repository for deploying and discovering forkable `.fab` models used across the Fabric ecosystem.

This document provides the guidelines and required steps to successfully submit a model or fork to this repository.

---

## 🧩 What Can I Contribute?

You can contribute:

- ✅ New foundational models (e.g. vision, NLP, audio, quantum)
- ✅ Forked or improved variants of existing models
- ✅ Datasets used to train Fabric agents (with manifests)
- ✅ Bug fixes to model manifests, structure, or metadata
- ✅ Performance benchmarks, training configs, and test cases

---

## 📁 Repository Structure

Organize your model like this:

```bash
fabric-models/
└── model-name/
    ├── 1.0.0/
    │   ├── model.yaml                  # Required manifest
    │   ├── inference/                  # Required weights
    │   │   └── weights_fp32.onnx
    │   ├── training/                   # Optional: configs, annotations
    │   ├── scripts/                    # Optional: train/infer scripts
    │   ├── tests/                      # Optional: test inference
    │   └── runs/                       # Optional: experiment logs
📝 Model Manifest Requirements
Each submission must include a model.yaml manifest conforming to model.schema.json.

Key required fields:

yaml

name: your-model-name
version: 1.0.0
description: "Short description"
framework: "ONNX"  # or TensorRT, PyTorch, etc.
parameter_count: 1234567890

input_schema: {...}
output_schema: {...}

performance:
  metrics:
    - name: accuracy
      value: 0.92
      dataset: COCO-2017

policy:
  privacy: anonymized
  energy_budget: "50J/inference"

finance:
  fee_per_call: 0.001
  margin: 0.80
  fork_royalty_pct: 0.10
  currency: "FABRIC"

signatures:
  artifact_sig: "<sha256_hex>"
  manifest_sig: "<sha256_hex>"
Use the validator before submitting:

bash

npx ajv validate -s model.schema.json -d path/to/model.yaml
💸 Royalty and Fork Logic
When forking a model:

Create a new version folder (e.g., 1.1.0/)

Update training_commit and training_data_hash

Adjust fork_royalty_pct if applicable

Always preserve original credit in provenance

📜 License and Attribution
All submissions must include:

license field in model.yaml

Permissible licenses: Apache-2.0, MIT, CC-BY-4.0, etc.

Attribution to original authors (if forked or fine-tuned)

✅ Submission Steps
Fork this repository

Add your model folder to the correct directory

Run validation on model.yaml

Commit your changes

Open a Pull Request

Please use the following PR format:

markdown

### 📌 Model Submission: edge-vision-v2 1.1.0

**Type:** New version (fork of 1.0.0)  
**Description:** Enhanced segmentation with 7B ViT + improved masks  
**License:** Apache-2.0  
**Tags:** vision, vit, 7b, quantized, forked  
**Manifest:** ✅ Validated  
📬 Need Help?
If you're unsure how to format your model or need assistance with .fab integration:

Open an issue here

Join the Fabric community via Discord (coming soon)

Let’s build a thriving, forkable, royalty-backed AI registry for the world.

Every agent. Every block. Every model — on Fabric.
