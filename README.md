# fabric-models
Public registry of forkable Fabric AI agents and models — organized by domain, versioned, and royalty-enabled.
Welcome to the **Fabric Model Registry** — the official public directory for all AI agents, inference models, and datasets deployed on the [Fabric 1.0](https://github.com/fab-project-V1) platform.

This repository provides:
- ✅ Open access to forkable, composable `.fab`-ready models
- ✅ Complete version histories and manifest compliance
- ✅ Royalty-tracked lineage via Fabric agent economy
- ✅ Trusted provenance and audit metadata

---

## 📚 Directory Structure

```bash
fabric-models/
├── edge-vision-v2/
│   ├── 1.0.0/
│   │   ├── model.yaml
│   │   └── inference/weights_fp32.onnx
│   └── 1.1.0/
│       ├── model.yaml
│       └── inference/weights_fp32.onnx
├── multilingual-speech/
│   └── 0.1.0/
│       ├── model.yaml
│       └── inference/model.pt
├── model.schema.json
├── README.md
└── CONTRIBUTING.md
Each model version folder contains:

model.yaml: A full Fabric-compliant manifest (see below)

inference/: Pretrained weights or inference artifacts

(Optional): training/, scripts/, tests/, runs/

📖 What Is a Fabric Model?
A Fabric model is a self-contained, forkable unit of intelligence used by .fab agents. Each is governed by a model.yaml manifest, which includes:

🔒 Privacy and policy settings

🧠 Model architecture and performance

📜 Licensing and training provenance

💸 Royalty and usage economics

🧾 Cryptographic signatures and validation

Sample model.yaml:

yaml

name: edge-vision-v2
version: 1.1.0
description: "7B-parameter ViT-based model for semantic segmentation at the edge"
framework: "TensorRT"
input_schema:
  type: object
  properties:
    image:
      type: string
      format: binary
output_schema:
  type: object
  properties:
    segmentation_map:
      type: array
      items: integer
parameter_count: 7000000000
policy:
  privacy: anonymized
  energy_budget: "50J/inference"
finance:
  fee_per_call: 0.001
  margin: 0.80
  fork_royalty_pct: 0.10
currency: "FABRIC"
🛠 Model Format Specification
All models must follow the model.schema.json spec.

Use this command to validate:

bash

npx ajv validate -s model.schema.json -d edge-vision-v2/1.1.0/model.yaml
🧬 Forking and Versioning
Fabric supports royalty-tracked model forking. Every fork:

Must increment the version (1.1.0 → 1.2.0)

Must specify a unique training_commit and training_data_hash

May override performance or policy while inheriting lineage

🧠 Discoverable Categories
To help users browse models, use tags such as:

vision, nlp, quantum, multimodal

7b, 13b, tiny, onnx, trt, pt

edge, cloud, forkable, anonymized

🧾 Licensing
All models must be open-source licensed. Preferred licenses:

Apache-2.0

MIT

CC-BY-4.0 (for datasets or training artifacts)

Include licensing in:

model.yaml

LICENSE.txt (optional but recommended)

🤝 Contributing
Want to publish your own model?

See CONTRIBUTING.md for instructions on:

Forking this repo

Creating compliant model.yaml files

Adding weights and signatures

Submitting a pull request to the registry

🌐 Learn More
📘 Fabric Documentation
🔗 Fabric CLI
🔍 Fabric Explorer (Coming Soon)

This registry powers the next generation of distributed AI agents, fully traceable and programmable using Fabric's open .fab DSL.
🧬 Every model, fork, and run is part of a transparent, royalty-backed AI economy.
