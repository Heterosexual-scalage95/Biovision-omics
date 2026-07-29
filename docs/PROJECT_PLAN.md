# Project plan

## Phase 0 — Provenance and audit

- Download through a versioned script
- Record URL, date, HTTP metadata, size, and SHA-256
- Extract safely
- Inventory every file
- Inspect schemas, biological annotations, and imaging assets

## Phase 1 — Dataset-specific loader

The loader will be written only after the audit identifies the actual schemas.

## Phase 2 — Valid task definition

Potential tasks, subject to real labels:

- cell-type prediction from morphology
- HSC-state classification
- fibrosis-associated niche classification
- multimodal image and expression fusion

## Phase 3 — Baselines and leakage audit

- group-aware splitting by patient, slide, or FOV
- handcrafted morphology baseline
- transcriptomic baseline
- ResNet and ViT only after labels and independent units are confirmed
