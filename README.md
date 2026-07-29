# BioVision-Omics

Production-grade multimodal deep learning for biological imaging with transcriptomic interpretation.

## Current milestone: dataset acquisition and audit

This repository intentionally starts with **data provenance and validation**, not model training.

First public dataset:

- GEO accession: `GSE292268`
- Biological context: human MASLD liver fibrosis
- Platform: NanoString CosMx Spatial Molecular Imager
- Planned use: inspect available spatial expression, cell metadata, coordinates, morphology-related outputs, and labels before defining any supervised ML task

## Scientific rule

No target labels, image channels, segmentation masks, or file formats are assumed. The first model will be defined only after the audit confirms what the public files actually contain.

## Setup in Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Test the download URL without downloading

```bash
python scripts/download_geo.py --accession GSE292268 --dry-run
```

## Download

```bash
python scripts/download_geo.py --accession GSE292268
```

Outputs:

```text
data/raw/GSE292268/GSE292268_RAW.tar
data/manifests/GSE292268_download.json
```

## Extract

```bash
python scripts/extract_archive.py --archive data/raw/GSE292268/GSE292268_RAW.tar --output data/raw/GSE292268/extracted
```

## Audit

```bash
python scripts/audit_dataset.py --input data/raw/GSE292268/extracted --output data/manifests/GSE292268_inventory.csv
```

## Tests

```bash
pytest -q
```
