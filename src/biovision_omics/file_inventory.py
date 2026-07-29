import json
from pathlib import Path

import pandas as pd

from .checksums import sha256_file


def classify_file(path: Path) -> list[str]:
    name = path.name.lower()
    groups = {
        "expression": ["expr", "matrix", "count", "gene"],
        "metadata": ["meta", "annotation", "celltype", "cell_type"],
        "spatial": ["coord", "centroid", "spatial", "transcript"],
        "boundary": ["boundary", "polygon", "mask", "segment"],
        "image": ["image", "morphology", "dapi", "tif", "tiff"],
        "sample": ["sample", "patient", "donor", "slide", "fov", "roi"],
    }
    labels = [label for label, words in groups.items() if any(word in name for word in words)]
    if path.suffix.lower() in {".rds", ".rda", ".rdata"}:
        labels.append("r_object")
    return sorted(set(labels))


def preview_table(path: Path):
    try:
        name = path.name.lower()
        if name.endswith(".csv.gz"):
            frame = pd.read_csv(path, nrows=100, compression="gzip", low_memory=False)
        elif name.endswith((".tsv.gz", ".txt.gz")):
            frame = pd.read_csv(path, sep="\t", nrows=100, compression="gzip", low_memory=False)
        elif path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, nrows=100, low_memory=False)
        elif path.suffix.lower() in {".tsv", ".txt"}:
            frame = pd.read_csv(path, sep="\t", nrows=100, low_memory=False)
        else:
            return None, None, None, None
        return len(frame), len(frame.columns), [str(c) for c in frame.columns], None
    except (OSError, ValueError, pd.errors.ParserError) as exc:
    	return None, None, None, f"{type(exc).__name__}: {exc}"


def inventory_directory(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    records = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rows, cols, columns, error = preview_table(path)
        records.append({
            "relative_path": str(path.relative_to(root)),
            "filename": path.name,
            "suffix": "".join(path.suffixes).lower(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "categories": ";".join(classify_file(path)),
            "preview_rows": rows,
            "preview_columns": cols,
            "columns_json": json.dumps(columns) if columns else None,
            "preview_error": error,
        })
    return pd.DataFrame.from_records(records)
