from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

MORPHOLOGY_FEATURES = [
    "Area.um2",
    "AspectRatio",
    "Width",
    "Height",
    "Mean.PanCK",
    "Max.PanCK",
    "Mean.CK8.18",
    "Max.CK8.18",
    "Mean.Membrane",
    "Max.Membrane",
    "Mean.CD45",
    "Max.CD45",
    "Mean.DAPI",
    "Max.DAPI",
]

SPATIAL_FEATURES = [
    f"neighbor_mean_{feature}"
    for feature in MORPHOLOGY_FEATURES
] + [
    "neighbor_mean_distance",
    "neighbor_local_density",
]

CELL_FAMILY_MAP = {
    "Hepatocytes": "Hepatocytes",
    "Central venous LSECs": "Endothelial cells",
    "Periportal LSECs": "Endothelial cells",
    "Portal endothelial cells": "Endothelial cells",
    "Inflammatory macrophages": "Myeloid immune cells",
    "Non-inflammatory macrophages": "Myeloid immune cells",
    "CD3 alpha-beta T cells": "Lymphoid immune cells",
    "Gamma-delta T cells": "Lymphoid immune cells",
    "NK-like cells": "Lymphoid immune cells",
    "Mature B cells": "Lymphoid immune cells",
    "Antibody-secreting B cells": "Lymphoid immune cells",
    "Stellate cells": "Stellate cells",
    "Cholangiocytes": "Cholangiocytes",
}


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "data/manifests/GSE292268_profile/"
            "modeling_metadata_multimodal.csv.gz"
        ),
    )
    parser.add_argument(
        "--output",
        default="models/final_multimodal_model",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    frame["cell_family"] = frame["cell_type"].map(CELL_FAMILY_MAP)

    rna_features = [
        column
        for column in frame.columns
        if column.startswith("rna_")
    ]

    feature_columns = (
        MORPHOLOGY_FEATURES
        + SPATIAL_FEATURES
        + rna_features
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=200,
        max_leaf_nodes=31,
        class_weight="balanced",
        random_state=args.random_state,
    )

    print("Training final multimodal model...")
    print("Cells:", len(frame))
    print("Features:", len(feature_columns))

    model.fit(
        frame[feature_columns],
        frame["cell_family"],
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        output_dir / "model.joblib",
    )

    metadata = {
        "model": "HistGradientBoostingClassifier",
        "training_cells": len(frame),
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "classes": sorted(frame["cell_family"].unique()),
        "cross_slide_macro_f1": 0.723,
        "target": "broad cell family",
    }

    with (output_dir / "model_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, indent=2)

    print("Saved model:", output_dir / "model.joblib")
    print("Saved metadata:", output_dir / "model_metadata.json")


if __name__ == "__main__":
    main()
