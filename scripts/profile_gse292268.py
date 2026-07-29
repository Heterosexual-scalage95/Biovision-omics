from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

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

LABEL_TERMS = ("clusters", "assignment", "celltype", "cell_type", "annotation")


def find_cell_type_column(frame: pd.DataFrame) -> str:
    candidates = [
        column
        for column in frame.columns
        if "clusters" in column.lower()
    ]

    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one RNA cluster annotation column; found {candidates}"
        )

    return candidates[0]


def find_niche_column(frame: pd.DataFrame) -> str:
    candidates = [
        column
        for column in frame.columns
        if "spatialclust" in column.lower()
        and "assignments" in column.lower()
    ]

    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one spatial-niche assignment column; found {candidates}"
        )

    return candidates[0]


def harmonize_label(label: object) -> str | None:
    if pd.isna(label):
        return None

    value = str(label)

    if re.fullmatch(r"[a-j]", value):
        return None

    if value.startswith("Hep."):
        return "Hepatocytes"

    replacements = {
        "Stellate.cells": "Stellate cells",
        "Central.venous.LSECs": "Central venous LSECs",
        "Periportal.LSECs": "Periportal LSECs",
        "Portal.endothelial.cells": "Portal endothelial cells",
        "Inflammatory.macrophages": "Inflammatory macrophages",
        "Non.inflammatory.macrophages": "Non-inflammatory macrophages",
        "Cholangiocytes": "Cholangiocytes",
        "CD3..alpha.beta.T.cells": "CD3 alpha-beta T cells",
        "NK.like.cells": "NK-like cells",
        "Mature.B.cells": "Mature B cells",
        "Antibody.secreting.B.cells": "Antibody-secreting B cells",
        "gamma.delta.T.cells.1": "Gamma-delta T cells",
        "gamma.delta.T.cells.2": "Gamma-delta T cells",
        "Erthyroid.cells": "Erythroid cells",
    }

    return replacements.get(value, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing the GSE292268 extracted files.",
    )
    parser.add_argument(
        "--output",
        default="data/manifests/GSE292268_profile",
    )
    parser.add_argument(
        "--minimum-class-size",
        type=int,
        default=200,
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_paths = sorted(input_dir.glob("*metadata_file.csv.gz"))
    if len(metadata_paths) != 2:
        raise ValueError(
            f"Expected two metadata files, found {len(metadata_paths)} in {input_dir}"
        )

    frames = []

    for slide_number, path in enumerate(metadata_paths, start=1):
        frame = pd.read_csv(path)

        cell_type_column = find_cell_type_column(frame)
        niche_column = find_niche_column(frame)

        missing_features = [
            feature for feature in MORPHOLOGY_FEATURES if feature not in frame.columns
        ]
        if missing_features:
            raise ValueError(
                f"{path.name} is missing morphology features: {missing_features}"
            )

        output = frame[
            [
                "cell",
                "cell_ID",
                "fov",
                "CenterX_global_px",
                "CenterY_global_px",
                *MORPHOLOGY_FEATURES,
                cell_type_column,
                niche_column,
            ]
        ].copy()

        output = output.rename(
            columns={
                cell_type_column: "cell_type_original",
                niche_column: "spatial_niche",
            }
        )

        output["slide"] = f"Slide{slide_number}"
        output["source_file"] = path.name
        output["cell_type"] = output["cell_type_original"].map(harmonize_label)
        output["fov_group"] = (
            output["slide"] + "_FOV" + output["fov"].astype(str)
        )

        frames.append(output)

    combined = pd.concat(frames, ignore_index=True)

    class_counts = (
        combined["cell_type"]
        .value_counts(dropna=False)
        .rename_axis("cell_type")
        .reset_index(name="cell_count")
    )

    retained_classes = set(
        class_counts.loc[
            class_counts["cell_count"] >= args.minimum_class_size,
            "cell_type",
        ].dropna()
    )

    modeling = combined[
        combined["cell_type"].isin(retained_classes)
    ].copy()

    modeling.to_csv(
        output_dir / "modeling_metadata.csv.gz",
        index=False,
        compression="gzip",
    )

    class_counts.to_csv(
        output_dir / "cell_type_counts.csv",
        index=False,
    )

    slide_class_counts = (
        modeling.groupby(["slide", "cell_type"])
        .size()
        .reset_index(name="cell_count")
    )

    slide_class_counts.to_csv(
        output_dir / "slide_cell_type_counts.csv",
        index=False,
    )

    summary = {
        "total_cells": len(combined),
        "modeling_cells": len(modeling),
        "slides": int(combined["slide"].nunique()),
        "fov_groups": int(combined["fov_group"].nunique()),
        "retained_classes": sorted(retained_classes),
        "morphology_features": MORPHOLOGY_FEATURES,
        "excluded_unresolved_letter_labels": True,
        "minimum_class_size": args.minimum_class_size,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"\nSaved profile to: {output_dir}")


if __name__ == "__main__":
    main()
