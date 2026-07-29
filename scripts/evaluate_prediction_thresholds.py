from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

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
        default="outputs/prediction_thresholds.csv",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    frame["cell_family"] = frame["cell_type"].map(CELL_FAMILY_MAP)

    rna_features = [
        column for column in frame.columns
        if column.startswith("rna_")
    ]

    features = (
        MORPHOLOGY_FEATURES
        + SPATIAL_FEATURES
        + rna_features
    )

    thresholds = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    rows = []

    slides = sorted(frame["slide"].unique())

    for train_slide, test_slide in [
        (slides[0], slides[1]),
        (slides[1], slides[0]),
    ]:
        train = frame.loc[frame["slide"] == train_slide]
        test = frame.loc[frame["slide"] == test_slide]

        model = HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=200,
            max_leaf_nodes=31,
            class_weight="balanced",
            random_state=42,
        )

        model.fit(
            train[features],
            train["cell_family"],
        )

        predictions = model.predict(test[features])
        probabilities = model.predict_proba(test[features])
        confidence = probabilities.max(axis=1)

        for threshold in thresholds:
            accepted = confidence >= threshold
            accepted_count = int(accepted.sum())

            accuracy = (
                accuracy_score(
                    test.loc[accepted, "cell_family"],
                    predictions[accepted],
                )
                if accepted_count > 0
                else float("nan")
            )

            rows.append(
                {
                    "train_slide": train_slide,
                    "test_slide": test_slide,
                    "threshold": threshold,
                    "accepted_cells": accepted_count,
                    "total_cells": len(test),
                    "coverage": accepted.mean(),
                    "accuracy_among_accepted": accuracy,
                }
            )

    results = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    summary = (
        results.groupby("threshold")
        .agg(
            mean_coverage=("coverage", "mean"),
            mean_accuracy=("accuracy_among_accepted", "mean"),
        )
        .reset_index()
    )

    print(summary.to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
