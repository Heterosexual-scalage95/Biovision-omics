from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)

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


def evaluate(model, features, labels) -> dict:
    predictions = model.predict(features)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            labels,
            predictions,
        ),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
        ),
        "weighted_f1": f1_score(
            labels,
            predictions,
            average="weighted",
        ),
        "classification_report": classification_report(
            labels,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }


def build_model(random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=200,
        max_leaf_nodes=31,
        class_weight="balanced",
        random_state=random_state,
    )


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
        default="outputs/multimodal_cross_slide",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    frame["cell_family"] = frame["cell_type"].map(
        CELL_FAMILY_MAP
    )

    if frame["cell_family"].isna().any():
        missing = sorted(
            frame.loc[
                frame["cell_family"].isna(),
                "cell_type",
            ].unique()
        )
        raise ValueError(f"Unmapped cell types: {missing}")

    rna_features = [
        column
        for column in frame.columns
        if column.startswith("rna_")
    ]

    feature_sets = {
        "morphology_only": MORPHOLOGY_FEATURES,
        "rna_only": rna_features,
        "morphology_plus_spatial": (
            MORPHOLOGY_FEATURES + SPATIAL_FEATURES
        ),
        "multimodal": (
            MORPHOLOGY_FEATURES
            + SPATIAL_FEATURES
            + rna_features
        ),
    }

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    slides = sorted(frame["slide"].unique())

    if len(slides) != 2:
        raise ValueError(
            f"Expected exactly two slides, found {slides}"
        )

    results = {}

    for train_slide, test_slide in [
        (slides[0], slides[1]),
        (slides[1], slides[0]),
    ]:
        direction = f"{train_slide}_to_{test_slide}"
        results[direction] = {}

        train = frame.loc[frame["slide"] == train_slide]
        test = frame.loc[frame["slide"] == test_slide]

        print("\n" + "=" * 80)
        print(f"Train: {train_slide} | Test: {test_slide}")

        for feature_set_name, columns in feature_sets.items():
            print(f"\nFeature set: {feature_set_name}")
            print(f"Number of features: {len(columns)}")

            model = build_model(args.random_state)

            model.fit(
                train[columns],
                train["cell_family"],
            )

            metrics = evaluate(
                model,
                test[columns],
                test["cell_family"],
            )

            results[direction][feature_set_name] = metrics

            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print(
                "Balanced accuracy: "
                f"{metrics['balanced_accuracy']:.3f}"
            )
            print(f"Macro F1: {metrics['macro_f1']:.3f}")

    averages = {}

    for feature_set_name in feature_sets:
        averages[feature_set_name] = {
            metric: sum(
                results[direction][feature_set_name][metric]
                for direction in results
            )
            / len(results)
            for metric in [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
            ]
        }

    summary = {
        "validation": "bidirectional cross-slide",
        "target": "broad cell family",
        "model": "HistGradientBoostingClassifier",
        "feature_counts": {
            name: len(columns)
            for name, columns in feature_sets.items()
        },
        "directions": results,
        "average_metrics": averages,
    }

    with (output_dir / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)

    print("\n" + "=" * 80)
    print("Average cross-slide results")

    for feature_set_name, metrics in averages.items():
        print(f"\n{feature_set_name}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(
            "Balanced accuracy: "
            f"{metrics['balanced_accuracy']:.3f}"
        )
        print(f"Macro F1: {metrics['macro_f1']:.3f}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
