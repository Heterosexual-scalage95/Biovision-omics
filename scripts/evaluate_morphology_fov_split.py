from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def calculate_metrics(model, features, labels) -> dict:
    predictions = model.predict(features)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "weighted_f1": f1_score(labels, predictions, average="weighted"),
        "classification_report": classification_report(
            labels,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "data/manifests/GSE292268_profile/"
            "modeling_metadata.csv.gz"
        ),
    )
    parser.add_argument(
        "--output",
        default="outputs/morphology_fov_split",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path)

    features = frame[MORPHOLOGY_FEATURES]
    labels = frame["cell_type"]
    groups = frame["fov_group"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    train_indices, test_indices = next(
        splitter.split(features, labels, groups)
    )

    x_train = features.iloc[train_indices]
    x_test = features.iloc[test_indices]
    y_train = labels.iloc[train_indices]
    y_test = labels.iloc[test_indices]

    train_groups = set(groups.iloc[train_indices])
    test_groups = set(groups.iloc[test_indices])

    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise RuntimeError(f"FOV leakage detected: {sorted(overlap)}")

    models = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=args.random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=args.random_state,
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=200,
            max_leaf_nodes=31,
            class_weight="balanced",
            random_state=args.random_state,
        ),
    }

    results = {}

    for model_name, model in models.items():
        print(f"\nTraining: {model_name}")

        model.fit(x_train, y_train)
        metrics = calculate_metrics(model, x_test, y_test)
        results[model_name] = metrics

        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(
            "Balanced accuracy: "
            f"{metrics['balanced_accuracy']:.3f}"
        )
        print(f"Macro F1: {metrics['macro_f1']:.3f}")

    summary = {
        "split": "FOV-grouped holdout",
        "total_cells": len(frame),
        "training_cells": len(x_train),
        "test_cells": len(x_test),
        "training_fovs": len(train_groups),
        "test_fovs": len(test_groups),
        "fov_overlap": len(overlap),
        "features": MORPHOLOGY_FEATURES,
        "models": results,
    }

    with (output_dir / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nTraining FOVs: {len(train_groups)}")
    print(f"Test FOVs: {len(test_groups)}")
    print(f"FOV overlap: {len(overlap)}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
