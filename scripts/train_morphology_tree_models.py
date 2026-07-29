from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split

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


def evaluate_model(model, features, labels) -> dict:
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
        default="outputs/morphology_tree_models",
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

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
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

        metrics = evaluate_model(model, x_test, y_test)
        results[model_name] = metrics

        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(
            "Balanced accuracy: "
            f"{metrics['balanced_accuracy']:.3f}"
        )
        print(f"Macro F1: {metrics['macro_f1']:.3f}")

        joblib.dump(
            model,
            output_dir / f"{model_name}.joblib",
        )

    summary = {
        "dataset": str(input_path),
        "total_cells": len(frame),
        "training_cells": len(x_train),
        "test_cells": len(x_test),
        "features": MORPHOLOGY_FEATURES,
        "split": "random stratified cell split",
        "models": results,
    }

    with (output_dir / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
