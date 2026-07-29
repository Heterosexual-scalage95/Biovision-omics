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


def build_models(random_state: int) -> dict:
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=200,
            max_leaf_nodes=31,
            class_weight="balanced",
            random_state=random_state,
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
        default="outputs/morphology_cross_slide",
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
    frame["cell_family"] = frame["cell_type"].map(CELL_FAMILY_MAP)

    if frame["cell_family"].isna().any():
        missing = sorted(
            frame.loc[frame["cell_family"].isna(), "cell_type"].unique()
        )
        raise ValueError(f"Unmapped cell types: {missing}")

    slide_names = sorted(frame["slide"].unique())

    if len(slide_names) != 2:
        raise ValueError(
            f"Expected exactly two slides, found: {slide_names}"
        )

    results = {}

    for train_slide, test_slide in [
        (slide_names[0], slide_names[1]),
        (slide_names[1], slide_names[0]),
    ]:
        print("\n" + "=" * 80)
        print(f"Train: {train_slide} | Test: {test_slide}")

        train_frame = frame.loc[frame["slide"] == train_slide]
        test_frame = frame.loc[frame["slide"] == test_slide]

        x_train = train_frame[MORPHOLOGY_FEATURES]
        y_train = train_frame["cell_family"]
        x_test = test_frame[MORPHOLOGY_FEATURES]
        y_test = test_frame["cell_family"]

        direction = f"{train_slide}_to_{test_slide}"
        results[direction] = {
            "training_cells": len(train_frame),
            "test_cells": len(test_frame),
            "models": {},
        }

        for model_name, model in build_models(args.random_state).items():
            print(f"\nTraining: {model_name}")

            model.fit(x_train, y_train)
            metrics = evaluate(model, x_test, y_test)

            results[direction]["models"][model_name] = metrics

            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print(
                "Balanced accuracy: "
                f"{metrics['balanced_accuracy']:.3f}"
            )
            print(f"Macro F1: {metrics['macro_f1']:.3f}")

    model_names = list(
        next(iter(results.values()))["models"].keys()
    )

    averages = {}

    for model_name in model_names:
        averages[model_name] = {
            metric: sum(
                direction["models"][model_name][metric]
                for direction in results.values()
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
        "target": "broad cell family",
        "validation": "bidirectional cross-slide",
        "features": MORPHOLOGY_FEATURES,
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

    for model_name, metrics in averages.items():
        print(f"\n{model_name}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(
            "Balanced accuracy: "
            f"{metrics['balanced_accuracy']:.3f}"
        )
        print(f"Macro F1: {metrics['macro_f1']:.3f}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
