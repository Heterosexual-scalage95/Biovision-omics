from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="CSV or CSV.GZ containing the required model features.",
    )
    parser.add_argument(
        "--model-directory",
        default="models/final_multimodal_model",
    )
    parser.add_argument(
        "--output",
        default="outputs/predictions/cell_family_predictions.csv.gz",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.60,
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    model_directory = Path(args.model_directory)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = joblib.load(model_directory / "model.joblib")

    metadata = json.loads(
        (model_directory / "model_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    frame = pd.read_csv(input_path)
    required_features = metadata["features"]

    missing_features = [
        feature
        for feature in required_features
        if feature not in frame.columns
    ]

    if missing_features:
        raise ValueError(
            "The input dataset is missing required features: "
            f"{missing_features}"
        )

    feature_table = frame[required_features]

    predictions = model.predict(feature_table)
    probabilities = model.predict_proba(feature_table)

    probability_frame = pd.DataFrame(
        probabilities,
        columns=[
            f"probability_{cell_class}"
            for cell_class in model.classes_
        ],
        index=frame.index,
    )

    maximum_probability = probability_frame.max(axis=1)

    output = frame.copy()
    output["predicted_cell_family_raw"] = predictions
    output["prediction_confidence"] = maximum_probability

    output["predicted_cell_family"] = output[
        "predicted_cell_family_raw"
    ].where(
        maximum_probability >= args.confidence_threshold,
        "Unknown",
    )

    output = pd.concat(
        [output, probability_frame],
        axis=1,
    )

    output.to_csv(
        output_path,
        index=False,
        compression="gzip",
    )

    print("Cells predicted:", len(output))
    print(
        "Confident predictions:",
        int(
            (
                output["predicted_cell_family"]
                != "Unknown"
            ).sum()
        ),
    )
    print(
        "Unknown predictions:",
        int(
            (
                output["predicted_cell_family"]
                == "Unknown"
            ).sum()
        ),
    )
    print("\nPrediction counts:")
    print(
        output["predicted_cell_family"]
        .value_counts()
        .to_string()
    )
    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
