from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

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


def add_neighbor_features(
    frame: pd.DataFrame,
    neighbors: int,
) -> pd.DataFrame:
    output_frames = []

    for slide, slide_frame in frame.groupby("slide", sort=True):
        slide_frame = slide_frame.copy().reset_index(drop=True)

        coordinates = slide_frame[
            ["CenterX_global_px", "CenterY_global_px"]
        ].to_numpy()

        model = NearestNeighbors(
            n_neighbors=neighbors + 1,
            algorithm="auto",
        )
        model.fit(coordinates)

        distances, indices = model.kneighbors(coordinates)

        # Remove the cell itself, which is always the closest neighbor.
        neighbor_distances = distances[:, 1:]
        neighbor_indices = indices[:, 1:]

        morphology = slide_frame[MORPHOLOGY_FEATURES].to_numpy()

        for feature_index, feature_name in enumerate(MORPHOLOGY_FEATURES):
            neighbor_values = morphology[
                neighbor_indices,
                feature_index,
            ]

            slide_frame[f"neighbor_mean_{feature_name}"] = (
                neighbor_values.mean(axis=1)
            )

        slide_frame["neighbor_mean_distance"] = (
            neighbor_distances.mean(axis=1)
        )

        slide_frame["neighbor_local_density"] = (
            neighbors
            / (
                np.pi
                * np.maximum(
                    neighbor_distances[:, -1],
                    1.0,
                )
                ** 2
            )
        )

        output_frames.append(slide_frame)

        print(
            f"{slide}: {len(slide_frame)} cells processed"
        )

    return pd.concat(output_frames, ignore_index=True)


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
        default=(
            "data/manifests/GSE292268_profile/"
            "modeling_metadata_spatial.csv.gz"
        ),
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path)

    required_columns = [
        "slide",
        "CenterX_global_px",
        "CenterY_global_px",
        *MORPHOLOGY_FEATURES,
    ]

    missing = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    augmented = add_neighbor_features(
        frame,
        neighbors=args.neighbors,
    )

    augmented.to_csv(
        output_path,
        index=False,
        compression="gzip",
    )

    print(f"\nRows saved: {len(augmented)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
