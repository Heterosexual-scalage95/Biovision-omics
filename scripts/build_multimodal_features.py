from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MARKER_GENES = [
    # Hepatocytes
    "APOA1",
    "TTR",
    "GC",
    "ARG1",
    # Cholangiocytes and epithelial cells
    "EPCAM",
    "KRT7",
    "KRT8",
    "KRT18",
    "KRT19",
    # Endothelial cells
    "PECAM1",
    "VWF",
    "KDR",
    "ENG",
    "KDR",
    "LYVE1",
    # Stellate and stromal cells
    "COL1A1",
    "COL1A2",
    "COL3A1",
    "DCN",
    "LUM",
    "ACTA2",
    "PDGFRA",
    "PDGFRB",
    # Myeloid immune cells
    "LYZ",
    "CD68",
    "CD163",
    "C1QA",
    "C1QB",
    "C1QC",
    "MRC1",
    "SPP1",
    # T and NK cells
    "CD3D",
    "CD3E",
    "CD3G",
    "CD4",
    "CD8A",
    "NKG7",
    "GNLY",
    # B and antibody-secreting cells
    "MS4A1",
    "CD79A",
    "MZB1",
    "JCHAIN",
]


def load_expression(
    path: Path,
    slide_name: str,
    marker_genes: list[str],
) -> pd.DataFrame:
    header = pd.read_csv(path, compression="gzip", nrows=0)

    available_genes = [
        gene for gene in marker_genes if gene in header.columns
    ]
    missing_genes = [
        gene for gene in marker_genes if gene not in header.columns
    ]

    print(f"\n{slide_name}")
    print(f"Available marker genes: {len(available_genes)}")
    print(f"Missing marker genes: {missing_genes}")

    columns = ["fov", "cell_ID", *available_genes]

    expression = pd.read_csv(
        path,
        compression="gzip",
        usecols=columns,
    )

    expression["slide"] = slide_name

    gene_counts = expression[available_genes].astype(float)
    library_size = gene_counts.sum(axis=1).replace(0, 1)

    normalized = gene_counts.div(library_size, axis=0) * 10_000
    normalized = np.log1p(normalized)

    normalized.columns = [
        f"rna_{gene}" for gene in normalized.columns
    ]

    return pd.concat(
        [
            expression[["slide", "fov", "cell_ID"]],
            normalized,
        ],
        axis=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metadata",
        default=(
            "data/manifests/GSE292268_profile/"
            "modeling_metadata_spatial.csv.gz"
        ),
    )
    parser.add_argument(
        "--expression-directory",
        required=True,
    )
    parser.add_argument(
        "--output",
        default=(
            "data/manifests/GSE292268_profile/"
            "modeling_metadata_multimodal.csv.gz"
        ),
    )

    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    expression_directory = Path(args.expression_directory)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path)

    marker_genes = list(dict.fromkeys(MARKER_GENES))

    expression_paths = sorted(
        expression_directory.glob("*exprMat_file.csv.gz")
    )

    if len(expression_paths) != 2:
        raise ValueError(
            "Expected two expression matrices, "
            f"but found {len(expression_paths)}."
        )

    expression_frames = []

    for slide_number, expression_path in enumerate(
        expression_paths,
        start=1,
    ):
        slide_name = f"Slide{slide_number}"

        expression_frames.append(
            load_expression(
                expression_path,
                slide_name,
                marker_genes,
            )
        )

    expression = pd.concat(
        expression_frames,
        ignore_index=True,
    )

    multimodal = metadata.merge(
        expression,
        on=["slide", "fov", "cell_ID"],
        how="left",
        validate="one_to_one",
    )

    rna_columns = [
        column
        for column in multimodal.columns
        if column.startswith("rna_")
    ]

    cells_without_rna = int(
        multimodal[rna_columns].isna().all(axis=1).sum()
    )

    if cells_without_rna:
        raise ValueError(
            f"{cells_without_rna} cells did not match expression data."
        )

    multimodal.to_csv(
        output_path,
        index=False,
        compression="gzip",
    )

    print("\nMultimodal dataset complete")
    print("Rows:", len(multimodal))
    print("RNA features:", len(rna_columns))
    print("Unmatched cells:", cells_without_rna)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
