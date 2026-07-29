import argparse
from pathlib import Path
from biovision_omics.file_inventory import inventory_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.input)
    if not source.exists():
        raise FileNotFoundError(source)
    inventory = inventory_directory(source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output, index=False)
    print(f"Files inventoried: {len(inventory):,}")
    print(f"Total bytes: {inventory['size_bytes'].sum():,}")
    print(f"Saved inventory: {output}")


if __name__ == "__main__":
    main()
