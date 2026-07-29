import argparse
import tarfile
from pathlib import Path


def safe_extract(archive: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    root = output.resolve()
    with tarfile.open(archive, "r:*") as handle:
        for member in handle.getmembers():
            target = (output / member.name).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.name}")
        handle.extractall(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    safe_extract(Path(args.archive), Path(args.output))
    print(f"Extracted to: {args.output}")


if __name__ == "__main__":
    main()
