import argparse
import re
from pathlib import Path

MULTI_SPACE_PATTERN = re.compile(r" {2,}")


def collapse_spaces(line: str) -> str:
    """Replace runs of 2+ spaces with a single space."""
    return MULTI_SPACE_PATTERN.sub(" ", line)


def process_file(path: Path):
    """Collapse spaces in-place for the given file."""
    with path.open() as fin:
        lines = [collapse_spaces(l.rstrip("\n")) for l in fin]

    # Overwrite
    with path.open("w") as fout:
        for l in lines:
            fout.write(l + "\n")


def main():
    parser = argparse.ArgumentParser(description="Collapse consecutive spaces in each line of a text file (in-place).")
    parser.add_argument("file", type=Path, help="Path to the text file to modify (e.g., fat_data.txt)")
    args = parser.parse_args()

    process_file(args.file)


if __name__ == "__main__":
    main() 