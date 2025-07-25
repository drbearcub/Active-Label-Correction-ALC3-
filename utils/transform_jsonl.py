import argparse
import json
from pathlib import Path


def convert_line(obj: dict) -> str:
    """Return the new JSON object with the requested 'text' field."""
    course = obj.get('course', '').rstrip(' \n')
    user_query = obj.get('UserQuery', '').rstrip(' \n')
    resolved_query = obj.get('ResolvedQuery', '').rstrip(' \n')
    text = (
        f"[Course]{course}"
        f"[UserQuery]{user_query}"
        f"[ResolvedQuery]{resolved_query}"
    )
    return text


def process_file(in_path: Path, out_path: Path):
    """Read `in_path` JSONL and write transformed JSONL to `out_path`."""
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue  # skip empty lines
            obj = json.loads(line)
            raw_text = convert_line(obj)
            # Use json.dumps to preserve escape sequences (e.g., \n) then strip outer quotes.
            dumped = json.dumps(raw_text)
            fout.write(dumped[1:-1])  # remove the surrounding quotation marks
            fout.write("\n")


if __name__ == "__main__":
    input_path = Path("data.jsonl")
    output_path = Path("smallnospace.txt")
    process_file(input_path, output_path) 