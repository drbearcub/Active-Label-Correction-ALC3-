import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

TOKENIZER_NAME = "openai-community/gpt2"  # match training tokenizer

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)


def validate(path: Path):
    count = 0
    with path.open() as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", "")
            except json.JSONDecodeError:
                print(f"Line {line_num}: JSON decode error")
                continue

            tokens = tokenizer(text, add_special_tokens=False).input_ids
            decoded = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            def _norm(s: str) -> str:
                # remove *all* whitespace characters for comparison
                return "".join(s.split())

            identical = _norm(text) == _norm(decoded)

            status = "MATCH" if identical else "DIFF"
            print(f"Line {line_num}: {len(tokens)} tokens -> {status}")

            if not identical:
                print("  Original:", text[:120] + ("…" if len(text) > 120 else ""))
                print("  Decoded :", decoded[:120] + ("…" if len(decoded) > 120 else ""))
            count += 1
    print(f"\nTotal valid lines: {count}")


def main():
    parser = argparse.ArgumentParser(description="Validate token length of JSONL lines.")
    parser.add_argument("file", type=Path, help="Path to .jsonl file to validate")
    args = parser.parse_args()

    validate(args.file)


if __name__ == "__main__":
    main() 