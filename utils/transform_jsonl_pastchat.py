import argparse
import json
from pathlib import Path
from typing import Dict

# --- Simple character-based trimming ---------------------------

MAX_PASTCHAT_CHARS = 3000
TRIM_FROM_START = True  # True → drop oldest characters first


def _trim_pastchat(past_chat: str, verbose: bool = False) -> str:
    """Return a shortened `past_chat` string no longer than MAX_PASTCHAT_CHARS."""
    if len(past_chat) <= MAX_PASTCHAT_CHARS:
        return past_chat

    if TRIM_FROM_START:
        trimmed = past_chat[-MAX_PASTCHAT_CHARS:]
    else:
        trimmed = past_chat[:MAX_PASTCHAT_CHARS]

    if verbose:
        print(
            f"Trimming PastChat from {len(past_chat)} -> {len(trimmed)} characters (limit {MAX_PASTCHAT_CHARS})."
        )
    return trimmed


def convert_line(obj: Dict[str, str], verbose: bool = False) -> str:
    course = obj.get("course", obj.get("Course", ""))
    past_chat_raw = obj.get("pastChat", obj.get("PastChat", ""))
    user_query = obj.get("UserQuery", "")
    resolved = obj.get("ResolvedQuery", "")

    past_chat = _trim_pastchat(past_chat_raw, verbose)

    text = (
        f"[Course] {course} "
        f"[PastChat] {past_chat} "
        f"[UserQuery] {user_query} "
        f"[ResolvedQuery] {resolved}"
    )

    if verbose:
        print("Final length (chars):", len(text))
    return text


def process_file(in_path: Path, out_path: Path, verbose: bool = False):
    line_num = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line_num += 1
            if not line.strip():
                continue
            obj = json.loads(line)
            try:
                new_obj = convert_line(obj, verbose)
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue

            # Write raw text without surrounding quotation marks, preserving escapes
            raw_text = new_obj
            dumped = json.dumps(raw_text)  # ensures \n and others are escaped
            fout.write(dumped[1:-1])  # strip the surrounding quotes
            fout.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Transform JSONL adding PastChat and ensuring length <1024 tokens."
    )
    parser.add_argument("input", type=Path, help="Input .jsonl path")
    parser.add_argument("output", type=Path, help="Output .jsonl path")
    parser.add_argument("--verbose", action="store_true", help="Print debug information")
    args = parser.parse_args()

    process_file(args.input, args.output, verbose=args.verbose)


if __name__ == "__main__":
    main() 