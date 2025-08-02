#!/usr/bin/env python3
"""
Transform JSONL file to text file by concatenating prompt and completion fields.
"""

import json
import argparse
from pathlib import Path


def transform_jsonl_to_text(input_file: Path, output_file: Path):
    """
    Read JSONL file and create text file with prompt + completion concatenated.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output text file
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    lines_processed = 0
    
    with input_file.open('r') as infile, output_file.open('w') as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                data = json.loads(line)
                
                # Extract prompt and completion
                prompt = data.get('prompt', '')
                completion = data.get('completion', '')
                
                # Concatenate prompt and completion
                combined_text = f"{prompt}{completion}"
                
                # Write to output file
                outfile.write(combined_text + '\n')
                lines_processed += 1
                
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue
    
    print(f"Successfully processed {lines_processed} lines")
    print(f"Output written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Transform JSONL to text file")
    parser.add_argument(
        "input_file", 
        type=Path,
        help="Input JSONL file path"
    )
    parser.add_argument(
        "output_file",
        type=Path, 
        help="Output text file path"
    )
    
    args = parser.parse_args()
    
    transform_jsonl_to_text(args.input_file, args.output_file)


if __name__ == "__main__":
    main()