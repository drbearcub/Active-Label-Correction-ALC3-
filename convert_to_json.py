#!/usr/bin/env python3
"""
Script to convert the txt dataset to JSON format for the new trainer.
"""

import json
import argparse
from pathlib import Path

def convert_txt_to_json(input_file, output_file):
    """
    Convert txt dataset to JSON format.
    
    Input format: [Course]...[UserQuery]...[ResolvedQuery]completion
    Output format: {"prompt": "[Course]...[UserQuery]...[ResolvedQuery]", "completion": "completion"}
    """
    data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            # Find the position of [ResolvedQuery]
            resolved_query_marker = "[ResolvedQuery]"
            marker_pos = line.find(resolved_query_marker)
            
            if marker_pos == -1:
                print(f"Warning: Line {line_num} doesn't contain [ResolvedQuery] marker: {line}")
                continue
            
            # Split into prompt and completion
            prompt = line[:marker_pos + len(resolved_query_marker)]
            completion = line[marker_pos + len(resolved_query_marker):]
            
            data.append({
                "prompt": prompt,
                "completion": completion
            })
    
    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(data)} entries from {input_file} to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert txt dataset to JSON format")
    parser.add_argument("input_file", help="Input txt file")
    parser.add_argument("output_file", help="Output JSON file")
    
    args = parser.parse_args()
    
    convert_txt_to_json(args.input_file, args.output_file)

if __name__ == "__main__":
    main()