import math
import json
from pathlib import Path

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from probability_utils import calculate_forced_probabilities

# Silence transformers logging
transformers.logging.set_verbosity_error()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_PATH = Path("./trainedmodelspecialtokens")
DATA_FILE = Path("smallnospace.txt")
OUTPUT_FILE = Path("probability_forced.jsonl")


def aggregate_stats(probs):
    """Calculate aggregate statistics from probabilities."""
    if not probs:
        return {
            "prob_product": 0.0,
            "geo_mean": 0.0,
            "perplexity": float("inf"),
            "avg_log_prob": float("-inf"),
        }
    
    log_probs = [math.log(p) if p > 0 else -float("inf") for p in probs]
    sum_log = sum(log_probs)
    avg_log = sum_log / len(log_probs)
    
    return {
        "prob_product": math.exp(sum_log),
        "geo_mean": math.exp(avg_log),
        "perplexity": math.exp(-avg_log),
        "avg_log_prob": avg_log,
    }


def main():
    print("Loading model …")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("Model ready.\n")

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Read the training data
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    results = []
    
    print(f"Processing {len(lines)} examples...")
    
    for rowid, line in enumerate(lines):
        line = line.strip()
        if not line:
            print(f"Warning: Empty line {rowid}")
            continue
            
        # Split at [ResolvedQuery]
        if "[ResolvedQuery]" not in line:
            print(f"Warning: No [ResolvedQuery] found in line {rowid}")
            continue
            
        # Split the line into prompt and forced sections
        parts = line.split("[ResolvedQuery]", 1)
        prompt = parts[0] + "[ResolvedQuery]"
        forced_text = parts[1]
        
        # Calculate probabilities for the forced section
        try:
            probs = calculate_forced_probabilities(model, tokenizer, prompt, forced_text)
            
            # Calculate aggregate statistics
            stats = aggregate_stats(probs)
            
            # Create result entry
            result = {
                "rowid": rowid,
                "prompt": prompt,
                "inference": forced_text,
                "probabilities": probs,
                **stats
            }
            
            results.append(result)
            
            if (rowid + 1) % 100 == 0:
                print(f"Processed {rowid + 1} examples...")
                
        except Exception as e:
            print(f"Error processing line {rowid}: {e}")
            continue

    # Save results to JSONL file
    print(f"\nSaving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"Completed! Processed {len(results)} examples.")
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main() 