# How to call this script:
# For JSON file:
# python inference.py \
#     --model_path models/trainedmodelspecialtokens \
#     --data_path data/output.json \
#     --prob_output_path results/probability.jsonl \
#     --inference_output_path results/inference_results.jsonl

import argparse
import json
from pathlib import Path
from tqdm import tqdm
from typing import Union

from probability_utils import calculate_forced_probabilities
import transformers
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import math

# Silence transformers logging
transformers.logging.set_verbosity_error()

# Constants
MAX_NEW_DEFAULT = 128  # fallback cap if prompt already near the context window
MAX_EXAMPLES = None  # Set to an int to limit, or None to process all examples


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on trained model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/trained_model_json",
        help="Path to the trained model directory"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="alcIterations/iteration_0_dataset.json",
        help="Path to the input data file (.json)"
    )
    parser.add_argument(
        "--prob_output_path",
        type=str,
        default="results/probability.jsonl",
        help="Path to save probability results"
    )
    parser.add_argument(
        "--inference_output_path",
        type=str,
        default="results/inference_results.jsonl",
        help="Path to save inference results"
    )

    return parser.parse_args()


def split_prompt_completion(item: Union[str, dict]) -> tuple[str, str]:
    """
    Extracts prompt and completion from a data item.
    The item can be a string (from a .txt/.jsonl file) or a dict (from a .json file).
    """
    # Handle the new JSON object format
    course_name = item.get('course_name', '')
    user_query = item.get('user_query', '')
    # The prompt is formed by concatenating course and query, ending with the marker
    prompt = f"{course_name}{user_query}"
    completion = item.get('completion.', '')
    return prompt, completion

def generate_answer(prompt: str, model, tokenizer):
    """Generate continuation for a single prompt and return per-token probabilities.

    Returns
    -------
    answer : str
        The generated continuation string.
    token_details : List[Dict[str, Union[str, List[Tuple[str, float]]]]
        List of dictionaries containing token information and top-5 choices.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # Ensure we don't exceed the model's context window
    max_allowed_new = model.config.n_positions - input_ids.size(1) - 1
    if max_allowed_new <= 0:
        print(
            f"Warning: Prompt length ({input_ids.size(1)}) exceeds model's context window ({model.config.n_positions}). Skipping generation.")
        return "", []
    max_new = min(MAX_NEW_DEFAULT, max_allowed_new)

    # Generate while also returning the logits for each generated step
    outputs = model.generate(
        input_ids,
        do_sample=False,  # greedy decoding
        max_new_tokens=max_new,
        return_dict_in_generate=True,
        output_scores=True,
    )

    sequences = outputs.sequences[0]  # (prompt_len + new_tokens)
    gen_tokens = sequences[input_ids.size(1):]

    # Compute probabilities for each generated token and capture top-5 choices
    token_details = []  # list of dicts: {token, prob, top5: [(tok, prob), ...]}
    for token_id, score in zip(gen_tokens, outputs.scores):
        probs = torch.softmax(score, dim=-1)[0]
        token_prob = probs[token_id].item()

        top_probs, top_ids = torch.topk(probs, k=5)
        top5 = [
            (tokenizer.convert_ids_to_tokens(int(idx)), prob.item())
            for idx, prob in zip(top_ids, top_probs)
        ]

        token_details.append(
            {
                "token": tokenizer.convert_ids_to_tokens(int(token_id)),
                "prob": token_prob,
                "top5": top5,
            }
        )

    answer = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return answer, token_details


def main():
    args = parse_args()

    # Convert string paths to Path objects
    data_path = Path(args.data_path)
    prob_path = Path(args.prob_output_path)
    output_path = Path(args.inference_output_path)
    model_path = args.model_path

    # Load fine-tuned model and tokenizer
    print("Loading model …")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Model ready.\n")

    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found")

    # Load data based on file extension
    data_source = None
    try:
        with data_path.open('r', encoding='utf-8') as f:
            data_source = json.load(f)

        row_id = 0
        with output_path.open("w") as out_f, prob_path.open("w") as prob_f:
            iterable = tqdm(data_source, desc="Generating", unit="example")
            for idx, item in enumerate(iterable):
                if MAX_EXAMPLES is not None and idx >= MAX_EXAMPLES:
                    break

                # Strip whitespace if the item is a line from a text file
                item_to_process = item.strip() if isinstance(item, str) else item
                if not item_to_process:
                    continue

                prompt, completion = split_prompt_completion(item_to_process)

                try:
                    answer, token_details = generate_answer(prompt, model, tokenizer)
                except ValueError as e:
                    print(f"Error generating answer for item {idx}: {e}")
                    continue

                print("PROMPT:", prompt)
                print("COMPLETION:", completion)
                print("ANSWER:", answer)

                # Check if generated answer matches original completion
                matches_completion = answer == completion
                print("MATCHES COMPLETION:", matches_completion)
                print("=" * 80)

                # Write prompt + answer as a single JSONL entry
                concat_text = f"{prompt} {answer}".strip()
                json.dump({"text": concat_text}, out_f)
                out_f.write("\n")

                # Aggregate probability statistics for inference
                probs = [info["prob"] for info in token_details]
                if probs:
                    log_probs = [math.log(p) if p > 0 else -float("inf") for p in probs]
                    sum_log = sum(log_probs)
                    avg_log_prob = sum_log / len(log_probs)
                    geo_mean = math.exp(avg_log_prob)
                else:
                    avg_log_prob = float("nan")
                    geo_mean = float("nan")

                forced_token_probs = calculate_forced_probabilities(model, tokenizer, prompt, completion)
                if forced_token_probs:
                    log_probs_forced = [math.log(p) if p > 0 else -float("inf") for p in forced_token_probs]
                    sum_log_forced = sum(log_probs_forced)
                    avg_log_prob_forced = sum_log_forced / len(log_probs_forced)
                    geo_mean_forced = math.exp(avg_log_prob_forced)

                # Save probabilities for each generated token
                prob_entry = {
                    "inference": answer,
                    output.json
                    "matches_completion": matches_completion,
                    "geo_mean": geo_mean,
                    "forced_geo_mean": geo_mean_forced,
                    "probabilities": probs,
                    "forced_probabilities": forced_token_probs,
                }
                json.dump(prob_entry, prob_f)
                prob_f.write("\n")
                row_id += 1
    finally:
        # Ensure the file handle for line-based files is closed
        if data_source and hasattr(data_source, 'close'):
            data_source.close()


if __name__ == "__main__":
    main()