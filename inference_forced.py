import math
from pathlib import Path

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional

from probability_utils import calculate_forced_probabilities

# Silence transformers logging
transformers.logging.set_verbosity_error()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_PATH = Path("./models/trained_model_json")
PROMPT = "[Course]Eng_srtc[UserQuery]what is the steps of the research process[ResolvedQuery]"
FORCED = "what is the steps of the research process"


def main():
    print("Loading model …")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("Model ready.\n")

    # Ensure pad token is set for generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    prompt_ids = tokenizer.encode(PROMPT, return_tensors="pt")
    prompt_token_length = prompt_ids.shape[1]

    # -----------------------------------------------------------------------------
    # 🚀 1. Standard Inference with Token Probabilities
    # -----------------------------------------------------------------------------
    print("Performing standard inference and calculating token probabilities...")
    # Generate output, asking the model to return the scores (logits)
    outputs = model.generate(
        prompt_ids,
        max_new_tokens=50,
        do_sample=False,  # Use greedy decoding
        output_scores=True, # 💡 Instruct the model to return logits
        return_dict_in_generate=True # 💡 Return a dictionary with all outputs
    )

    # The generated sequence including the prompt
    generated_ids = outputs.sequences[0]
    # The scores (logits) for each generated token
    generated_scores = outputs.scores

    # Decode the full generated text
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    print("-" * 60)
    print("Model Generated Output:")
    print("["+generated_text+"]")
    print("\nToken-by-Token Probabilities (from Inference):")
    print("-" * 45)
    print(f"{'Token':<25} | {'Probability':<20}")
    print("-" * 45)
    
    # Calculate and print probability for each new token
    # The scores are the logits for each step, so we apply softmax
    all_probs = [torch.softmax(logits, dim=-1) for logits in generated_scores]

    for i, token_id in enumerate(generated_ids[prompt_token_length:]):
        # Get the probability of the generated token at this step
        prob = all_probs[i][0, token_id].item()
        token_str = tokenizer.decode(token_id)
        print(f"{repr(token_str):<25} | {prob:.4f}")
    
    print("-" * 45)
    print("\n")

    # -----------------------------------------------------------------------------
    # 🚀 2. Forced Token Probabilities using utility function
    # -----------------------------------------------------------------------------
    print("Calculating forced token probabilities...")
    
    # Use the utility function to calculate probabilities for the forced section
    forced_token_probs = calculate_forced_probabilities(model, tokenizer, PROMPT, FORCED)
    
    # Get the token IDs for display purposes
    fulltext = PROMPT + FORCED
    input_ids = tokenizer.encode(fulltext, return_tensors="pt")
    prompt_token_length = len(tokenizer.encode(PROMPT))
    forced_token_ids = input_ids[0, prompt_token_length:]

    print("input_ids: ", input_ids)
    print("input_ids shape: ", input_ids.shape)
    print("input_ids type: ", type(input_ids))
    print("input_ids dtype: ", input_ids.dtype)
    print("input_ids device: ", input_ids.device)
    print("input_ids requires_grad: ", input_ids.requires_grad)

    # Display the results
    print(f"Prompt: '{PROMPT}'")
    print(f"Forced sequence: '{FORCED}'\n")
    print("Token-by-Token Probabilities (Forced):")
    print("-" * 40)
    print(f"{'Token':<20} | {'Probability':<20}")
    print("-" * 40)

    for i, token_id in enumerate(forced_token_ids):
        token_str = tokenizer.decode(token_id)
        prob = forced_token_probs[i]
        print(f"{repr(token_str):<25} | {prob:.4f}")

    print("-" * 40)


if __name__ == "__main__":
    main() 