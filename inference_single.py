import math
from pathlib import Path

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional

# Silence transformers logging
transformers.logging.set_verbosity_error()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_PATH = Path("./models/trained_model_json")
INPUT_TEXT = "[CourseName]KBAI[UserQuery]what should I study for the final exam[ResolvedQuery]"


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def compute_token_probabilities(model, tokenizer, text: str, skip_until_token: Optional[str] = None):
    """Return list of (token, prob) for each token in `text` (after first).

    If `skip_until_token` is provided, tokens up to **and including** the first
    occurrence of that token are omitted from the returned list.  This mimics
    the training-time behaviour where loss is only computed for tokens that
    come *after* a certain sentinel (e.g. `[ResolvedQuery]`).
    """
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)

    #print tokens and their ids
    print("tokens: ", len(tokenizer.convert_ids_to_tokens(input_ids[0])), tokenizer.convert_ids_to_tokens(input_ids[0]))
    print("ids: ", len(input_ids[0]), input_ids[0])

    
    with torch.no_grad():
        outputs = model(input_ids)

    logits = outputs.logits[0]  # (seq_len, vocab)
    if logits.size(0) < 2:
        raise ValueError("Input text too short to compute probabilities")

    shifted_logits = logits[:-1]          # predict next token
    shifted_labels = input_ids[0, 1:]     # ground-truth next token ids

    # Softmax probabilities for the *entire* distribution so we can grab top-k
    all_probs = torch.softmax(shifted_logits, dim=-1)  # (seq_len-1, vocab)

    # Probability assigned to the actual ground-truth token
    probs = all_probs.gather(1, shifted_labels.unsqueeze(1)).squeeze(1)

    # Top-3 predictions for each position
    topk_vals, topk_indices = torch.topk(all_probs, k=3, dim=-1)  # (seq_len-1, 3)

    token_strings = tokenizer.convert_ids_to_tokens(shifted_labels.tolist())
    topk_token_strings = [tokenizer.convert_ids_to_tokens(row.tolist()) for row in topk_indices]

    tokens_list = token_strings
    probs_list = probs.tolist()
    topk_tokens_list = topk_token_strings
    topk_probs_list = topk_vals.tolist()

    if skip_until_token is not None and skip_until_token in tokens_list:
        # index of the sentinel in the predictions list
        sentinel_idx = tokens_list.index(skip_until_token)
        # drop tokens up to and including the sentinel
        tokens_list = tokens_list[sentinel_idx + 1 :]
        probs_list = probs_list[sentinel_idx + 1 :]
        topk_tokens_list = topk_tokens_list[sentinel_idx + 1 :]
        topk_probs_list = topk_probs_list[sentinel_idx + 1 :]

    return [
        (tok, p, tk, tv)
        for tok, p, tk, tv in zip(
            tokens_list, probs_list, topk_tokens_list, topk_probs_list
        )
    ]


def aggregate_stats(probs):
    log_probs = [math.log(p) if p > 0 else -float("inf") for p in probs]
    sum_log = sum(log_probs)
    avg_log = sum_log / len(log_probs)
    return {
        "prob_product": math.exp(sum_log),
        "geo_mean": math.exp(avg_log),
        "perplexity": math.exp(-avg_log),
        "avg_log_prob": avg_log,
    }


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def main():
    print("Loading model …")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("Model ready.\n")

    text = INPUT_TEXT + tokenizer.eos_token
    # Mimic training masking: ignore tokens up to and including the special
    # `[ResolvedQuery]` marker when computing per-token probabilities.
    token_probs = compute_token_probabilities(
        model,
        tokenizer,
        text,
        skip_until_token="[ResolvedQuery]",
    )

    print("TEXT:", text)
    print("TOKEN PROBABILITIES (conditioning on left context):")
    for idx, (tok, prob, top_tokens, top_probs) in enumerate(token_probs, 1):
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        top_str = ", ".join(
            f"{t} ({p:.4f})" for t, p in zip(top_tokens, top_probs)
        )
        print(
            f"Token {idx}: {tok} (id {tok_id})\t{prob:.6f}\tTop-3: {top_str}"
        )

    probs_only = [p for _, p, *_ in token_probs]
    stats = aggregate_stats(probs_only)
    print("\nAggregate stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("=" * 80)


if __name__ == "__main__":
    main() 