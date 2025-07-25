import json
from pathlib import Path
from tqdm import tqdm

import transformers
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import math

# Silence transformers logging
transformers.logging.set_verbosity_error()

# Load fine-tuned model and tokenizer once
print("Loading model …")
model = AutoModelForCausalLM.from_pretrained("./trainedmodel")
tokenizer = AutoTokenizer.from_pretrained("./trainedmodel")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if model.device.type == "cuda" else -1)
print("Model ready.\n")

# Parameters
DATA_PATH = Path("smoldata.txt")
MAX_NEW_DEFAULT = 128  # fallback cap if prompt already near the context window
OUTPUT_PATH = Path("inference_results.jsonl")
PROB_PATH = Path("probability.jsonl")
MAX_EXAMPLES = None  # Set to an int to limit, or None to process all examples


def build_prompt(text: str) -> str:
    marker = "[ResolvedQuery]"
    idx = text.find(marker)
    if idx == -1:
        print("No [ResolvedQuery] found in text")
        return text
    return text[: idx + len(marker)]


def generate_answer(prompt: str):
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
        raise ValueError("Prompt length exceeds model's context window")
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
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found")

    row_id = 0
    with DATA_PATH.open() as f, OUTPUT_PATH.open("w") as out_f, PROB_PATH.open("w") as prob_f:
        iterable = tqdm(f, desc="Generating", unit="example")
        for idx, line in enumerate(iterable):
            if MAX_EXAMPLES is not None and idx >= MAX_EXAMPLES:
                break
            line = line.strip()
            prompt = build_prompt(line)
            answer, token_details = generate_answer(prompt)
            print("PROMPT:", prompt)
            print("ANSWER:", answer)
            print("TOKEN PROBABILITIES (with top-5 choices):")
            for idx, info in enumerate(token_details, 1):
                tok = info["token"]
                prob = info["prob"]
                print(f"Step {idx}: {tok}\t{prob:.6f}")
                for cand_tok, cand_prob in info["top5"]:
                    print(f"    {cand_tok}\t{cand_prob:.6f}")
            print("=" * 80)

            # Write prompt + answer as a single JSONL entry
            concat_text = f"{prompt} {answer}".strip()
            json.dump({"text": concat_text}, out_f)
            out_f.write("\n")

            # Aggregate probability statistics
            probs = [info["prob"] for info in token_details]
            if probs:
                log_probs = [math.log(p) if p > 0 else -float("inf") for p in probs]
                sum_log = sum(log_probs)
                avg_log_prob = sum_log / len(log_probs)
                prob_product = math.exp(sum_log)
                geo_mean = math.exp(avg_log_prob)
                perplexity = math.exp(-avg_log_prob)
            else:
                avg_log_prob = float("nan")
                prob_product = float("nan")
                geo_mean = float("nan")
                perplexity = float("nan")

            # Save probabilities for each generated token
            prob_entry = {
                "rowid": row_id,
                "prompt": prompt,
                "inference": answer,
                "probabilities": probs,
                "prob_product": prob_product,
                "geo_mean": geo_mean,
                "perplexity": perplexity,
                "avg_log_prob": avg_log_prob,
            }
            json.dump(prob_entry, prob_f)
            prob_f.write("\n")
            row_id += 1


if __name__ == "__main__":
    main()
