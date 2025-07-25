import torch


def calculate_forced_probabilities(model, tokenizer, prompt, forced_text):
    """Calculate token probabilities for the forced section only."""
    # Combine the text and tokenize
    fulltext = prompt + forced_text
    input_ids = tokenizer.encode(fulltext, return_tensors="pt")

    # Get the token IDs for the forced part of the text
    prompt_token_length = len(tokenizer.encode(prompt))
    forced_token_ids = input_ids[0, prompt_token_length:]

    # Perform a single forward pass
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        
    # Extract the relevant logits for the forced tokens
    # The logits at index i are the predictions for the token at index i+1
    start_index = prompt_token_length - 1
    end_index = input_ids.shape[1] - 1
    forced_logits = logits[0, start_index:end_index, :]

    # Convert logits to probabilities and get the probability of each forced token
    probabilities = torch.softmax(forced_logits, dim=1)

    # Gather the probability of the actual forced token at each step
    forced_token_probs = probabilities.gather(
        1, forced_token_ids.unsqueeze(1)
    ).squeeze(1)

    return forced_token_probs.tolist() 