import torch


def calculate_forced_probabilities(model, tokenizer, prompt, forced_text):
    """Calculate token probabilities for the forced section only."""
    # Combine the text and tokenize
    fulltext = prompt + forced_text
    input_ids = tokenizer.encode(fulltext, return_tensors="pt")
    
    # Append EOS token id
    eos_token_id = tokenizer.eos_token_id
    input_ids = torch.cat([input_ids, torch.tensor([[eos_token_id]])], dim=1)
    # Get the token IDs for the forced part of the text
    prompt_token_length = len(tokenizer.encode(prompt))
    forced_token_ids = input_ids[0, prompt_token_length:]

    # 🔹 Print tokens for debugging
    # print("Full input tokens:", tokenizer.convert_ids_to_tokens(input_ids[0]))
    # print("Prompt tokens:", tokenizer.convert_ids_to_tokens(input_ids[0, :prompt_token_length]))
    # print("Forced tokens:", tokenizer.convert_ids_to_tokens(forced_token_ids))

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


def calculate_forced_probabilities_stepwise(model, tokenizer, prompt, forced_text):
    """
    Calculate token probabilities for the forced section one token at a time,
    mimicking the greedy inference process, and return the token along with its probability.
    """
    # Tokenize the forced text without adding special tokens
    forced_token_ids = tokenizer.encode(forced_text, add_special_tokens=False)
    
    # Start with the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    token_details = []
    
    for token_id in forced_token_ids:
        # Perform a single forward pass on the current sequence
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            
        # Get the logits for the *next* token prediction
        next_token_logits = logits[0, -1, :]
        
        # Convert logits to probabilities using softmax
        probs = torch.softmax(next_token_logits, dim=-1)
        
        # Get the probability of the current forced token
        token_prob = probs[token_id].item()
        
        # Decode the token ID back to its string representation
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        
        # Append a dictionary with the token and its probability
        token_details.append({
            "token": token_str,
            "probability": token_prob
        })
        
        # Append the current forced token ID to the input_ids for the next step
        new_token_tensor = torch.tensor([[token_id]], device=input_ids.device)
        input_ids = torch.cat([input_ids, new_token_tensor], dim=1)

    return token_details