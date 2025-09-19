import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sys
import matplotlib.pyplot as plt
import numpy as np

def calculate_bleu(completion, groundtruth):
    completion = completion.lower()
    groundtruth = groundtruth.lower()
    
    reference = groundtruth.split()
    candidate = completion.split()
    
    # Using SmoothingFunction().method1 as a simple smoothing method
    return sentence_bleu([reference], candidate, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)

def main(input_file, output_file=None):
    # Load JSON list from file
    noise = 0
    avg_score = 0
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- This section is unchanged ---
    # Compute BLEU for each element
    for item in data:
        #comp = item.get("completion.", "")
        comp = item.get("inference", "")
        truth = item.get("groundtruth", "")
        bleu = calculate_bleu(comp.lower(), truth.lower())
        item["bleu_score"] = bleu

        if bleu < 0.75:
            noise += 1
        

    # Print results
    for item in data:
        #print(f"ID: {item['id']}, BLEU: {item['bleu_score']:.4f}")
        avg_score += item['bleu_score']
    # --- End of unchanged section ---

    # --------------------------------------------------------------------
    # NEW: Plotting the BLEU scores
    # --------------------------------------------------------------------
    # Extract all bleu scores into a list
    bleu_scores = [item['bleu_score'] for item in data if 'bleu_score' in item and item['bleu_score'] < 0.95]

    if not bleu_scores:
        print("\nNo BLEU scores were calculated, skipping histogram.")
    else:
        # Create a histogram
        plt.figure(figsize=(10, 6)) # Set a good figure size
        
        # Define bins with a width of 0.05
        bins = np.arange(0, 1.05, 0.05) 
        
        plt.hist(bleu_scores, bins=bins, edgecolor='black', alpha=0.7)
        
        # Add titles and labels for clarity
        plt.title('Distribution of BLEU Scores')
        plt.xlabel('BLEU Score')
        plt.ylabel('Frequency')
        plt.xticks(bins) # Set x-ticks to align with bins
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save the plot to a file
        plot_filename = 'bleu_scores_histogram.png'
        plt.savefig(plot_filename)
        print(f"\nHistogram saved to {plot_filename}")

        # Optionally, display the plot
        plt.show()
    # --------------------------------------------------------------------

    # Optionally save results back to a file
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    print("noise " , noise)
    print("avg score ", avg_score / 500)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bleu_score.py <input.json> [output.json]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    main(input_file, output_file)
    
