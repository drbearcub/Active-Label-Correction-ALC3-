#!/usr/bin/env python3
"""
Active Label Correction (ALC) Pipeline

This script implements an iterative process to improve dataset quality through:
1. Training a model on the current dataset
2. Running inference to get confidence scores
3. Sorting probability results by confidence metrics
4. Auto-correcting the most confident incorrect prediction
5. Creating the corrected dataset for the next iteration
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any
import argparse


class ALCPipeline:
    def __init__(self, iterations: int = 5, initial_data: str = "data/reduced.txt"):
        self.iterations = iterations
        self.initial_data = Path(initial_data)
        self.alc_dir = Path("alcIterations")
        self.models_dir = Path("alcmodels")
        self.current_iteration = 0
        
        # Create directories
        self.alc_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
    def setup_initial_dataset(self) -> Path:
        """Copy the initial dataset to start the ALC process."""
        if not self.initial_data.exists():
            raise FileNotFoundError(f"Initial dataset not found: {self.initial_data}")
        
        initial_copy = self.alc_dir / "iteration_0_dataset.txt"
        shutil.copy2(self.initial_data, initial_copy)
        print(f"✓ Initial dataset copied to {initial_copy}")
        return initial_copy
        
    def run_training(self, train_file: Path, output_model: Path) -> bool:
        """Run the training using command line arguments."""
        print(f"🔄 Training model (iteration {self.current_iteration})...")
        
        # Determine base model path
        if self.current_iteration == 0:
            base_model = 'openai-community/gpt2'
        else:
            prev_model = self.models_dir / f"iteration_{self.current_iteration-1}"
            base_model = str(prev_model)
        
        # Build command arguments for trainer.py
        cmd = [
            sys.executable, "trainer.py",
            "--model_name_or_path", base_model,
            "--train_file", str(train_file),
            "--validation_split_percentage", "5", 
            "--per_device_train_batch_size", "1",
            "--per_device_eval_batch_size", "1",
            "--num_train_epochs", "3",
            "--output_dir", str(output_model),
            "--gradient_accumulation_steps", "10",
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        try:
            # Run without capturing output so we can see logs in real-time
            result = subprocess.run(cmd, check=True)
            print(f"✓ Training completed. Model saved to {output_model}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed with return code: {e.returncode}")
            return False

    def run_inference(self, model_path: Path, data_path: Path, prob_output: Path) -> bool:
        """Run inference using command line arguments."""
        print(f"🔄 Running inference (iteration {self.current_iteration})...")
        
        # Ensure output directory exists
        prob_output.parent.mkdir(parents=True, exist_ok=True)
        
        # Build command arguments for inference.py
        cmd = [
            sys.executable, "inference.py",
            "--model_path", str(model_path),
            "--data_path", str(data_path),
            "--prob_output_path", str(prob_output),
            "--inference_output_path", str(prob_output.parent / f"iteration_{self.current_iteration}_inference_results.jsonl")
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        try:
            # Run without capturing output so we can see logs in real-time
            result = subprocess.run(cmd, check=True)
            print(f"✓ Inference completed. Results saved to {prob_output}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Inference failed with return code: {e.returncode}")
            return False
    
    def load_probability_rows(self, jsonl_path: Path) -> List[Dict[str, Any]]:
        """Load JSONL probability data into a list of dictionaries."""
        rows = []
        with jsonl_path.open() as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    rows.append(obj)
                except json.JSONDecodeError as e:
                    print(f"Warning: skipping malformed JSON at line {line_num}: {e}")
        return rows
    
    def write_sorted_file(self, rows: List[Dict[str, Any]], metric: str, out_path: Path, descending: bool = True):
        """Write rows sorted by metric to a JSONL file."""
        # Filter out rows missing the metric or non-numeric values
        filtered = [r for r in rows if isinstance(r.get(metric), (int, float))]
        sorted_rows = sorted(filtered, key=lambda r: r[metric], reverse=descending)

        with out_path.open("w") as f:
            for obj in sorted_rows:
                json.dump(obj, f)
                f.write("\n")
        print(f"✓ Wrote {len(sorted_rows)} rows sorted by '{metric}' to {out_path}")
    
    def run_sorting(self, prob_output: Path) -> bool:
        """Sort the probability file by confidence metrics."""
        print(f"🔄 Sorting probability file by confidence (iteration {self.current_iteration})...")
        
        if not prob_output.exists():
            print(f"❌ Probability file not found: {prob_output}")
            return False
        
        try:
            # Load probability data
            rows = self.load_probability_rows(prob_output)
            print(f"✓ Loaded {len(rows)} rows from {prob_output}")
            
            # Sort by different metrics
            metrics = ["geo_mean", "forced_geo_mean"]
            out_dir = prob_output.parent
            
            for metric in metrics:
                out_path = out_dir / f"{prob_output.stem}_{metric}_sorted.jsonl"
                self.write_sorted_file(rows, metric, out_path, descending=True)
            
            print(f"✓ Sorting completed. Sorted files created in {out_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Sorting failed: {e}")
            return False
    
    def run_auto_correction(self, prob_output: Path) -> bool:
        """Perform auto-correction by finding the most confident incorrect prediction."""
        print(f"🔄 Running auto-correction (iteration {self.current_iteration})...")
        
        # Load the geo_mean sorted file
        geo_mean_sorted_file = prob_output.parent / f"{prob_output.stem}_geo_mean_sorted.jsonl"
        
        if not geo_mean_sorted_file.exists():
            print(f"❌ Sorted file not found: {geo_mean_sorted_file}")
            return False
        
        try:
            # Load sorted probability data
            rows = self.load_probability_rows(geo_mean_sorted_file)
            print(f"✓ Loaded {len(rows)} sorted rows from {geo_mean_sorted_file}")
            
            # Find rows where inference != completion (incorrect predictions)
            incorrect_rows = [row for row in rows if not row.get('matches_completion', True)]
            
            if not incorrect_rows:
                print("✓ No incorrect predictions found - all inferences match completions!")
                return True
            
            print(f"📊 Found {len(incorrect_rows)} incorrect predictions out of {len(rows)} total")
            
            # ALC Strategy: Only correct 1 row per iteration for gradual, controlled improvement
            # The file is already sorted by geo_mean (descending), so the first incorrect row
            # has the highest confidence among incorrect predictions
            most_confident_incorrect = incorrect_rows[0]
            
            print(f"🎯 Most confident incorrect prediction (correcting only 1 per iteration):")
            print(f"   Row ID: {most_confident_incorrect.get('rowid', 'N/A')}")
            print(f"   Geo Mean: {most_confident_incorrect.get('geo_mean', 'N/A'):.6f}")
            print(f"   Original completion: {most_confident_incorrect.get('completion', '')[:100]}...")
            print(f"   Model inference: {most_confident_incorrect.get('inference', '')[:100]}...")
            
            # Create corrected dataset by updating ONLY the most confident incorrect prediction
            corrected_rows = []
            corrections_made = 0
            target_row_id = most_confident_incorrect.get('rowid')
            
            for row in rows:
                new_row = row.copy()
                
                # Correct only the single most confident incorrect prediction
                if row.get('rowid') == target_row_id and corrections_made == 0:
                    new_row['original_completion'] = row['completion']  # Save original before overwriting
                    new_row['completion'] = row['inference']            # Replace completion with inference
                    new_row['autocorrected'] = True                     # Mark as auto-corrected
                    corrections_made += 1
                    print(f"✓ Corrected row {row.get('rowid', 'N/A')}: completion updated to match inference (original saved)")
                # No else clause - don't add autocorrected: false to unchanged rows
                
                corrected_rows.append(new_row)
            
            # Verify we only corrected exactly 1 row
            assert corrections_made == 1, f"Expected to correct exactly 1 row, but corrected {corrections_made}"
            
            # Save corrected data to new file
            corrected_file = prob_output.parent / f"{prob_output.stem}_autocorrected.jsonl"
            with corrected_file.open("w") as f:
                for row in corrected_rows:
                    json.dump(row, f)
                    f.write("\n")
            
            print(f"✓ Auto-correction completed: {corrections_made} correction(s) made")
            print(f"✓ Corrected data saved to: {corrected_file}")
            return True
            
        except Exception as e:
            print(f"❌ Auto-correction failed: {e}")
            return False
    
    def create_next_dataset(self, prob_output: Path) -> Path:
        """Create the text dataset for the next iteration using autocorrected data."""
        print(f"🔄 Creating dataset for next iteration (iteration {self.current_iteration})...")
        
        # Load the autocorrected data
        corrected_file = prob_output.parent / f"{prob_output.stem}_autocorrected.jsonl"
        
        if not corrected_file.exists():
            print(f"❌ Autocorrected file not found: {corrected_file}")
            return None
        
        try:
            # Load autocorrected probability data
            rows = self.load_probability_rows(corrected_file)
            print(f"✓ Loaded {len(rows)} rows from {corrected_file}")
            
            # Create next iteration dataset file
            next_dataset = self.alc_dir / f"iteration_{self.current_iteration + 1}_dataset.txt"
            
            # Write concatenated prompt + completion for each row
            with next_dataset.open("w") as f:
                for row in rows:
                    prompt = row.get('prompt', '')
                    completion = row.get('completion', '')
                    # Concatenate prompt and completion
                    combined_text = f"{prompt}{completion}"
                    f.write(combined_text + '\n')
            
            print(f"✓ Created dataset for next iteration: {next_dataset}")
            print(f"✓ Dataset contains {len(rows)} examples with autocorrected labels")
            return next_dataset
            
        except Exception as e:
            print(f"❌ Failed to create next dataset: {e}")
            return None
    
    def run_iteration(self, current_dataset: Path) -> Path:
        """Run a single ALC iteration.
        
        This will create:
        - Model: alcmodels/iteration_X/
        - Probabilities: alcIterations/iteration_X_probabilities.jsonl
        - Sorted files: alcIterations/iteration_X_probabilities_geo_mean_sorted.jsonl
                       alcIterations/iteration_X_probabilities_forced_geo_mean_sorted.jsonl
        - Auto-corrected: alcIterations/iteration_X_probabilities_autocorrected.jsonl
        - Next dataset: alcIterations/iteration_{X+1}_dataset.txt
        """
        print(f"\n{'='*50}")
        print(f"Starting ALC Iteration {self.current_iteration}")
        print(f"{'='*50}")
        
        # Paths for this iteration
        model_output = self.models_dir / f"iteration_{self.current_iteration}"
        prob_output = self.alc_dir / f"iteration_{self.current_iteration}_probabilities.jsonl"

        try:
            # Step 1: Run training
            if not self.run_training(current_dataset, model_output):
                return None

            # Step 2: Run inference
            if not self.run_inference(model_output, current_dataset, prob_output):
                return None
            
            # Step 3: Sort probability file by confidence
            if not self.run_sorting(prob_output):
                return None
            
            # Step 4: Auto-correct the most confident incorrect prediction
            if not self.run_auto_correction(prob_output):
                return None
            
            # Step 5: Create the dataset for the next iteration
            next_dataset = self.create_next_dataset(prob_output)
            if next_dataset is None:
                return None
            
            print(f"✅ Iteration {self.current_iteration} completed successfully!")
            return next_dataset  # Return the corrected dataset for next iteration
            
        except Exception as e:
            print(f"❌ Iteration {self.current_iteration} failed: {e}")
            return None
    
    def run_pipeline(self):
        """Run the complete ALC pipeline."""
        print("🚀 Starting Active Label Correction Pipeline")
        print(f"📊 Running {self.iterations} iterations")
        
        # Setup initial dataset
        current_dataset = self.setup_initial_dataset()
        
        print(f"Current dataset is: {current_dataset}")
        # Run iterations (just training for now)
        for i in range(self.iterations):
            self.current_iteration = i
            next_dataset = self.run_iteration(current_dataset)
            
            if next_dataset is None:
                print(f"⚠️  Pipeline stopped at iteration {i} due to failure")
                break
            
            current_dataset = next_dataset
        
        print(f"\n🎉 ALC Pipeline completed!")
        print(f"📁 Results saved in: {self.alc_dir}")
        print(f"   - Dataset files: iteration_X_dataset.txt")
        print(f"   - Probability files: iteration_X_probabilities.jsonl")
        print(f"   - Sorted files: iteration_X_probabilities_geo_mean_sorted.jsonl")
        print(f"   - Auto-corrected files: iteration_X_probabilities_autocorrected.jsonl")
        print(f"🤖 Models saved in: {self.models_dir}")


def main():
    parser = argparse.ArgumentParser(description="Active Label Correction Pipeline")
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=5, 
        help="Number of ALC iterations to run"
    )
    parser.add_argument(
        "--initial_data", 
        type=str, 
        default="data/reduced.txt", 
        help="Path to initial dataset"
    )
    
    args = parser.parse_args()
    
    # Run the pipeline
    pipeline = ALCPipeline(iterations=args.iterations, initial_data=args.initial_data)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()