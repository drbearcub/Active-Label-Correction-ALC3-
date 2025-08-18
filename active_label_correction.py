#!/usr/bin/env python3
"""
Active Label Correction (ALC) Pipeline

This script implements an iterative process to improve dataset quality through:
1. Training a model on the current JSON dataset (prompt-completion pairs)
2. Running inference to get confidence scores
3. Sorting probability results by confidence metrics
4. Auto-correcting the most confident incorrect prediction
5. Ranking autocorrected data by forced_geo_mean (lowest first)
6. Human annotation using Anthropic API on the 2 lowest confidence rows
7. Creating the corrected JSON dataset for the next iteration

Dataset Format: JSON files with [{"prompt": "...", "completion": "..."}, ...]
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any
import argparse
import shutil

class ALCPipeline:
    def __init__(self, iterations: int = 5, initial_data: str = "alcIterations/iteration_0_dataset.json"):
        self.iterations = iterations
        self.initial_data = Path(initial_data)
        self.alc_dir = Path("alcIterations")
        self.models_dir = Path("alcmodels")
        self.current_iteration = 0
        
        # Create directories
        self.alc_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        Path("results").mkdir(exist_ok=True)  # Ensure results directory exists

    def run_training(self, train_file: Path, output_model: Path) -> bool:
        """Run the training using command line arguments."""
        print(f"🔄 Training model (iteration {self.current_iteration})...")
        
        # Determine base model path
        if self.current_iteration == 0:
            base_model = 'openai-community/gpt2'
        else:
            prev_model = self.models_dir / f"iteration_{self.current_iteration-1}"
            base_model = str(prev_model)
        
        # Build command arguments for trainer_json.py
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

        output = prob_output.parent / f"{prob_output.stem}_step_2_probabilities.json"
        cmd = [
            sys.executable, "inference.py",
            "--model_path", str(model_path),
            "--data_path", str(data_path),
            "--prob_output_path", output,
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        try:
            # Run without capturing output so we can see logs in real-time
            result = subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Inference failed with return code: {e.returncode}")
            return False

    def load_probability_rows(self, json_path: Path) -> List[Dict[str, Any]]:
        """Load probability data from a JSON array file into a list of dictionaries."""
        try:
            with json_path.open(encoding='utf-8') as f:
                # Load the entire JSON array from the file
                data = json.load(f)

            # Ensure the loaded data is a list
            if not isinstance(data, list):
                print(f"Warning: JSON file '{json_path}' does not contain a list (array).")
                return []

            return data

        except json.JSONDecodeError as e:
            print(f"Warning: Failed to decode JSON file '{json_path}': {e}")
            return []
        except FileNotFoundError:
            print(f"Error: File not found at '{json_path}'")
            return []

    def write_sorted_file(self, rows: list, metric: str, out_path: Path, descending: bool = True):
        """Sorts rows by a given metric and writes them to a JSON array file."""

        sorted_rows = sorted(rows, key=lambda r: r.get(metric, 0), reverse=descending)

        try:
            with out_path.open('w', encoding='utf-8') as f:
                # Dump the entire sorted list into the file as a JSON array
                # Using indent=4 makes the output file human-readable
                json.dump(sorted_rows, f, indent=4)
            print(f"✓ Successfully wrote {len(sorted_rows)} rows to {out_path}")

        except IOError as e:
            print(f"Error writing to file {out_path}: {e}")

    def run_sorting(self, prob_output: Path) -> bool:
        """Sort the probability file by confidence metrics."""
        print(f"🔄 Sorting probability file by confidence (iteration {self.current_iteration})...")

        # Load the geo_mean sorted file
        prob_file = prob_output.parent / f"{prob_output.stem}_step_2_probabilities.json"
        if not prob_file.exists():
            print(f"❌ Sorted file not found: {prob_file}")
            return False
        
        try:
            # Load probability data
            rows = self.load_probability_rows(prob_output)
            print(f"✓ Loaded {len(rows)} rows from {prob_output}")
            
            # Sort by different metrics
            metrics = ["geo_mean"]
            out_dir = prob_output.parent
            
            for metric in metrics:
                out_path = out_dir / f"{prob_output.stem}_step_3_geo_mean_sorted.json"
                self.write_sorted_file(rows, "geo_mean", out_path, descending=False)

            print(f"✓ Sorting completed. Sorted files created in {out_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Sorting failed: {e}")
            return False
    
    def run_auto_correction(self, prob_output: Path) -> bool:
        """Perform auto-correction by finding the most confident incorrect prediction."""
        print(f"🔄 Running auto-correction (iteration {self.current_iteration})...")

        print(prob_output.stem)

        # Load the geo_mean sorted file
        geo_mean_sorted_file = prob_output.parent / f"{prob_output.stem}_step_3_geo_mean_sorted.json"
        if not geo_mean_sorted_file.exists():
            print(f"❌ Sorted file not found: {geo_mean_sorted_file}")
            return False
        
        try:
            # Load sorted probability data
            rows = self.load_probability_rows(geo_mean_sorted_file)
            print(f"✓ Loaded {len(rows)} sorted rows from {geo_mean_sorted_file}")
            
            # Find rows where inference != completion (incorrect predictions)
            incorrect_rows = [row for row in rows if not row.get('matches_completion', True) and row.get("human_corrected_iteration") is None]
            
            if not incorrect_rows:
                print("✓ No incorrect predictions found - all inferences match completions!")
                return True
            
            print(f"📊 Found {len(incorrect_rows)} incorrect predictions out of {len(rows)} total")
            
            # ALC Strategy: Only correct 1 row per iteration for gradual, controlled improvement
            # The file is already sorted by geo_mean (descending), so the first incorrect row
            # has the highest confidence among incorrect predictions
            most_confident_incorrect = incorrect_rows[0]
            
            print(f"🎯 Most confident incorrect prediction (correcting only 1 per iteration):")
            print(f"   Row ID: {most_confident_incorrect.get('id', 'N/A')}")
            print(f"   Geo Mean: {most_confident_incorrect.get('geo_mean', 'N/A'):.6f}")
            print(f"   Original completion: {most_confident_incorrect.get('completion.', '')[:100]}...")
            print(f"   Model inference: {most_confident_incorrect.get('inference', '')[:100]}...")
            
            # Create corrected dataset by updating ONLY the most confident incorrect prediction
            corrections_made = 0
            target_row_id = most_confident_incorrect.get('id')
            
            for row in rows:
                # Correct only the single most confident incorrect prediction
                if row.get('id') == target_row_id and corrections_made == 0:
                    row[f'original_completion_{self.current_iteration}'] = row['completion.']  # Save original with iteration
                    row['completion.'] = row['inference']                                        # Replace completion with inference
                    row[f'autocorrected_{self.current_iteration}'] = True                       # Mark as auto-corrected with iteration
                    corrections_made += 1
                    print(f"✓ Corrected row {row.get('id', 'N/A')}: completion updated to match inference (original saved)")
                # No else clause - don't add autocorrected: false to unchanged rows

            # Verify we only corrected exactly 1 row
            assert corrections_made == 1, f"Expected to correct exactly 1 row, but corrected {corrections_made}"

            # Save corrected data to new file
            corrected_file = prob_output.parent / f"{prob_output.stem}_step_4_autocorrected.json"
            try:
                with corrected_file.open('w', encoding='utf-8') as f:
                    # Dump the entire sorted list into the file as a JSON array
                    # Using indent=4 makes the output file human-readable
                    json.dump(rows, f, indent=4)

            except IOError as e:
                print(f"Error writing to file {corrected_file}: {e}")

            print(f"✓ Auto-correction completed: {corrections_made} correction(s) made")
            print(f"✓ Corrected data saved to: {corrected_file}")
            return True
            
        except Exception as e:
            print(f"❌ Auto-correction failed: {e}")
            return False
    
    def create_next_dataset(self, prob_output: Path) -> Path:
        """Create the text dataset for the next iteration using human annotated data."""
        print(f"🔄 Creating dataset for next iteration (iteration {self.current_iteration})...")
        
        # Always expect human annotated data to exist
        human_annotated_file = prob_output.parent / f"{prob_output.stem}_step_5_human_annotated.json"
        
        if not human_annotated_file.exists():
            raise FileNotFoundError(f"Human annotated file not found: {human_annotated_file}")
        
        corrected_file = human_annotated_file
        print(f"✓ Using human annotated data: {corrected_file}")
        next_dataset = self.alc_dir / f"iteration_{self.current_iteration + 1}_dataset.json"
        shutil.copy(human_annotated_file, next_dataset)
        return next_dataset

    def run_human_annotation(self, prob_output: Path) -> bool:
        """Perform human annotation using Anthropic API on the 2 lowest forced_geo_mean rows."""
        print(f"🔄 Running human annotation with Anthropic API (iteration {self.current_iteration})...")
        
        # Load the ranked file (sorted by forced_geo_mean, lowest first)
        ranked_file = prob_output.parent / f"{prob_output.stem}_step_4_autocorrected.json"
        
        if not ranked_file.exists():
            print(f"❌ Ranked file not found: {ranked_file}")
            return False
        
        try:
            # Load ranked data
            rows = self.load_probability_rows(ranked_file)
            print(f"✓ Loaded {len(rows)} rows from {ranked_file}")
            
            if len(rows) < 2:
                print("⚠️  Warning: Less than 2 rows available for human annotation")
                return True

            rows_examined = 0
            rows_corrected = 0
            row_id_to_remove = ''
            should_remove = False

            for index, row in enumerate(rows):
                if row.get(f"autocorrected_{self.current_iteration}", False):
                    print("skipping row as it is human annotated")
                    continue

                # Get rid of remove logic for now
                # if rows_examined == 5:
                #     if should_remove:
                #         row_id_to_remove = row["id"]
                #     print(rows_corrected, "out of ", rows_examined, "are human annotated, removed", row_id_to_remove)
                #     break

                rows_examined += 1

                # Get correction from Anthropic (pass full prompt, get back corrected UserQuery)
                corrected_user_query = row.get('human_annotation')
                print("corrected_user_query: ", corrected_user_query)
                # Extract the original user query to compare

                if corrected_user_query != row.get('completion.', ''):
                    # Save original and update fields with iteration number
                    row[f'original_completion_{self.current_iteration}'] = row.get('completion.', '')
                    row['completion.'] = corrected_user_query
                    row[f'human_corrected_iteration'] = self.current_iteration
                    rows_corrected += 1
                    print(f"✓ Human corrected row {row.get('id', 'N/A')}")
                    should_remove = True

            # Save human annotated data to new file
            annotated_file = prob_output.parent / f"{prob_output.stem}_step_5_human_annotated.json"
            final_rows = [row for row in rows if row.get('id') != row_id_to_remove]

            try:
                with annotated_file.open('w', encoding='utf-8') as f:
                    # Dump the entire sorted list into the file as a JSON array
                    # Using indent=4 makes the output file human-readable
                    json.dump(final_rows, f, indent=4)

            except IOError as e:
                print(f"Error writing to file {annotated_file}: {e}")

            print(f"✓ Human annotated data saved to: {annotated_file}")
            return True
            
        except Exception as e:
            print(f"❌ Human annotation failed: {e}")
            return False
    
    def run_iteration(self, current_dataset: Path) -> Path:
        """Run a single ALC iteration.
        
        This will create:
        - Model: alcmodels/iteration_X/
        - Probabilities: alcIterations/iteration_X_probabilities.json
        - Sorted files: alcIterations/iteration_X_probabilities_geo_mean_sorted.json
                       alcIterations/iteration_X_probabilities_forced_geo_mean_sorted.json
        - Auto-corrected: alcIterations/iteration_X_probabilities_autocorrected.json
        - Ranked by forced_geo_mean: alcIterations/iteration_X_probabilities_ranked_by_forced_geo_mean.json
        - Human annotated: alcIterations/iteration_X_probabilities_human_annotated.json
        - Next dataset: alcIterations/iteration_{X+1}_dataset.json
        """
        print(f"\n{'='*50}")
        print(f"Starting ALC Iteration {self.current_iteration}")
        print(f"{'='*50}")
        
        # Paths for this iteration
        model_output = self.models_dir / f"iteration_{self.current_iteration}"
        prob_output = self.alc_dir / f"iteration_{self.current_iteration}.json"

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

            # Step 5: Human annotation
            if not self.run_human_annotation(prob_output):
                return None

            # Step 6: Create the dataset for the next iteration
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
        current_dataset = "alcIterations/iteration_0_dataset.json"
        

        for i in range(self.iterations):
            print(f"Current dataset is: {current_dataset}")
            self.current_iteration = i
            next_dataset = self.run_iteration(current_dataset)
            
            if next_dataset is None:
                print(f"⚠️  Pipeline stopped at iteration {i} due to failure")
                break
            
            current_dataset = next_dataset
            print(current_dataset)
        
        print(f"\n🎉 ALC Pipeline completed!")
        print(f"📁 Results saved in: {self.alc_dir}")
        print(f"   - Dataset files: iteration_X_dataset.json")
        print(f"   - Probability files: iteration_X_probabilities.json")
        print(f"   - Sorted files: iteration_X_probabilities_geo_mean_sorted.json")
        print(f"   - Auto-corrected files: iteration_X_probabilities_autocorrected.json")
        print(f"   - Human annotated files: iteration_X_probabilities_human_annotated.json")
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
        default="alcIterations/iteration_0_dataset.json", 
        help="Path to initial dataset"
    )
    
    args = parser.parse_args()
    
    # Run the pipeline
    pipeline = ALCPipeline(iterations=args.iterations, initial_data=args.initial_data)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()