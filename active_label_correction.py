#!/usr/bin/env python3
"""
Active Label Correction (ALC) Pipeline

This script implements an iterative process to improve dataset quality through:
1. Training a model on the current dataset
2. Running inference to get confidence scores
3. Sorting probability results by confidence metrics
4. Auto-correcting the most confident incorrect prediction
5. Ranking autocorrected data by forced_geo_mean (lowest first)
6. Human annotation using Anthropic API on the 2 lowest confidence rows
7. Creating the corrected dataset for the next iteration

API Key Configuration:
- Option 1: Create secrets.json file with {"ANTHROPIC_API_KEY": "your_key_here"}
- Option 2: Set environment variable ANTHROPIC_API_KEY=your_key_here
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any
import argparse

try:
    import anthropic
except ImportError:
    print("⚠️  Warning: anthropic package not installed. Install with: pip install anthropic")
    anthropic = None


class ALCPipeline:
    def __init__(self, iterations: int = 5, initial_data: str = "data/reduced.txt"):
        self.iterations = iterations
        self.initial_data = Path(initial_data)
        self.alc_dir = Path("alcIterations")
        self.models_dir = Path("alcmodels")
        self.current_iteration = 0
        
        # Read Anthropic API key from secret file or environment
        self.anthropic_api_key = self.load_api_key()
        if not self.anthropic_api_key:
            print("⚠️  Warning: ANTHROPIC_API_KEY not found")
            print("   Option 1: Create secrets.json file: {'ANTHROPIC_API_KEY': 'your_key_here'}")
            print("   Option 2: Set environment variable: export ANTHROPIC_API_KEY=your_key_here")
        else:
            print("✓ Found ANTHROPIC_API_KEY")
        
        # Create directories
        self.alc_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        Path("results").mkdir(exist_ok=True)  # Ensure results directory exists
    
    def load_api_key(self) -> str:
        """Load API key from secrets.json file or environment variable."""
        # Try to load from secrets.json file first
        secrets_file = Path("secrets.json")
        if secrets_file.exists():
            try:
                with open(secrets_file, 'r') as f:
                    secrets = json.load(f)
                    api_key = secrets.get("ANTHROPIC_API_KEY")
                    if api_key:
                        # Clean the API key of any whitespace
                        api_key = api_key.strip()
                        print("✓ Loaded API key from secrets.json")
                        return api_key
            except Exception as e:
                print(f"⚠️  Warning: Could not read secrets.json: {e}")
        else:
            print("⚠️  Warning: secrets.json not found")
        
        return None
        
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
                    new_row[f'original_completion_{self.current_iteration}'] = row['completion']  # Save original with iteration
                    new_row['completion'] = row['inference']                                        # Replace completion with inference
                    new_row[f'autocorrected_{self.current_iteration}'] = True                       # Mark as auto-corrected with iteration
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
        """Create the text dataset for the next iteration using human annotated data."""
        print(f"🔄 Creating dataset for next iteration (iteration {self.current_iteration})...")
        
        # Always expect human annotated data to exist
        human_annotated_file = prob_output.parent / f"{prob_output.stem}_human_annotated.jsonl"
        
        if not human_annotated_file.exists():
            raise FileNotFoundError(f"Human annotated file not found: {human_annotated_file}")
        
        corrected_file = human_annotated_file
        print(f"✓ Using human annotated data: {corrected_file}")
        
        try:
            # Load human annotated probability data
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
            print(f"✓ Dataset contains {len(rows)} examples with human-annotated labels")
            return next_dataset
            
        except Exception as e:
            print(f"❌ Failed to create next dataset: {e}")
            return None
    
    def rank_autocorrected_by_forced_geo_mean(self, prob_output: Path) -> bool:
        """Rank the autocorrected file by forced_geo_mean (lowest first)."""
        print(f"🔄 Ranking autocorrected data by forced_geo_mean (iteration {self.current_iteration})...")
        
        # Load the autocorrected file
        corrected_file = prob_output.parent / f"{prob_output.stem}_autocorrected.jsonl"
        
        if not corrected_file.exists():
            print(f"❌ Autocorrected file not found: {corrected_file}")
            return False
        
        try:
            # Load autocorrected data
            rows = self.load_probability_rows(corrected_file)
            print(f"✓ Loaded {len(rows)} rows from {corrected_file}")
            
            # Create ranked file (sorted by forced_geo_mean, lowest first)
            ranked_file = prob_output.parent / f"{prob_output.stem}_ranked_by_forced_geo_mean.jsonl"
            self.write_sorted_file(rows, "forced_geo_mean", ranked_file, descending=False)  # ascending order
            
            print(f"✓ Ranking completed. Ranked file saved to: {ranked_file}")
            return True
            
        except Exception as e:
            print(f"❌ Ranking failed: {e}")
            return False
    
    def call_anthropic_for_correction(self, full_prompt: str) -> str:
        """Call Anthropic API to correct spelling and grammar in the full prompt."""
        # Debug information
        print(f"🔍 Debug: API key exists: {bool(self.anthropic_api_key)}")
        print(f"🔍 Debug: API key length: {len(self.anthropic_api_key) if self.anthropic_api_key else 0}")
        print(f"🔍 Debug: Anthropic package available: {anthropic is not None}")
        
        if not self.anthropic_api_key or anthropic is None:
            print("❌ Cannot call Anthropic API: Missing API key or package")
            if not self.anthropic_api_key:
                print("   Issue: API key is missing or empty")
            if anthropic is None:
                print("   Issue: Anthropic package not available - install with: pip install anthropic")
            return full_prompt
        
        try:
            client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            
            anthropic_prompt = f"""You will be given a text containing multiple sections separated by these tags:
            [Course], [UserQuery], [PastChat], and a [ResolvedQuery] tag. I want you to focus on only the [UserQuery] section,
            Could you help me correct only spelling mistakes for the [UserQuery] section, respond only the corrected value
            for the user query section, without including any tags or the other sections. 

            For example, if the text is:
            [Course]Eng_srtc[UserQuery]what is the steps of the resarch process[ResolvedQuery]
            you simply respond with:
            what are the steps of the research process 

            Here is the actual text to correct:
            {full_prompt}"""
            
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": anthropic_prompt}
                ]
            )
            
            corrected_user_query = message.content[0].text.strip()
            print(f"   Original prompt: {full_prompt}")
            print(f"   Corrected UserQuery: {corrected_user_query}")
            return corrected_user_query
            
        except Exception as e:
            print(f"❌ Anthropic API error: {e}")
            return full_prompt
    
    def run_human_annotation(self, prob_output: Path) -> bool:
        """Perform human annotation using Anthropic API on the 2 lowest forced_geo_mean rows."""
        print(f"🔄 Running human annotation with Anthropic API (iteration {self.current_iteration})...")
        
        # Load the ranked file (sorted by forced_geo_mean, lowest first)
        ranked_file = prob_output.parent / f"{prob_output.stem}_ranked_by_forced_geo_mean.jsonl"
        
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
            
            # Get the top 2 lowest forced_geo_mean rows
            top_2_lowest = rows[:2]
            print(f"🎯 Selecting top 2 rows with lowest forced_geo_mean for human annotation:")
            
            annotated_rows = []
            corrections_made = 0
            
            for row in rows:
                new_row = row.copy()
                
                # Check if this row is one of the top 2 lowest
                if any(row.get('rowid') == target.get('rowid') for target in top_2_lowest):
                    print(f"   Row ID: {row.get('rowid', 'N/A')}, Forced Geo Mean: {row.get('forced_geo_mean', 'N/A'):.6f}")
                    
                    # Get the full prompt
                    prompt = row.get('prompt', '')
                    
                    if prompt:
                        # Get correction from Anthropic (pass full prompt, get back corrected UserQuery)
                        corrected_user_query = self.call_anthropic_for_correction(prompt)
                        print("corrected_user_query: ", corrected_user_query)
                        # Extract the original user query to compare
  
                        if corrected_user_query != row.get('completion', ''):
                            # Save original and update fields with iteration number
                            new_row[f'original_completion_{self.current_iteration}'] = row.get('completion', '')
                            new_row['completion'] = corrected_user_query
                            new_row[f'human_corrected_{self.current_iteration}'] = True
                            corrections_made += 1
                            print(f"✓ Human corrected row {row.get('rowid', 'N/A')}")
                        else:
                            new_row[f'human_corrected_{self.current_iteration}'] = False
                            print(f"   No changes needed for row {row.get('rowid', 'N/A')}")
                       
                    else:
                        new_row[f'human_corrected_{self.current_iteration}'] = False
                        print(f"   No prompt found for row {row.get('rowid', 'N/A')}")
                
                annotated_rows.append(new_row)
            
            # Save human annotated data to new file
            annotated_file = prob_output.parent / f"{prob_output.stem}_human_annotated.jsonl"
            with annotated_file.open("w") as f:
                for row in annotated_rows:
                    json.dump(row, f)
                    f.write("\n")
            
            print(f"✓ Human annotation completed: {corrections_made} correction(s) made")
            print(f"✓ Human annotated data saved to: {annotated_file}")
            return True
            
        except Exception as e:
            print(f"❌ Human annotation failed: {e}")
            return False
    
    def run_iteration(self, current_dataset: Path) -> Path:
        """Run a single ALC iteration.
        
        This will create:
        - Model: alcmodels/iteration_X/
        - Probabilities: alcIterations/iteration_X_probabilities.jsonl
        - Sorted files: alcIterations/iteration_X_probabilities_geo_mean_sorted.jsonl
                       alcIterations/iteration_X_probabilities_forced_geo_mean_sorted.jsonl
        - Auto-corrected: alcIterations/iteration_X_probabilities_autocorrected.jsonl
        - Ranked by forced_geo_mean: alcIterations/iteration_X_probabilities_ranked_by_forced_geo_mean.jsonl
        - Human annotated: alcIterations/iteration_X_probabilities_human_annotated.jsonl
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
            
            # Step 5: Rank autocorrected data by forced_geo_mean (lowest first)
            if not self.rank_autocorrected_by_forced_geo_mean(prob_output):
                return None
            
            # Step 6: Human annotation with Anthropic API
            if not self.run_human_annotation(prob_output):
                return None
            
            # Step 7: Create the dataset for the next iteration
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
        print(f"   - Ranked files: iteration_X_probabilities_ranked_by_forced_geo_mean.jsonl")
        print(f"   - Human annotated files: iteration_X_probabilities_human_annotated.jsonl")
        print(f"🤖 Models saved in: {self.models_dir}")
        print(f"🧠 Human annotation: Required - Uses Anthropic API to correct only spelling in UserQuery sections")
        print(f"   Configure API key via secrets.json file or ANTHROPIC_API_KEY environment variable")


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