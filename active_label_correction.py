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
from typing import List, Dict, Any, Optional
import argparse
import shutil


class ALCPipeline:
    def __init__(self, iterations: int = 5, initial_data: str = "alcIterations/iteration_0_dataset.json", start_iteration = 0):
        self.iterations = iterations
        self.initial_data = Path(initial_data)
        self.start_iteration = start_iteration
        self.alc_dir = Path("alcIterations")
        self.models_dir = Path("alcmodels")

        self.current_iteration = 0

        # Create directories
        self.alc_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        Path("results").mkdir(exist_ok=True)  # Ensure results directory exists

    def run_training(self, train_file: Path) -> Optional[Path]:
        """
        Run training and return the path to the output model.
        """
        print(f"🔄 Training model (iteration {self.current_iteration})...")

        # Determine base model path
        base_model = 'openai-community/gpt2'

        # do not retrain previously trained model. always use the same gpt2 model
        # if self.current_iteration > 0:
        #     prev_model = self.models_dir / f"iteration_{self.current_iteration - 1}"
        #     if prev_model.exists():
        #         base_model = str(prev_model)

        output_model = self.models_dir / f"iteration_{self.current_iteration}"

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
            "--current_iteration", f"{self.current_iteration}",
        ]

        print(f"Running command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Training completed. Model saved to {output_model}")
            return output_model
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed with return code: {e.returncode}")
            return None

    def run_inference(self, model_path: Path, data_path: Path) -> Optional[Path]:
        """
        Run inference and return the path to the probabilities file.
        """
        print(f"🔄 Running inference (iteration {self.current_iteration})...")

        output_path = self.alc_dir / f"iteration_{self.current_iteration}_step_2_probabilities.json"

        cmd = [
            sys.executable, "inference.py",
            "--model_path", str(model_path),
            "--data_path", str(data_path),
            "--prob_output_path", str(output_path),
        ]

        print(f"Running command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Inference completed. Probabilities saved to {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"❌ Inference failed with return code: {e.returncode}")
            return None

    def load_json_rows(self, json_path: Path) -> List[Dict[str, Any]]:
        """Load data from a JSON array file into a list of dictionaries."""
        try:
            with json_path.open(encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                print(f"⚠️  Warning: JSON file '{json_path}' does not contain a list.")
                return []
            return data
        except json.JSONDecodeError as e:
            print(f"❌ Error: Failed to decode JSON from '{json_path}': {e}")
            return []
        except FileNotFoundError:
            print(f"❌ Error: File not found at '{json_path}'")
            return []

    def write_json_rows(self, rows: list, out_path: Path):
        """Writes rows to a JSON array file with pretty printing."""
        try:
            with out_path.open('w', encoding='utf-8') as f:
                json.dump(rows, f, indent=4)
            print(f"✓ Successfully wrote {len(rows)} rows to {out_path}")
        except IOError as e:
            print(f"❌ Error writing to file {out_path}: {e}")

    def run_sorting(self, prob_file: Path) -> Optional[Path]:
        """
        Sort the probability file by confidence metrics and return the output path.
        """
        print(f"🔄 Sorting probability file by confidence (iteration {self.current_iteration})...")

        rows = self.load_json_rows(prob_file)
        if not rows:
            return None

        # Sort by geo_mean (desc)
        sorted_rows = sorted(rows, key=lambda r: r.get("geo_mean", 0), reverse=True)

        output_path = self.alc_dir / f"iteration_{self.current_iteration}_step_3_geo_mean_sorted.json"
        self.write_json_rows(sorted_rows, output_path)

        print(f"✓ Sorting completed.")
        return output_path

    def run_sorting_by_forced(self, prob_file: Path) -> Optional[Path]:
        """
        Sort the probability file by confidence metrics and return the output path.
        """
        print(f"🔄 Sorting probability file by confidence (iteration {self.current_iteration})...")

        rows = self.load_json_rows(prob_file)
        if not rows:
            return None

        # Sort by forced geo_mean (asc)
        sorted_rows = sorted(rows, key=lambda r: r.get("forced_geo_mean", 0), reverse=False)

        output_path = self.alc_dir / f"iteration_{self.current_iteration}_step_5_forced_geo_mean_sorted.json"
        self.write_json_rows(sorted_rows, output_path)

        print(f"✓ Sorting completed.")
        return output_path

    def run_auto_correction(self, sorted_file: Path) -> Optional[Path]:
        """
        Perform auto-correction and return the path to the corrected file.
        """
        print(f"🔄 Running auto-correction (iteration {self.current_iteration})...")

        rows = self.load_json_rows(sorted_file)
        if not rows:
            return None

        rows_corrected = 0
        # Iterate through rows to find and apply human annotations
        # ALC Strategy: For the most confident model prediction, use them for training instead of the original completion.
        for row in rows:
            geo_mean = row.get("geo_mean", 0)
            if geo_mean <= 0.998229: # hardcoded delta value. only auto-correct if model is really confident.
                break

            humanCorrected = row.get('human_corrected_iteration')

            if humanCorrected is not None:
                print("Skip auto correct because it is already human annotated")
                continue

            if "autocorrected_iterations" not in row:
                row["autocorrected_iterations"] = []

            row["autocorrected_iterations"].append(self.current_iteration)
            rows_corrected += 1

        output_path = self.alc_dir / f"iteration_{self.current_iteration}_step_4_autocorrected.json"
        self.write_json_rows(rows, output_path)

        print(f"✓ Auto-correction completed.")
        return output_path

    def run_human_annotation(self, autocorrected_file: Path) -> Optional[Path]:
        """
        Perform human annotation and return the path to the annotated file.
        """
        print(f"🔄 Running human annotation (iteration {self.current_iteration})...")

        rows = self.load_json_rows(autocorrected_file)
        if not rows:
            return None

        rows_corrected = 0
        # Iterate through rows to find and apply human annotations
        for row in rows:
            # Skip rows that were just auto-corrected in this same iteration
            auto_corrected_history = row.get("autocorrected_iterations")
            humanCorrected = row.get('human_corrected_iteration')

            if auto_corrected_history is not None and self.current_iteration in auto_corrected_history:
                print("Skip human annotation for rows that were just auto-corrected in this same iteration")
                continue

            if humanCorrected is not None:
                print("Skip human annotation because it is already human annotated")
                continue

            # Check if an external human annotation has been provided
            corrected_user_query = row.get('human_annotation')
            if corrected_user_query and corrected_user_query != row.get('completion.', ''):
                row[f'original_completion_{self.current_iteration}'] = row.get('completion.', '')
                row['completion.'] = corrected_user_query
                row[f'human_corrected_iteration'] = self.current_iteration
                rows_corrected += 1
                print(f"✓ Applied human correction to row {row.get('id', 'N/A')}")

            if rows_corrected == 2:
                break

        print(f"✓ Total human corrections applied: {rows_corrected}")

        output_path = self.alc_dir / f"iteration_{self.current_iteration}_step_6_human_annotated.json"
        self.write_json_rows(rows, output_path)

        print(f"✓ Human annotation step completed.")
        print(f"✓ Human annotation step completed.")
        return output_path

    def run_filter(self, human_corrected_file: Path) -> Optional[Path]:
        """
        Perform filter and save file to filtered file.
        """
        print(f"🔄 Running filter (iteration {self.current_iteration})...")

        rows = self.load_json_rows(human_corrected_file)
        if not rows:
            return None

        rows_filtered = 0
        # Iterate through rows to find and apply filter
        for row in rows:
            if rows_filtered == 2 :
                break
            # Skip rows that were just auto-corrected in this same iteration
            auto_corrected_history = row.get("autocorrected_iterations")
            humanCorrected = row.get('human_corrected_iteration')

            if auto_corrected_history is not None and self.current_iteration in auto_corrected_history:
                print("Skip human annotation for rows that were just auto-corrected in this same iteration")
                continue

            if humanCorrected is not None:
                print("Skip human annotation because it is already human annotated")
                continue

            if "filtered_iterations" not in row:
                row["filtered_iterations"] = []

            row["filtered_iterations"].append(self.current_iteration)
            rows_filtered = rows_filtered + 1

        print(f"✓ Total human corrections applied: {rows_filtered}")

        output_path = self.alc_dir / f"iteration_{self.current_iteration}_step_7_filtered.json"
        self.write_json_rows(rows, output_path)

        print(f"✓ filter step completed.")
        return output_path

    def create_next_dataset(self, filtered_file: Path) -> Optional[Path]:
        """
        Create the dataset for the next iteration and return its path.
        """
        print(f"🔄 Creating dataset for next iteration...")

        if not filtered_file.exists():
            print(f"❌ Annotated file not found: {filtered_file}")
            return None

        next_dataset_path = self.alc_dir / f"iteration_{self.current_iteration + 1}_dataset.json"
        shutil.copy(filtered_file, next_dataset_path)

        print(f"✓ Next dataset created at: {next_dataset_path}")
        return next_dataset_path

    def run_iteration(self, current_dataset: Path) -> Optional[Path]:
        """
        Run a single, complete ALC iteration.
        """
        print(f"\n{'=' * 50}\n🚀 Starting ALC Iteration {self.current_iteration}\n{'=' * 50}")

        try:
            # Step 1: Train model
            model_path = self.run_training(current_dataset)
            if not model_path: return None

            # Step 2: Run inference
            prob_file = self.run_inference(model_path, current_dataset)
            if not prob_file: return None

            # Step 3: Sort results
            sorted_file = self.run_sorting(prob_file)
            if not sorted_file: return None

            # Step 4: Auto-correct most confident error
            autocorrected_file = self.run_auto_correction(sorted_file)
            if not autocorrected_file: return None

            # Step 5: Sort again by forced geo mean, ascending
            sorted_file_forced_geomean = self.run_sorting_by_forced(autocorrected_file)
            if not sorted_file_forced_geomean: return None

            # Step 6: Apply human annotations
            human_annotated_file = self.run_human_annotation(sorted_file_forced_geomean)
            if not human_annotated_file: return None

            # Step 7: Apply filter for next iteration
            filtered_file = self.run_filter(human_annotated_file)
            if not filtered_file: return None

            # Step 8: Create dataset for the next iteration
            next_dataset = self.create_next_dataset(filtered_file)
            if not next_dataset: return None

            print(f"✅ Iteration {self.current_iteration} completed successfully!")
            return next_dataset

        except Exception as e:
            print(f"❌ Iteration {self.current_iteration} failed with an unexpected error: {e}")
            return None

    def run_pipeline(self):
        """Run the complete ALC pipeline for the specified number of iterations."""
        print("🚀 Starting Active Label Correction Pipeline")
        print(f"📊 Running {self.iterations} iterations")

        current_dataset = self.initial_data
        if not current_dataset.exists():
            print(f"❌ Initial dataset not found at {self.initial_data}. Aborting.")
            return

        for i in range(self.start_iteration, self.iterations):
            self.current_iteration = i
            next_dataset = self.run_iteration(current_dataset)

            if next_dataset is None:
                print(f"⚠️  Pipeline stopped at iteration {i} due to a failure.")
                break

            current_dataset = next_dataset

        print(f"\n🎉 ALC Pipeline finished!")
        print(f"📁 Final outputs are in: {self.alc_dir}")
        print(f"🤖 Final model is in: {self.models_dir / f'iteration_{self.current_iteration}'}")


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
        help="Path to the initial dataset"
    )
    parser.add_argument(
        "--start_iteration",
        type=int,
        default=0,
        help="Iteration number to start from (default: 0)"
    )


    args = parser.parse_args()

    pipeline = ALCPipeline(
        iterations=args.iterations,
        initial_data=args.initial_data,
        start_iteration=args.start_iteration
    )
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
