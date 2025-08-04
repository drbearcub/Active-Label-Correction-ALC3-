import json


def transform_jsonl_to_json(input_file_path, output_file_path):
    """
    Reads a JSONL file, transforms each line into a new JSON object format,
    and writes the result to a single JSON file.

    Args:
        input_file_path (str): The path to the input JSONL file.
        output_file_path (str): The path where the output JSON file will be saved.
    """
    transformed_data = []

    try:
        # Open the input JSONL file for reading
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            # Process each line in the file
            for line in infile:
                # Skip empty lines
                if not line.strip():
                    continue

                # Load the JSON object from the current line
                original_record = json.loads(line)

                # --- Transformation Logic ---

                user_query = original_record.get('query', '')
                course_name = original_record.get('question_metadata').get('course')

                # 4. Create the new transformed JSON object
                new_record = {
                    'id': original_record.get('question_db_id'),
                    'course_name': '[CourseName]' + course_name,
                    'user_query': '[UserQuery]' + user_query + '[ResolvedQuery]',
                    'completion.': original_record.get('gpt3_5_jill'),
                    'groundtruth': original_record.get('ground_truth'),
                    'pastchat': original_record.get('pastChat', [])
                }

                # 5. Add the newly created record to our list
                transformed_data.append(new_record)

        # Open the output file for writing
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            # Write the entire list of transformed objects to the file
            # `indent=4` makes the JSON output human-readable (pretty-printed)
            json.dump(transformed_data, outfile, indent=4)

        print(f"Successfully transformed '{input_file_path}' to '{output_file_path}'")

    except FileNotFoundError:
        print(f"Error: The file '{input_file_path}' was not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from the input file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- How to use the script ---

# 1. Replace 'input.jsonl' with the actual path to your JSONL file.
input_filename = './rawdata/train_formatted.jsonl'

# 2. Replace 'output.json' with the desired name for your new JSON file.
output_filename = 'alcIterations/iteration_a_dataset.json'

# 3. (Optional) Create a dummy input file for testing purposes.
# If you don't have your file ready, you can uncomment the block below
# to create a sample 'input.jsonl' to test the script.
"""
with open(input_filename, 'w') as f:
    f.write('{"question_db_id": "65a707c45546a1eff21bcf88", "input_query": "Course: CogSci\\nQuestion: heuristic\\nPast Chat: USER: hi\\nASSISTANT: Hello! How can I assist you today?", "query": "heuristic", "pastChat": [{"user": "hi"}, {"assistant": "Hello! How can I assist you today?"}], "pastChatTruncated": [{"user": "hi"}, {"assistant": "Hello! How can I assist you today?"}], "question_metadata": {"course": "CogSci", "semester": "Spring", "year": 2024}, "gpt3_5_jill": "heuristic", "gpt4o": "heuristic", "gpt4o_mini": "heuristic", "o3mini": "heuristic", "shard": "validation", "ground_truth": "heuristic"}\\n')
    f.write('{"question_db_id": "65a707c45546a1eff21bcf89", "input_query": "Course: Eng_srtc\\nQuestion: what is the steps of the research process", "query": "research process steps", "pastChat": [], "pastChatTruncated": [], "question_metadata": {"course": "Eng_srtc", "semester": "Fall", "year": 2023}, "gpt3_5_jill": "research steps", "gpt4o": "research process", "gpt4o_mini": "research steps", "o3mini": "research steps", "shard": "training", "ground_truth": "The research process involves identifying a topic, conducting a literature review, formulating a hypothesis, designing the study, collecting data, analyzing data, and reporting the findings."}\\n')
"""

# 4. Run the transformation function.
transform_jsonl_to_json(input_filename, output_filename)
