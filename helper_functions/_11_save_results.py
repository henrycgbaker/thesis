import os
import json

def get_persistent_unique_id():
    """
    Retrieves a persistent unique ID from a local file and increments it.
    """
    ID_FILE = "experiment_id.txt"
    if os.path.exists(ID_FILE):
        with open(ID_FILE, "r") as f:
            try:
                last_id = int(f.read().strip())
            except ValueError:
                last_id = 0
    else:
        last_id = 0
    new_id = last_id + 1
    with open(ID_FILE, "w") as f:
        f.write(str(new_id))
    return f"{new_id:04d}"

def save_results(task_type, benchmark_results):
    """
    Saves benchmark results as a JSON log.
    
    Parameters:
      - task_type: A string representing the type of task.
      - benchmark_results: A dictionary of results to save.
      
    Returns:
      The file path where the results were saved.
    """
    output_dir = "benchmark_results"
    os.makedirs(output_dir, exist_ok=True)
    output_json_path = os.path.join(output_dir, f"{task_type}_results.json")
    
    if os.path.exists(output_json_path):
        with open(output_json_path, "r") as json_file:
            try:
                existing_data = json.load(json_file)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []
    
    existing_data.append(benchmark_results)
    with open(output_json_path, "w") as json_file:
        json.dump(existing_data, json_file, indent=4)
    
    return output_json_path
