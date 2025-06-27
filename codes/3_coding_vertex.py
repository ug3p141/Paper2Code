import json
import os
from tqdm import tqdm
import sys
import copy
from utils import extract_planning, content_to_json, extract_code_from_content, print_response, load_accumulated_cost, save_accumulated_cost
from vertex_utils import (initialize_vertex_ai, get_claude_model_names, vertex_claude_api_call, 
                         print_log_cost_vertex)
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--paper_name', type=str)
parser.add_argument('--vertex_model', type=str, default="claude-3-5-sonnet")
parser.add_argument('--paper_format', type=str, default="JSON", choices=["JSON", "LaTeX"])
parser.add_argument('--pdf_json_path', type=str)  # json format
parser.add_argument('--pdf_latex_path', type=str)  # latex format
parser.add_argument('--output_dir', type=str, default="")
parser.add_argument('--output_repo_dir', type=str, default="")
parser.add_argument('--max_retries', type=int, default=5)
parser.add_argument('--base_delay', type=int, default=60)
parser.add_argument('--project_id', type=str, default=None)
parser.add_argument('--location', type=str, default="us-central1")

args = parser.parse_args()

# Initialize Vertex AI
project_id, location, credentials = initialize_vertex_ai(args.project_id, args.location)

paper_name = args.paper_name
vertex_model = args.vertex_model
paper_format = args.paper_format
pdf_json_path = args.pdf_json_path
pdf_latex_path = args.pdf_latex_path
output_dir = args.output_dir
output_repo_dir = args.output_repo_dir
max_retries = args.max_retries
base_delay = args.base_delay

# Validate model name
available_model_names = get_claude_model_names()
if vertex_model not in available_model_names and vertex_model not in available_model_names.values():
    print(f"❌ Error: Model '{vertex_model}' not found.")
    print("Available models:")
    for short_name, full_name in available_model_names.items():
        print(f"  - {short_name} ({full_name})")
    sys.exit(1)

if paper_format == "JSON":
    with open(f'{pdf_json_path}') as f:
        paper_content = json.load(f)
elif paper_format == "LaTeX":
    with open(f'{pdf_latex_path}') as f:
        paper_content = f.read()
else:
    print(f"[ERROR] Invalid paper format. Please select either 'JSON' or 'LaTeX.")
    sys.exit(0)

with open(f'{output_dir}/planning_config.yaml') as f:
    config_yaml = f.read()

context_lst = extract_planning(f'{output_dir}/planning_trajectories.json')
# 0: overview, 1: detailed, 2: PRD
task_list = content_to_json(context_lst[2])

todo_file_lst = task_list['Task list']
done_file_lst = ['config.yaml']
done_file_dict = {}

code_msg = [
    {"role": "system", "content": f"""You are an expert researcher and software engineer with a deep understanding of experimental design and reproducibility in scientific research.
You will receive a research paper in {paper_format} format, an overview of the plan, a Design in JSON format consisting of "Implementation approach", "File list", "Data structures and interfaces", and "Program call flow", followed by a Task in JSON format that includes "Required packages", "Required other language third-party packages", "Logic Analysis", and "Task list", along with a configuration file named "config.yaml". 
Your task is to write code to reproduce the experiments and methodologies described in the paper. 

The code you write must be elegant, modular, and maintainable, adhering to Google-style guidelines. 
The code must strictly align with the paper's methodology, experimental setup, and evaluation metrics. 
Write code with triple quotes."""}]

def get_write_msg(todo_file_name, detailed_logic_analysis, done_file_lst):
    code_files = ""
    for done_file in done_file_lst:
        if done_file.endswith(".yaml"): continue
        code_files += f"""
```python
{done_file_dict[done_file]}
```

"""

    write_msg = [
        {'role': 'user', "content": f"""# Context
## Paper
{paper_content}

-----

## Overview of the plan
{context_lst[0]}

-----

## Design
{context_lst[1]}

-----

## Task
{context_lst[2]}

-----

## Configuration file
```yaml
{config_yaml}
```
-----

## Code Files
{code_files}

-----

# Format example
## Code: {todo_file_name}
```python
## {todo_file_name}
...
```

-----

# Instruction
Based on the paper, plan, design, task and configuration file(config.yaml) specified previously, follow "Format example", write the code. 

We have {done_file_lst}.
Next, you must write only the "{todo_file_name}".
1. Only One file: do your best to implement THIS ONLY ONE FILE.
2. COMPLETE CODE: Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.
3. Set default value: If there is any setting, ALWAYS SET A DEFAULT VALUE, ALWAYS USE STRONG TYPE AND EXPLICIT VARIABLE. AVOID circular import.
4. Follow design: YOU MUST FOLLOW "Data structures and interfaces". DONT CHANGE ANY DESIGN. Do not use public member functions that do not exist in your design.
5. CAREFULLY CHECK THAT YOU DONT MISS ANY NECESSARY CLASS/FUNCTION IN THIS FILE.
6. Before using a external variable/module, make sure you import it first.
7. Write out EVERY CODE DETAIL, DON'T LEAVE TODO.
8. REFER TO CONFIGURATION: you must use configuration from "config.yaml". DO NOT FABRICATE any configuration values.

{detailed_logic_analysis}

## Code: {todo_file_name}"""}]
    return write_msg

# Load detailed logic analysis for each file
detailed_logic_analysis_dict = {}
for todo_file_name in todo_file_lst:
    save_todo_file_name = todo_file_name.replace("/", "_")

    if todo_file_name == "config.yaml":
        continue

    with open(f"{output_dir}/{save_todo_file_name}_simple_analysis_response.json") as f:
        detailed_logic_analysis_response = json.load(f)
    detailed_logic_analysis_dict[todo_file_name] = detailed_logic_analysis_response[0]['choices'][0]['message']['content']

artifact_output_dir = f'{output_dir}/coding_artifacts'
os.makedirs(artifact_output_dir, exist_ok=True)

total_accumulated_cost = load_accumulated_cost(f"{output_dir}/accumulated_cost.json")
for todo_idx, todo_file_name in enumerate(tqdm(todo_file_lst)):
    responses = []
    trajectories = copy.deepcopy(code_msg)

    current_stage = f"[CODING] {todo_file_name}"
    print(current_stage)

    if todo_file_name == "config.yaml":
        continue

    instruction_msg = get_write_msg(todo_file_name, detailed_logic_analysis_dict[todo_file_name], done_file_lst)
    trajectories.extend(instruction_msg)

    completion = vertex_claude_api_call(project_id, location, credentials, vertex_model, 
                                        trajectories, max_tokens=4096, 
                                        max_retries=max_retries, base_delay=base_delay)

    responses.append(completion)

    # trajectories
    message = completion['choices'][0]['message']
    trajectories.append({'role': message['role'], 'content': message['content']})

    done_file_lst.append(todo_file_name)

    # Create output directories
    os.makedirs(f'{output_repo_dir}', exist_ok=True)
    save_todo_file_name = todo_file_name.replace("/", "_")

    # print and logging
    print_response(completion)
    temp_total_accumulated_cost = print_log_cost_vertex(completion, vertex_model, current_stage, output_dir, total_accumulated_cost)
    total_accumulated_cost = temp_total_accumulated_cost

    # save artifacts - handle subdirectories for artifacts
    artifact_file_path = f'{artifact_output_dir}/{save_todo_file_name}_coding.txt'
    artifact_dir = os.path.dirname(artifact_file_path)
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
    
    with open(artifact_file_path, 'w') as f:
        f.write(completion['choices'][0]['message']['content'])

    # extract code save
    code = extract_code_from_content(message['content'])
    if len(code) == 0:
        code = message['content']

    done_file_dict[todo_file_name] = code
    
    # Create subdirectories for the actual code files if needed
    full_file_path = f"{output_repo_dir}/{todo_file_name}"
    file_dir = os.path.dirname(full_file_path)
    if file_dir and file_dir != output_repo_dir:  # Only create if there's a subdirectory
        os.makedirs(file_dir, exist_ok=True)
        print(f"📁 Created directory: {file_dir}")

    with open(full_file_path, 'w') as f:
        f.write(code)
    
    print(f"💾 Saved file: {full_file_path}")

save_accumulated_cost(f"{output_dir}/accumulated_cost.json", total_accumulated_cost)