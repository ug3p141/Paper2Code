import json
import os
from tqdm import tqdm
import sys
from utils import extract_planning, content_to_json, print_response, load_accumulated_cost, save_accumulated_cost
from vertex_utils import (initialize_vertex_ai, get_claude_model_names, vertex_claude_api_call, 
                         print_log_cost_vertex)
import copy
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--paper_name', type=str)
parser.add_argument('--vertex_model', type=str, default="claude-3-5-sonnet")
parser.add_argument('--paper_format', type=str, default="JSON", choices=["JSON", "LaTeX"])
parser.add_argument('--pdf_json_path', type=str)  # json format
parser.add_argument('--pdf_latex_path', type=str)  # latex format
parser.add_argument('--output_dir', type=str, default="")
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
if os.path.exists(f'{output_dir}/task_list.json'):
    with open(f'{output_dir}/task_list.json') as f:
        task_list = json.load(f)
else:
    task_list = content_to_json(context_lst[2])

if 'Task list' in task_list:
    todo_file_lst = task_list['Task list']
elif 'task_list' in task_list:
    todo_file_lst = task_list['task_list']
elif 'task list' in task_list:
    todo_file_lst = task_list['task list']
else:
    print(f"[ERROR] 'Task list' does not exist. Please re-generate the planning.")
    sys.exit(0)

if 'Logic Analysis' in task_list:
    logic_analysis = task_list['Logic Analysis']
elif 'logic_analysis' in task_list:
    logic_analysis = task_list['logic_analysis']
elif 'logic analysis' in task_list:
    logic_analysis = task_list['logic analysis']
else:
    print(f"[ERROR] 'Logic Analysis' does not exist. Please re-generate the planning.")
    sys.exit(0)

done_file_lst = ['config.yaml']
logic_analysis_dict = {}
for desc in task_list['Logic Analysis']:
    logic_analysis_dict[desc[0]] = desc[1]

analysis_msg = [
    {"role": "system", "content": f"""You are an expert researcher, strategic analyzer and software engineer with a deep understanding of experimental design and reproducibility in scientific research.
You will receive a research paper in {paper_format} format, an overview of the plan, a design in JSON format consisting of "Implementation approach", "File list", "Data structures and interfaces", and "Program call flow", followed by a task in JSON format that includes "Required packages", "Required other language third-party packages", "Logic Analysis", and "Task list", along with a configuration file named "config.yaml". 

Your task is to conduct a comprehensive logic analysis to accurately reproduce the experiments and methodologies described in the research paper. 
This analysis must align precisely with the paper's methodology, experimental setup, and evaluation criteria.

1. Align with the Paper: Your analysis must strictly follow the methods, datasets, model configurations, hyperparameters, and experimental setups described in the paper.
2. Be Clear and Structured: Present your analysis in a logical, well-organized, and actionable format that is easy to follow and implement.
3. Prioritize Efficiency: Optimize the analysis for clarity and practical implementation while ensuring fidelity to the original experiments.
4. Follow design: YOU MUST FOLLOW "Data structures and interfaces". DONT CHANGE ANY DESIGN. Do not use public member functions that do not exist in your design.
5. REFER TO CONFIGURATION: Always reference settings from the config.yaml file. Do not invent or assume any values—only use configurations explicitly provided.
"""}]

def get_write_msg(todo_file_name, todo_file_desc):
    draft_desc = f"Write the logic analysis in '{todo_file_name}', which is intended for '{todo_file_desc}'."
    if len(todo_file_desc.strip()) == 0:
        draft_desc = f"Write the logic analysis in '{todo_file_name}'."

    write_msg = [{'role': 'user', "content": f"""## Paper
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

## Instruction
Conduct a Logic Analysis to assist in writing the code, based on the paper, the plan, the design, the task and the previously specified configuration file (config.yaml). 
You DON'T need to provide the actual code yet; focus on a thorough, clear analysis.

{draft_desc}

-----

## Logic Analysis: {todo_file_name}"""}]
    return write_msg

artifact_output_dir = f'{output_dir}/analyzing_artifacts'
os.makedirs(artifact_output_dir, exist_ok=True)

total_accumulated_cost = load_accumulated_cost(f"{output_dir}/accumulated_cost.json")
for todo_file_name in tqdm(todo_file_lst):
    responses = []
    trajectories = copy.deepcopy(analysis_msg)

    current_stage = f"[ANALYSIS] {todo_file_name}"
    print(current_stage)
    if todo_file_name == "config.yaml":
        continue

    if todo_file_name not in logic_analysis_dict:
        logic_analysis_dict[todo_file_name] = ""

    instruction_msg = get_write_msg(todo_file_name, logic_analysis_dict[todo_file_name])
    trajectories.extend(instruction_msg)

    completion = vertex_claude_api_call(project_id, location, credentials, vertex_model, 
                                        trajectories, max_tokens=4096, 
                                        max_retries=max_retries, base_delay=base_delay)

    responses.append(completion)

    # trajectories
    message = completion['choices'][0]['message']
    trajectories.append({'role': message['role'], 'content': message['content']})

    # print and logging
    print_response(completion)
    temp_total_accumulated_cost = print_log_cost_vertex(completion, vertex_model, current_stage, output_dir, total_accumulated_cost)
    total_accumulated_cost = temp_total_accumulated_cost

    # save
    # Create subdirectories for artifacts if needed
    artifact_file_path = f'{artifact_output_dir}/{todo_file_name}_simple_analysis.txt'
    artifact_dir = os.path.dirname(artifact_file_path)
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
    
    with open(artifact_file_path, 'w') as f:
        f.write(completion['choices'][0]['message']['content'])

    done_file_lst.append(todo_file_name)

    # save for next stage(coding) - handle subdirectories
    save_todo_file_name = todo_file_name.replace("/", "_")
    
    response_file_path = f'{output_dir}/{save_todo_file_name}_simple_analysis_response.json'
    trajectories_file_path = f'{output_dir}/{save_todo_file_name}_simple_analysis_trajectories.json'
    
    with open(response_file_path, 'w') as f:
        json.dump(responses, f)

    with open(trajectories_file_path, 'w') as f:
        json.dump(trajectories, f)

save_accumulated_cost(f"{output_dir}/accumulated_cost.json", total_accumulated_cost)