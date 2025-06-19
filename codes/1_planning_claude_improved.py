import anthropic
import json
from tqdm import tqdm
import argparse
import os
import sys
import time
from utils import print_response, load_accumulated_cost, save_accumulated_cost
from claude_utils import print_log_cost_claude, convert_messages_to_claude

parser = argparse.ArgumentParser()

parser.add_argument('--paper_name', type=str)
parser.add_argument('--claude_model', type=str, default="claude-sonnet-4-20250514")
parser.add_argument('--paper_format', type=str, default="JSON", choices=["JSON", "LaTeX"])
parser.add_argument('--pdf_json_path', type=str)  # json format
parser.add_argument('--pdf_latex_path', type=str)  # latex format
parser.add_argument('--output_dir', type=str, default="")
parser.add_argument('--max_retries', type=int, default=5)
parser.add_argument('--base_delay', type=int, default=60)  # Base delay in seconds

args = parser.parse_args()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

paper_name = args.paper_name
claude_model = args.claude_model
paper_format = args.paper_format
pdf_json_path = args.pdf_json_path
pdf_latex_path = args.pdf_latex_path
output_dir = args.output_dir
max_retries = args.max_retries
base_delay = args.base_delay

if paper_format == "JSON":
    with open(f'{pdf_json_path}') as f:
        paper_content = json.load(f)
elif paper_format == "LaTeX":
    with open(f'{pdf_latex_path}') as f:
        paper_content = f.read()
else:
    print(f"[ERROR] Invalid paper format. Please select either 'JSON' or 'LaTeX.")
    sys.exit(0)

# Convert messages to Claude format (single system message + user messages)
def convert_messages_to_claude(messages):
    system_message = ""
    user_messages = []
    
    for msg in messages:
        if msg['role'] == 'system':
            system_message = msg['content']
        elif msg['role'] == 'user':
            user_messages.append({"role": "user", "content": msg['content']})
        elif msg['role'] == 'assistant':
            user_messages.append({"role": "assistant", "content": msg['content']})
    
    return system_message, user_messages

def api_call_with_retry(msg, claude_model, max_retries=5, base_delay=60):
    """API call with exponential backoff retry logic for rate limiting."""
    system_content, messages = convert_messages_to_claude(msg)
    
    for attempt in range(max_retries):
        try:
            print(f"🔄 API call attempt {attempt + 1}/{max_retries}")
            
            response = client.messages.create(
                model=claude_model,
                max_tokens=4096,
                system=system_content,
                messages=messages
            )
            
            # Convert to OpenAI-like format for compatibility with existing code
            completion = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": response.content[0].text
                    }
                }],
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
            }
            
            print("✅ API call successful")
            return completion
            
        except anthropic.RateLimitError as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                print(f"⚠️  Rate limit hit. Waiting {delay} seconds before retry {attempt + 2}...")
                time.sleep(delay)
            else:
                print(f"❌ Max retries ({max_retries}) exceeded. Rate limit error: {e}")
                raise
        except anthropic.APIError as e:
            print(f"❌ API Error: {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)
            else:
                raise
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            raise

# Truncate paper content if it's too large
def truncate_content_if_needed(content, max_chars=50000):
    """Truncate content if it exceeds the maximum character limit."""
    if isinstance(content, str) and len(content) > max_chars:
        print(f"⚠️  Paper content is large ({len(content)} chars). Truncating to {max_chars} chars...")
        return content[:max_chars] + "\n\n[CONTENT TRUNCATED DUE TO LENGTH]"
    elif isinstance(content, dict):
        content_str = json.dumps(content)
        if len(content_str) > max_chars:
            print(f"⚠️  Paper content is large ({len(content_str)} chars). Truncating...")
            # Try to keep the structure but reduce content
            if 'body_text' in content:
                # Keep metadata but truncate body text
                truncated_content = content.copy()
                if isinstance(content['body_text'], list) and len(content['body_text']) > 10:
                    truncated_content['body_text'] = content['body_text'][:10]
                    truncated_content['body_text'].append({
                        "text": "[CONTENT TRUNCATED DUE TO LENGTH - SHOWING FIRST 10 SECTIONS ONLY]",
                        "section": "TRUNCATION_NOTICE"
                    })
                return truncated_content
            else:
                return json.loads(json.dumps(content)[:max_chars] + '"}')
    return content

# Truncate paper content if needed
paper_content = truncate_content_if_needed(paper_content)

plan_msg = [
    {'role': "system", "content": f"""You are an expert researcher and strategic planner with a deep understanding of experimental design and reproducibility in scientific research. 
You will receive a research paper in {paper_format} format. 
Your task is to create a detailed and efficient plan to reproduce the experiments and methodologies described in the paper.
This plan should align precisely with the paper's methodology, experimental setup, and evaluation metrics. 

Instructions:

1. Align with the Paper: Your plan must strictly follow the methods, datasets, model configurations, hyperparameters, and experimental setups described in the paper.
2. Be Clear and Structured: Present the plan in a well-organized and easy-to-follow format, breaking it down into actionable steps.
3. Prioritize Efficiency: Optimize the plan for clarity and practical implementation while ensuring fidelity to the original experiments."""},
    {"role": "user",
     "content": f"""## Paper
{paper_content}

## Task
1. We want to reproduce the method described in the attached paper. 
2. The authors did not release any official code, so we have to plan our own implementation.
3. Before writing any Python code, please outline a comprehensive plan that covers:
   - Key details from the paper's **Methodology**.
   - Important aspects of **Experiments**, including dataset requirements, experimental settings, hyperparameters, or evaluation metrics.
4. The plan should be as **detailed and informative** as possible to help us write the final code later.

## Requirements
- You don't need to provide the actual code yet; focus on a **thorough, clear strategy**.
- If something is unclear from the paper, mention it explicitly.

## Instruction
The response should give us a strong roadmap, making it easier to write the code later."""}]

file_list_msg = [
    {"role": "user", "content": """Your goal is to create a concise, usable, and complete software system design for reproducing the paper's method. Use appropriate open-source libraries and keep the overall architecture simple.
             
Based on the plan for reproducing the paper's main method, please design a concise, usable, and complete software system. 
Keep the architecture simple and make effective use of open-source libraries.

-----

## Format Example
[CONTENT]
{
    "Implementation approach": "We will ... ,
    "File list": [
        "main.py",  
        "dataset_loader.py", 
        "model.py",  
        "trainer.py",
        "evaluation.py" 
    ],
    "Data structures and interfaces": "\nclassDiagram\n    class Main {\n        +__init__()\n        +run_experiment()\n    }\n    class DatasetLoader {\n        +__init__(config: dict)\n        +load_data() -> Any\n    }\n    class Model {\n        +__init__(params: dict)\n        +forward(x: Tensor) -> Tensor\n    }\n    class Trainer {\n        +__init__(model: Model, data: Any)\n        +train() -> None\n    }\n    class Evaluation {\n        +__init__(model: Model, data: Any)\n        +evaluate() -> dict\n    }\n    Main --> DatasetLoader\n    Main --> Trainer\n    Main --> Evaluation\n    Trainer --> Model\n",
    "Program call flow": "\nsequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant MD as Model\n    participant TR as Trainer\n    participant EV as Evaluation\n    M->>DL: load_data()\n    DL-->>M: return dataset\n    M->>MD: initialize model()\n    M->>TR: train(model, dataset)\n    TR->>MD: forward(x)\n    MD-->>TR: predictions\n    TR-->>M: training complete\n    M->>EV: evaluate(model, dataset)\n    EV->>MD: forward(x)\n    MD-->>EV: predictions\n    EV-->>M: metrics\n",
    "Anything UNCLEAR": "Need clarification on the exact dataset format and any specialized hyperparameters."
}
[/CONTENT]

## Nodes: "<node>: <type>  # <instruction>"
- Implementation approach: <class 'str'>  # Summarize the chosen solution strategy.
- File list: typing.List[str]  # Only need relative paths. ALWAYS write a main.py or app.py here.
- Data structures and interfaces: typing.Optional[str]  # Use mermaid classDiagram code syntax, including classes, method(__init__ etc.) and functions with type annotations, CLEARLY MARK the RELATIONSHIPS between classes, and comply with PEP8 standards. The data structures SHOULD BE VERY DETAILED and the API should be comprehensive with a complete design.
- Program call flow: typing.Optional[str] # Use sequenceDiagram code syntax, COMPLETE and VERY DETAILED, using CLASSES AND API DEFINED ABOVE accurately, covering the CRUD AND INIT of each object, SYNTAX MUST BE CORRECT.
- Anything UNCLEAR: <class 'str'>  # Mention ambiguities and ask for clarifications.

## Constraint
Format: output wrapped inside [CONTENT][/CONTENT] like the format example, nothing else.

## Action
Follow the instructions for the nodes, generate the output, and ensure it follows the format example."""}
]

task_list_msg = [
    {'role': 'user', 'content': """Your goal is break down tasks according to PRD/technical design, generate a task list, and analyze task dependencies. 
You will break down tasks, analyze dependencies.
             
You outline a clear PRD/technical design for reproducing the paper's method and experiments. 

Now, let's break down tasks according to PRD/technical design, generate a task list, and analyze task dependencies.
The Logic Analysis should not only consider the dependencies between files but also provide detailed descriptions to assist in writing the code needed to reproduce the paper.

-----

## Format Example
[CONTENT]
{
    "Required packages": [
        "numpy==1.21.0",
        "torch==1.9.0"  
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "data_preprocessing.py",
            "DataPreprocessing class ........"
        ],
        [
            "trainer.py",
            "Trainer ....... "
        ],
        [
            "dataset_loader.py",
            "Handles loading and ........"
        ],
        [
            "model.py",
            "Defines the model ......."
        ],
        [
            "evaluation.py",
            "Evaluation class ........ "
        ],
        [
            "main.py",
            "Entry point  ......."
        ]
    ],
    "Task list": [
        "dataset_loader.py", 
        "model.py",  
        "trainer.py", 
        "evaluation.py",
        "main.py"  
    ],
    "Full API spec": "openapi: 3.0.0 ...",
    "Shared Knowledge": "Both data_preprocessing.py and trainer.py share ........",
    "Anything UNCLEAR": "Clarification needed on recommended hardware configuration for large-scale experiments."
}

[/CONTENT]

## Nodes: "<node>: <type>  # <instruction>"
- Required packages: typing.Optional[typing.List[str]]  # Provide required third-party packages in requirements.txt format.(e.g., 'numpy==1.21.0').
- Required Other language third-party packages: typing.List[str]  # List down packages required for non-Python languages. If none, specify "No third-party dependencies required".
- Logic Analysis: typing.List[typing.List[str]]  # Provide a list of files with the classes/methods/functions to be implemented, including dependency analysis and imports. Include as much detailed description as possible.
- Task list: typing.List[str]  # Break down the tasks into a list of filenames, prioritized based on dependency order. The task list must include the previously generated file list.
- Full API spec: <class 'str'>  # Describe all APIs using OpenAPI 3.0 spec that may be used by both frontend and backend. If front-end and back-end communication is not required, leave it blank.
- Shared Knowledge: <class 'str'>  # Detail any shared knowledge, like common utility functions or configuration variables.
- Anything UNCLEAR: <class 'str'>  # Mention any unresolved questions or clarifications needed from the paper or project scope.

## Constraint
Format: output wrapped inside [CONTENT][/CONTENT] like the format example, nothing else.

## Action
Follow the node instructions above, generate your output accordingly, and ensure it follows the given format example."""}]

# config
config_msg = [
    {'role': 'user', 'content': """You write elegant, modular, and maintainable code. Adhere to Google-style guidelines.

Based on the paper, plan, design specified previously, follow the "Format Example" and generate the code. 
Extract the training details from the above paper (e.g., learning rate, batch size, epochs, etc.), follow the "Format example" and generate the code. 
DO NOT FABRICATE DETAILS — only use what the paper provides.

You must write `config.yaml`.

ATTENTION: Use '##' to SPLIT SECTIONS, not '#'. Your output format must follow the example below exactly.

-----

# Format Example
## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: ...
  batch_size: ...
  epochs: ...
...
```

-----

## Code: config.yaml
"""}]

responses = []
trajectories = []
total_accumulated_cost = 0

# Add delays between API calls to respect rate limits
INTER_CALL_DELAY = 30  # seconds between calls

for idx, instruction_msg in enumerate([plan_msg, file_list_msg, task_list_msg, config_msg]):
    current_stage = ""
    if idx == 0:
        current_stage = f"[Planning] Overall plan"
    elif idx == 1:
        current_stage = f"[Planning] Architecture design"
    elif idx == 2:
        current_stage = f"[Planning] Logic design"
    elif idx == 3:
        current_stage = f"[Planning] Configuration file generation"
    print(f"\n{'='*50}")
    print(current_stage)
    print(f"{'='*50}")

    trajectories.extend(instruction_msg)

    # Add delay between calls (except for the first one)
    if idx > 0:
        print(f"⏳ Waiting {INTER_CALL_DELAY} seconds to respect rate limits...")
        time.sleep(INTER_CALL_DELAY)

    completion = api_call_with_retry(trajectories, claude_model, max_retries, base_delay)

    # print and logging
    print_response(completion)
    temp_total_accumulated_cost = print_log_cost_claude(completion, claude_model, current_stage, output_dir, total_accumulated_cost)
    total_accumulated_cost = temp_total_accumulated_cost

    responses.append(completion)

    # trajectories
    message = completion['choices'][0]['message']
    trajectories.append({'role': message['role'], 'content': message['content']})

# save
save_accumulated_cost(f"{output_dir}/accumulated_cost.json", total_accumulated_cost)

os.makedirs(output_dir, exist_ok=True)

with open(f'{output_dir}/planning_response.json', 'w') as f:
    json.dump(responses, f)

with open(f'{output_dir}/planning_trajectories.json', 'w') as f:
    json.dump(trajectories, f)

print(f"\n🎉 Planning completed successfully!")
print(f"💾 Results saved to: {output_dir}")
print(f"💵 Total cost: ${total_accumulated_cost:.8f}")
