import json
import os
import sys
import argparse
from utils import read_python_files, extract_planning, content_to_json, \
        read_all_files, extract_json_from_string, get_now_str
from vertex_utils import (initialize_vertex_ai, get_claude_model_names, vertex_claude_api_call, 
                         print_log_cost_vertex)

def main(args):
    paper_name = args.paper_name
    pdf_json_path = args.pdf_json_path
    output_dir = args.output_dir
    target_repo_dir = args.target_repo_dir
    eval_result_dir = args.eval_result_dir
    vertex_model = args.vertex_model
    generated_n = args.generated_n
    data_dir = args.data_dir
    eval_type = args.eval_type
    is_papercoder = True if args.papercoder else False
    max_retries = args.max_retries
    base_delay = args.base_delay

    gold_repo_dir = args.gold_repo_dir

    # Initialize Vertex AI
    project_id, location, credentials = initialize_vertex_ai(args.project_id, args.location)

    # paper
    with open(f'{pdf_json_path}') as f:
        paper_json = json.load(f)

    codes = ""
    if is_papercoder:
        # python files
        target_files_dict = read_python_files(target_repo_dir)

        # configuration
        with open(f'{output_dir}/planning_config.yaml') as f:
            config_yaml = f.read()

        context_lst = extract_planning(f'{output_dir}/planning_trajectories.json')

        if os.path.exists(f'{output_dir}/task_list.json'):
            with open(f'{output_dir}/task_list.json') as f:
                task_list = json.load(f)
        else:
            task_list = content_to_json(context_lst[2])

        todo_file_lst = task_list['Task list']
        for todo_file in todo_file_lst:
            if todo_file.endswith(".yaml"):
                continue
            codes += f"```python\n## File name: {todo_file}\n{target_files_dict[todo_file]}\n```\n\n"

        codes += f"```yaml\n## File name: config.yaml\n{config_yaml}\n```\n\n"
    else:
        target_files_dict = read_all_files(target_repo_dir, allowed_ext=[".py", ".yaml", ".yml", ".md", ".sh", ".bash"], is_print=False)
        for file_name, code in target_files_dict.items():
            codes += f"```## File name: {file_name}\n{code}\n```\n\n"

    prompt = open(f"{data_dir}/prompts/{eval_type}.txt").read()

    cur_prompt = prompt.replace('{{Paper}}', f"{paper_json}").replace('{{Code}}', codes)

    # reference-based
    if "ref_based" == eval_type and len(gold_repo_dir) > 0:
        all_files_dict = read_all_files(gold_repo_dir, allowed_ext=[".py", ".yaml", ".yml", ".md", ".sh", ".bash"], is_print=False)

        goldcodes = ""
        gold_cnt = 0
        if len(args.selected_file_path) > 0:
            selected_file_lst = []
            with open(args.selected_file_path) as f:
                selected_file_lst = f.readlines()

            for s_idx in range(len(selected_file_lst)):
                selected_file_lst[s_idx] = selected_file_lst[s_idx].strip()

            for all_file, all_file_code in all_files_dict.items():
                if all_file not in selected_file_lst:
                    continue

                goldcodes += f"```## File name: {all_file}\n{all_file_code}\n```\n\n"
                gold_cnt += 1

        else:
            for all_file, all_file_code in all_files_dict.items():
                goldcodes += f"```## File name: {all_file}\n{all_file_code}\n```\n\n"
                gold_cnt += 1

        cur_prompt = cur_prompt.replace('{{GoldCode}}', f"{goldcodes}")

    # For Vertex AI, we need to handle multiple generations differently
    # Make multiple calls since Vertex AI doesn't support n parameter like OpenAI
    all_scores = []
    rationales = []
    
    msg = [{"role": "user", "content": cur_prompt}]
    
    total_accumulated_cost = 0
    
    for i in range(generated_n):
        print(f"Generating evaluation {i+1}/{generated_n}...")
        
        completion = vertex_claude_api_call(project_id, location, credentials, vertex_model, 
                                            msg, max_tokens=4096, 
                                            max_retries=max_retries, base_delay=base_delay)
        
        # Calculate cost for this call
        temp_total_accumulated_cost = print_log_cost_vertex(completion, vertex_model, 
                                                           f"[Evaluation] {paper_name} - {eval_type} ({i+1}/{generated_n})", 
                                                           output_dir, total_accumulated_cost)
        total_accumulated_cost = temp_total_accumulated_cost
        
        output = completion['choices'][0]['message']['content'].strip()
        
        score_key = "score"
        rationale_key = "critique_list"
        
        try:
            output_json2 = json.loads(output)
            score = int(output_json2[score_key])

            if isinstance(output_json2[rationale_key], str):
                rationale = output_json2[rationale_key]
            else:
                rationale = json.dumps(output_json2[rationale_key])
        except Exception as e:
            try:
                output_json2 = json.loads(extract_json_from_string(output))
                score = int(output_json2[score_key])

                if isinstance(output_json2[rationale_key], str):
                    rationale = output_json2[rationale_key]
                else:
                    rationale = json.dumps(output_json2[rationale_key])
            except Exception as e2:  # Parsing Error
                print(f"[WARNING] Invalid response: parsing error")
                print(e2)
                print("-" * 40)
                continue

        # score validation
        if score < 1 or score > 5:
            print(f"[WARNING] Invalid response: score {score}, Score must be in the range of 1–5.")
            continue

        all_scores.append(int(score))
        rationales.append(rationale)

    if not all_scores:
        print(f"[ERROR] No valid responses generated for {paper_name}")
        return

    avg_score = sum(all_scores) / len(all_scores)

    output_json = {
        "paper_name": paper_name,
        "target_repo_dir": target_repo_dir,
        "eval_type": eval_type,
        "gold_repo_dir": gold_repo_dir,
        "generated_n": generated_n,
        "model": vertex_model,
        "project_id": project_id,
        "location": location,
        "eval_result": {
            "score": avg_score,
            "valid_n": len(all_scores),
            "score_lst": all_scores,
            "rationale_lst": rationales,
        },
    }

    now_str = get_now_str()
    os.makedirs(eval_result_dir, exist_ok=True)
    model_name_safe = vertex_model.replace('@', '_').replace('-', '_')
    with open(f"{eval_result_dir}/{paper_name}_eval_{eval_type}_{model_name_safe}_{now_str}.json", 'w', encoding='utf-8') as f:
        json.dump(output_json, f)

    # ---------------
    print()
    print("=" * 40)
    print("🌟 Evaluation Summary 🌟")
    print(f"📄 Paper name: {paper_name}")
    print(f"🧪 Evaluation type: {eval_type}")
    print(f"📁 Target repo directory: {target_repo_dir}")
    print(f"🛠️ Model: {vertex_model}")
    print(f"☁️ Project ID: {project_id}")
    print(f"📊 Evaluation result:")
    print(f"\t📈 Score: {avg_score:.4f}")
    print(f"\t✅ Valid: {output_json['eval_result']['valid_n']}/{generated_n}")
    print(f"💵 Total cost: ${total_accumulated_cost:.8f}")
    print("=" * 40)
    # ---------------


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--paper_name', type=str)
    argparser.add_argument('--pdf_json_path', type=str)
    argparser.add_argument('--data_dir', type=str, default="../data")

    argparser.add_argument('--output_dir', type=str)

    argparser.add_argument('--target_repo_dir', type=str)
    argparser.add_argument('--gold_repo_dir', type=str, default="")
    argparser.add_argument('--eval_result_dir', type=str)

    argparser.add_argument('--eval_type', type=str, default="ref_free", choices=["ref_free", "ref_based"])

    argparser.add_argument('--generated_n', type=int, default=8)
    argparser.add_argument('--vertex_model', type=str, default="claude-3-5-sonnet")
    
    argparser.add_argument('--max_retries', type=int, default=5)
    argparser.add_argument('--base_delay', type=int, default=60)
    argparser.add_argument('--project_id', type=str, default=None)
    argparser.add_argument('--location', type=str, default="us-central1")

    argparser.add_argument('--selected_file_path', type=str, default="")
    argparser.add_argument('--papercoder', action="store_true")

    args = argparser.parse_args()
    main(args)