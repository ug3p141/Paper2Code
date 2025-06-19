import json
import os
from datetime import datetime

def cal_cost_claude(response_json, model_name):
    """Calculate cost for Claude API calls based on current Anthropic pricing."""
    
    # Anthropic pricing (per million tokens) as of 2025
    model_cost = {
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},
    }
    
    # Extract token usage from response
    input_tokens = response_json["usage"]["input_tokens"]
    output_tokens = response_json["usage"]["output_tokens"]
    
    cost_info = model_cost.get(model_name, model_cost["claude-sonnet-4-20250514"])  # Default to Sonnet 4
    
    input_cost = (input_tokens / 1_000_000) * cost_info['input']
    output_cost = (output_tokens / 1_000_000) * cost_info['output']
    total_cost = input_cost + output_cost
    
    return {
        'model_name': model_name,
        'input_tokens': input_tokens,
        'input_cost': input_cost,
        'output_tokens': output_tokens,
        'output_cost': output_cost,
        'total_cost': total_cost,
    }

def print_log_cost_claude(completion_json, claude_model, current_stage, output_dir, total_accumulated_cost):
    """Print and log cost information for Claude API calls."""
    usage_info = cal_cost_claude(completion_json, claude_model)

    current_cost = usage_info['total_cost']
    total_accumulated_cost += current_cost

    output_lines = []
    output_lines.append("🌟 Usage Summary 🌟")
    output_lines.append(f"{current_stage}")
    output_lines.append(f"🛠️ Model: {usage_info['model_name']}")
    output_lines.append(f"📥 Input tokens: {usage_info['input_tokens']} (Cost: ${usage_info['input_cost']:.8f})")
    output_lines.append(f"📤 Output tokens: {usage_info['output_tokens']} (Cost: ${usage_info['output_cost']:.8f})")
    output_lines.append(f"💵 Current total cost: ${current_cost:.8f}")
    output_lines.append(f"🪙 Accumulated total cost so far: ${total_accumulated_cost:.8f}")
    output_lines.append("============================================\n")

    output_text = "\n".join(output_lines)
    
    print(output_text)

    with open(f"{output_dir}/cost_info.log", "a", encoding="utf-8") as f:
        f.write(output_text + "\n")
    
    return total_accumulated_cost

def convert_messages_to_claude(messages):
    """Convert OpenAI-style messages to Claude format."""
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

def claude_api_call(client, messages, model_name, max_tokens=4096):
    """Generic Claude API call function."""
    system_content, formatted_messages = convert_messages_to_claude(messages)
    
    response = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        system=system_content,
        messages=formatted_messages
    )
    
    # Convert to OpenAI-like format for compatibility
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
    
    return completion
