import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.cloud import aiplatform

def initialize_vertex_ai(project_id: str = None, location: str = "us-central1"):
    """Initialize Vertex AI with project credentials."""
    if project_id is None:
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise ValueError("Please set GOOGLE_CLOUD_PROJECT environment variable or pass project_id")
    
    vertexai.init(project=project_id, location=location)
    print(f"✅ Vertex AI initialized for project: {project_id}, location: {location}")
    return project_id, location

def get_vertex_model(model_name: str = "claude-3-5-sonnet-v2@20241022"):
    """Get Vertex AI model instance for Claude."""
    # Available Claude models in Vertex AI:
    # - claude-3-5-sonnet-v2@20241022
    # - claude-3-5-haiku@20241022  
    # - claude-3-opus@20240229
    
    model = GenerativeModel(model_name)
    return model

def convert_messages_to_vertex(messages: List[Dict[str, str]]) -> tuple:
    """Convert OpenAI-style messages to Vertex AI format."""
    system_instruction = ""
    conversation_parts = []
    
    for msg in messages:
        if msg['role'] == 'system':
            system_instruction = msg['content']
        elif msg['role'] == 'user':
            conversation_parts.append(Part.from_text(f"Human: {msg['content']}"))
        elif msg['role'] == 'assistant':
            conversation_parts.append(Part.from_text(f"Assistant: {msg['content']}"))
    
    return system_instruction, conversation_parts

def vertex_api_call(model, messages: List[Dict[str, str]], max_output_tokens: int = 4096, 
                   temperature: float = 0.7, max_retries: int = 5, base_delay: int = 60):
    """Make API call to Claude via Vertex AI with retry logic."""
    
    system_instruction, conversation_parts = convert_messages_to_vertex(messages)
    
    # Combine all parts into a single prompt for Claude
    full_prompt = ""
    if system_instruction:
        full_prompt += f"System: {system_instruction}\n\n"
    
    # Extract the conversation
    for part in conversation_parts:
        full_prompt += part.text + "\n\n"
    
    # Add final instruction for response
    full_prompt += "Assistant: "
    
    generation_config = {
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": 0.95,
    }
    
    for attempt in range(max_retries):
        try:
            print(f"🔄 Vertex AI API call attempt {attempt + 1}/{max_retries}")
            
            response = model.generate_content(
                [Part.from_text(full_prompt)],
                generation_config=generation_config
            )
            
            # Extract response text
            if response.candidates and len(response.candidates) > 0:
                response_text = response.candidates[0].content.parts[0].text
            else:
                raise Exception("No response candidates generated")
            
            # Convert to OpenAI-like format for compatibility
            completion = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    }
                }],
                "usage": {
                    "input_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
                    "output_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
                    "total_tokens": (response.usage_metadata.total_token_count if response.usage_metadata else 0)
                }
            }
            
            print("✅ Vertex AI API call successful")
            return completion
            
        except Exception as e:
            error_str = str(e).lower()
            if "quota" in error_str or "rate" in error_str or "limit" in error_str:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"⚠️  Rate limit/quota hit. Waiting {delay} seconds before retry {attempt + 2}...")
                    time.sleep(delay)
                else:
                    print(f"❌ Max retries ({max_retries}) exceeded. Error: {e}")
                    raise
            else:
                print(f"❌ API Error: {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                else:
                    raise

def cal_cost_vertex(response_json: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Calculate cost for Vertex AI Claude API calls."""
    
    # Vertex AI Claude pricing (per 1M tokens) - approximate as of 2025
    model_cost = {
        "claude-3-5-sonnet-v2@20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku@20241022": {"input": 0.25, "output": 1.25},
        "claude-3-opus@20240229": {"input": 15.00, "output": 75.00},
        # Default fallback
        "default": {"input": 3.00, "output": 15.00}
    }
    
    input_tokens = response_json["usage"]["input_tokens"]
    output_tokens = response_json["usage"]["output_tokens"]
    
    cost_info = model_cost.get(model_name, model_cost["default"])
    
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

def print_log_cost_vertex(completion_json: Dict[str, Any], model_name: str, 
                         current_stage: str, output_dir: str, total_accumulated_cost: float) -> float:
    """Print and log cost information for Vertex AI API calls."""
    usage_info = cal_cost_vertex(completion_json, model_name)

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

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/cost_info.log", "a", encoding="utf-8") as f:
        f.write(output_text + "\n")
    
    return total_accumulated_cost

def truncate_content_if_needed(content, max_chars: int = 50000):
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
