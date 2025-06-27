import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any
import requests
from google.auth import default
from google.auth.transport.requests import Request
from google.oauth2 import service_account

def initialize_vertex_ai(project_id: str = None, location: str = "europe-west1"):
    """Initialize Vertex AI with project credentials."""
    if project_id is None:
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise ValueError("Please set GOOGLE_CLOUD_PROJECT environment variable or pass project_id")
    
    # Get credentials for API calls with proper scopes
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    
    if credentials_path and os.path.exists(credentials_path):
        # Load from service account file
        from google.oauth2 import service_account
        
        # Define the required scopes for Vertex AI
        scopes = [
            'https://www.googleapis.com/auth/cloud-platform',
            'https://www.googleapis.com/auth/cloud-platform.read-only'
        ]
        
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=scopes
        )
        print(f"✅ Loaded credentials from: {credentials_path}")
    else:
        # Try default credentials (for environments like Colab, Cloud Shell)
        try:
            credentials, detected_project = default(scopes=[
                'https://www.googleapis.com/auth/cloud-platform'
            ])
            if not project_id and detected_project:
                project_id = detected_project
            print(f"✅ Using default credentials")
        except Exception as e:
            raise ValueError(
                f"Failed to load credentials. Please either:\n"
                f"1. Set GOOGLE_APPLICATION_CREDENTIALS to your service account key file\n"
                f"2. Run 'gcloud auth application-default login'\n"
                f"Error: {e}"
            )
    
    # Refresh credentials to ensure they're valid
    if not credentials.valid:
        credentials.refresh(Request())
    
    print(f"✅ Vertex AI initialized for project: {project_id}, location: {location}")
    return project_id, location, credentials

def get_claude_model_names():
    """Get available Claude model names for Vertex AI."""
    # Available Claude models in Vertex AI as of June 2025
    available_models = {
        # Claude 4 models (latest)
        "claude-opus-4": "claude-opus-4@20250514",
        "claude-sonnet-4": "claude-sonnet-4@20250514", 
        
        # Claude 3.7 models
        "claude-3-7-sonnet": "claude-3-7-sonnet@20250219",
        
        # Claude 3.5 models  
        "claude-3-5-sonnet": "claude-3-5-sonnet@20241022",
        "claude-3-5-haiku": "claude-3-5-haiku@20241022",
        
        # Claude 3 models
        "claude-3-opus": "claude-3-opus@20240229",
        "claude-3-sonnet": "claude-3-sonnet@20240229",
        "claude-3-haiku": "claude-3-haiku@20240307",
    }
    return available_models

def vertex_claude_api_call(project_id: str, location: str, credentials, model_name: str, 
                          messages: List[Dict[str, str]], max_tokens: int = 4096, 
                          temperature: float = 0.7, max_retries: int = 5, base_delay: int = 60):
    """Make API call to Claude via Vertex AI using the correct Anthropic API endpoint."""
    
    # Get the full model name with version
    available_models = get_claude_model_names()
    if model_name in available_models:
        full_model_name = available_models[model_name]
    else:
        # If it's already a full model name, use it directly
        full_model_name = model_name
    
    # Convert messages to Anthropic format
    system_message = ""
    formatted_messages = []
    
    for msg in messages:
        if msg['role'] == 'system':
            system_message = msg['content']
        elif msg['role'] == 'user':
            formatted_messages.append({"role": "user", "content": msg['content']})
        elif msg['role'] == 'assistant':
            formatted_messages.append({"role": "assistant", "content": msg['content']})
    
    # Prepare the request payload
    payload = {
        "anthropic_version": "vertex-2023-10-16",  # Required for Vertex AI
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": formatted_messages
    }
    
    if system_message:
        payload["system"] = system_message
    
    # Vertex AI endpoint for Anthropic models
    url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/anthropic/models/{full_model_name}:rawPredict"
    
    headers = {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json"
        # Note: anthropic_version goes in body for Vertex AI, not headers
    }
    
    for attempt in range(max_retries):
        try:
            print(f"🔄 Vertex AI Claude API call attempt {attempt + 1}/{max_retries}")
            print(f"🤖 Using model: {full_model_name}")
            
            # Refresh credentials if needed
            if not credentials.valid:
                credentials.refresh(Request())
                headers["Authorization"] = f"Bearer {credentials.token}"
            
            response = requests.post(url, headers=headers, json=payload, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract response content
                if 'content' in result and len(result['content']) > 0:
                    response_text = result['content'][0]['text']
                else:
                    raise Exception("No content in response")
                
                # Extract usage information
                usage = result.get('usage', {})
                input_tokens = usage.get('input_tokens', 0)
                output_tokens = usage.get('output_tokens', 0)
                
                # Convert to OpenAI-like format for compatibility
                completion = {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        }
                    }],
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens
                    }
                }
                
                print("✅ Vertex AI Claude API call successful")
                return completion
                
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                print(f"❌ API Error: {error_msg}")
                
                if response.status_code in [429, 503, 500]:  # Rate limit or server errors
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"⚠️  Rate limit/server error. Waiting {delay} seconds before retry {attempt + 2}...")
                        time.sleep(delay)
                        continue
                
                raise Exception(error_msg)
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"⚠️  Request timeout. Waiting {delay} seconds before retry {attempt + 2}...")
                time.sleep(delay)
            else:
                raise Exception("Request timed out after all retries")
                
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
    
    # Vertex AI Claude pricing (per 1M tokens) - as of June 2025
    model_cost = {
        # Claude 4 models
        "claude-opus-4@20250514": {"input": 15.00, "output": 75.00},
        "claude-sonnet-4@20250514": {"input": 3.00, "output": 15.00},
        
        # Claude 3.7 models
        "claude-3-7-sonnet@20250219": {"input": 3.00, "output": 15.00},
        
        # Claude 3.5 models
        "claude-3-5-sonnet@20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku@20241022": {"input": 0.25, "output": 1.25},
        
        # Claude 3 models
        "claude-3-opus@20240229": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet@20240229": {"input": 3.00, "output": 15.00},
        "claude-3-haiku@20240307": {"input": 0.25, "output": 1.25},
        
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

def truncate_content_if_needed(content, max_chars: int = 80000):
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

def list_available_models():
    """List all available Claude models."""
    models = get_claude_model_names()
    print("📋 Available Claude Models on Vertex AI:")
    print("=" * 50)
    for short_name, full_name in models.items():
        print(f"🤖 {short_name:20} -> {full_name}")
    print("=" * 50)
    return models