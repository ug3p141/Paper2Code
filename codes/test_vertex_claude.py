#!/usr/bin/env python3
"""
Test script for Vertex AI Claude integration.
"""

import os
import sys
from vertex_utils import (initialize_vertex_ai, get_claude_model_names, vertex_claude_api_call, 
                         list_available_models)

def test_vertex_claude():
    """Test Claude models on Vertex AI."""
    
    print("🧪 Testing Vertex AI Claude Integration")
    print("=" * 50)
    
    # Check environment variables
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    
    print(f"🔍 Environment Check:")
    print(f"   GOOGLE_CLOUD_PROJECT: {project_id or '❌ Not set'}")
    print(f"   GOOGLE_APPLICATION_CREDENTIALS: {credentials_path or '❌ Not set'}")
    
    if credentials_path:
        if os.path.exists(credentials_path):
            print(f"   ✅ Credentials file exists")
        else:
            print(f"   ❌ Credentials file not found at: {credentials_path}")
    
    if not project_id:
        print("❌ Error: GOOGLE_CLOUD_PROJECT environment variable not set")
        print("Please set: export GOOGLE_CLOUD_PROJECT='your-project-id'")
        return False
    
    if not credentials_path or not os.path.exists(credentials_path):
        print("❌ Error: GOOGLE_APPLICATION_CREDENTIALS not set or file doesn't exist")
        print("Please either:")
        print("1. Set: export GOOGLE_APPLICATION_CREDENTIALS='path/to/your/service-account-key.json'")
        print("2. Or run: gcloud auth application-default login")
        return False
    
    # Initialize Vertex AI
    try:
        project_id, location, credentials = initialize_vertex_ai()
        print(f"✅ Successfully initialized Vertex AI")
        print(f"   Project: {project_id}")
        print(f"   Location: {location}")
    except Exception as e:
        print(f"❌ Failed to initialize Vertex AI: {e}")
        return False
    
    # List available models
    print("\n📋 Available Models:")
    models = list_available_models()
    
    # Test Claude Sonnet 4
    test_model = "claude-sonnet-4"
    print(f"\n🧪 Testing model: {test_model}")
    print(f"   This will use the full model name with version suffix")
    
    test_messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Respond concisely and mention that you are Claude running on Vertex AI."},
        {"role": "user", "content": "Hello! Can you briefly explain what you are and confirm you're working correctly? Please confirm you're running on Vertex AI."}
    ]
    
    try:
        response = vertex_claude_api_call(
            project_id=project_id,
            location=location,
            credentials=credentials,
            model_name=test_model,
            messages=test_messages,
            max_tokens=200,
            temperature=0.7
        )
        
        print("✅ API call successful!")
        print(f"📝 Response: {response['choices'][0]['message']['content']}")
        print(f"📊 Tokens - Input: {response['usage']['input_tokens']}, Output: {response['usage']['output_tokens']}")
        
        return True
        
    except Exception as e:
        print(f"❌ API call failed: {e}")
        print("\n🔍 Troubleshooting tips:")
        print("1. Ensure you have enabled the Vertex AI API in your project")
        print("2. Check that Claude models are available in your region")
        print("3. Verify your authentication credentials have proper scopes")
        print("4. Check quotas in Google Cloud Console")
        print("5. Ensure your project has access to Anthropic models on Vertex AI")
        print("6. Try a different region (us-central1, us-east4, europe-west4)")
        
        # Check if it's a specific error we can help with
        error_str = str(e).lower()
        if "anthropic_version" in error_str:
            print("\n💡 Anthropic version error: This should be fixed now with vertex-2023-10-16")
        elif "invalid_request_error" in error_str:
            print("\n💡 Invalid request: Check the model name and payload format")
        elif "permission" in error_str or "access" in error_str:
            print("\n💡 Permission error: Ensure your service account has Vertex AI access")
        elif "quota" in error_str or "rate" in error_str:
            print("\n💡 Quota/rate limit: Check your Vertex AI quotas in Cloud Console")
        
        return False

if __name__ == "__main__":
    success = test_vertex_claude()
    sys.exit(0 if success else 1)