#!/usr/bin/env python3
"""
Verify Google Cloud setup for Vertex AI Claude integration.
"""

import os
import json
import sys

def verify_setup():
    """Verify all necessary setup components."""
    
    print("🔍 Verifying Google Cloud Setup for Vertex AI Claude")
    print("=" * 60)
    
    issues = []
    
    # Check environment variables
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    
    print("1. Environment Variables:")
    if project_id:
        print(f"   ✅ GOOGLE_CLOUD_PROJECT: {project_id}")
    else:
        print(f"   ❌ GOOGLE_CLOUD_PROJECT: Not set")
        issues.append("Set GOOGLE_CLOUD_PROJECT environment variable")
    
    if credentials_path:
        print(f"   ✅ GOOGLE_APPLICATION_CREDENTIALS: {credentials_path}")
        
        # Check if credentials file exists
        if os.path.exists(credentials_path):
            print(f"   ✅ Credentials file exists")
            
            # Try to parse the credentials file
            try:
                with open(credentials_path, 'r') as f:
                    creds = json.load(f)
                
                if 'type' in creds and creds['type'] == 'service_account':
                    print(f"   ✅ Valid service account file")
                    print(f"   📧 Service account: {creds.get('client_email', 'N/A')}")
                    print(f"   🆔 Project ID in file: {creds.get('project_id', 'N/A')}")
                    
                    # Check if project IDs match
                    if project_id and creds.get('project_id') != project_id:
                        print(f"   ⚠️  Warning: Project ID mismatch")
                        print(f"      Environment: {project_id}")
                        print(f"      Credentials: {creds.get('project_id')}")
                else:
                    print(f"   ❌ Invalid service account file format")
                    issues.append("Credentials file is not a valid service account key")
                    
            except json.JSONDecodeError:
                print(f"   ❌ Credentials file is not valid JSON")
                issues.append("Credentials file is corrupted or not JSON")
            except Exception as e:
                print(f"   ❌ Error reading credentials file: {e}")
                issues.append(f"Cannot read credentials file: {e}")
        else:
            print(f"   ❌ Credentials file not found")
            issues.append(f"Credentials file not found at: {credentials_path}")
    else:
        print(f"   ❌ GOOGLE_APPLICATION_CREDENTIALS: Not set")
        issues.append("Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
    
    print("\n2. Required Python Packages:")
    required_packages = [
        'google.auth',
        'google.oauth2.service_account',
        'google.auth.transport.requests',
        'requests'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            issues.append(f"Install missing package: pip install google-auth google-auth-oauthlib requests")
    
    print("\n3. Google Cloud CLI (Optional but recommended):")
    
    # Check if gcloud is installed
    import subprocess
    try:
        result = subprocess.run(['gcloud', 'version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"   ✅ gcloud CLI installed")
            
            # Check active project
            try:
                result = subprocess.run(['gcloud', 'config', 'get-value', 'project'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gcloud_project = result.stdout.strip()
                    if gcloud_project == project_id:
                        print(f"   ✅ Active project matches: {gcloud_project}")
                    else:
                        print(f"   ⚠️  Active project mismatch:")
                        print(f"      gcloud: {gcloud_project}")
                        print(f"      Environment: {project_id}")
            except:
                print(f"   ⚠️  Could not check active project")
        else:
            print(f"   ❌ gcloud CLI not working properly")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f"   ⚠️  gcloud CLI not installed (optional)")
        print(f"      Install from: https://cloud.google.com/sdk/docs/install")
    
    print("\n" + "=" * 60)
    
    if issues:
        print("❌ Issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        
        print(f"\n📋 Quick Setup Guide:")
        print(f"1. Create a service account in Google Cloud Console")
        print(f"2. Download the JSON key file")
        print(f"3. Set environment variables:")
        print(f"   export GOOGLE_CLOUD_PROJECT='your-project-id'")
        print(f"   export GOOGLE_APPLICATION_CREDENTIALS='/path/to/key.json'")
        print(f"4. Install dependencies:")
        print(f"   pip install -r requirements_vertex.txt")
        
        return False
    else:
        print("✅ All checks passed! You're ready to use Vertex AI Claude.")
        print(f"\n🧪 Next step: Run the test script")
        print(f"   python test_vertex_claude.py")
        return True

if __name__ == "__main__":
    success = verify_setup()
    sys.exit(0 if success else 1)
