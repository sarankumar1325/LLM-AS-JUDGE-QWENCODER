#!/usr/bin/env python3
"""
Test Gemini 1.5 Pro API availability and quota.
"""

import os
import sys
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_gemini_models():
    """Test different Gemini models for quota availability."""
    print("🧪 Testing Gemini Models for Quota Availability...")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ No API key found")
        return False
    
    genai.configure(api_key=api_key)
    
    # Models to test in order of preference
    models_to_test = [
        "gemini-1.5-flash",      # Usually has the best free quota
        "gemini-1.5-pro",        # Good balance of capability and quota
        "gemini-1.5-flash-002",  # Alternative flash model
        "gemini-pro",            # Fallback option
        "gemini-2.0-flash-exp"   # Original choice (likely quota limited)
    ]
    
    for model_name in models_to_test:
        print(f"\n🤖 Testing {model_name}...")
        
        try:
            model = genai.GenerativeModel(model_name)
            
            # Simple test prompt
            response = model.generate_content(
                "Explain what RAG (Retrieval-Augmented Generation) is in exactly one sentence.",
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 100
                }
            )
            
            if response and response.text:
                print(f"✅ {model_name} WORKING!")
                print(f"📝 Response: {response.text}")
                print(f"🎯 RECOMMENDED MODEL: {model_name}")
                return model_name
            else:
                print(f"❌ {model_name} - No response")
                
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                print(f"⚠️ {model_name} - Quota exceeded")
            elif "404" in error_msg or "not found" in error_msg.lower():
                print(f"❌ {model_name} - Model not available")
            else:
                print(f"❌ {model_name} - Error: {e}")
    
    print(f"\n❌ All models failed - need to wait for quota reset")
    return None

def update_system_config(working_model):
    """Update system configuration with working model."""
    if not working_model:
        return False
    
    print(f"\n🔧 Updating system configuration to use {working_model}...")
    
    try:
        # Add src to path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root / "src"))
        
        from src.models.gemini_client import GeminiClient
        
        # Test the client with the new model
        client = GeminiClient(model_name=working_model)
        print(f"✅ Client configured with {working_model}")
        
        # Test connection
        result = client.test_connection()
        if result:
            print("✅ Connection test successful!")
            return True
        else:
            print("❌ Connection test failed")
            return False
            
    except Exception as e:
        print(f"❌ System configuration failed: {e}")
        return False

def main():
    """Main function."""
    print("🚀 Gemini Model Quota Testing & Configuration")
    print("=" * 60)
    
    # Test models for availability
    working_model = test_gemini_models()
    
    if working_model:
        # Update system configuration
        success = update_system_config(working_model)
        
        if success:
            print("\n" + "=" * 60)
            print("✅ SUCCESS!")
            print(f"🎯 Using model: {working_model}")
            print("🚀 Ready to launch Streamlit app!")
            print("📝 Run: python run_streamlit.py")
            print("=" * 60)
        else:
            print("\n❌ Configuration update failed")
    else:
        print("\n" + "=" * 60)
        print("⏳ All models currently quota-limited")
        print("💡 Solutions:")
        print("1. Wait for quota reset (usually resets hourly)")
        print("2. Check billing settings at: https://console.cloud.google.com/")
        print("3. Try again in 1-2 hours")
        print("=" * 60)

if __name__ == "__main__":
    main()
