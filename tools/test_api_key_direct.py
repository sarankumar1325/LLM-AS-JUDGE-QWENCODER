#!/usr/bin/env python3
"""
Simple Gemini API key validation script.
"""

import os
import sys
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_api_key_direct():
    """Test API key directly with Google's genai library."""
    print("🧪 Testing API Key Direct Connection...")
    
    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ No API key found in environment variables")
        return False
    
    print(f"🔑 API Key: {api_key[:20]}...")
    
    try:
        # Configure the API
        genai.configure(api_key=api_key)
        
        # List available models
        print("📋 Listing available models...")
        models = list(genai.list_models())
        
        if models:
            print("✅ API key is valid! Available models:")
            for model in models[:5]:  # Show first 5 models
                print(f"   • {model.name}")
            
            # Try to find Gemini 2.0 Flash models
            flash_models = [m for m in models if 'flash' in m.name.lower()]
            if flash_models:
                print("\n🚀 Gemini Flash models available:")
                for model in flash_models:
                    print(f"   • {model.name}")
            
            return True
        else:
            print("❌ No models available - API key might be invalid")
            return False
            
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        return False

def test_simple_generation():
    """Test simple text generation."""
    print("\n🤖 Testing Simple Text Generation...")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return False
    
    try:
        genai.configure(api_key=api_key)
        
        # Try with Gemini 2.0 Flash first
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("✅ Using gemini-2.0-flash-exp")
        except:
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                print("✅ Using gemini-1.5-flash (fallback)")
            except:
                model = genai.GenerativeModel('gemini-pro')
                print("✅ Using gemini-pro (fallback)")
        
        # Simple test
        response = model.generate_content("Say 'Hello World' in exactly 2 words.")
        
        if response and response.text:
            print(f"✅ Generation successful: {response.text}")
            return True
        else:
            print("❌ Generation failed - no response text")
            return False
            
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("🚀 Gemini API Key Direct Validation")
    print("=" * 60)
    
    success = True
    success &= test_api_key_direct()
    success &= test_simple_generation()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ API KEY IS WORKING!")
        print("🎉 Ready for RAG evaluation system!")
    else:
        print("❌ API key validation failed.")
        print("💡 Please check:")
        print("   1. API key is correct")
        print("   2. API key has not expired")
        print("   3. Quota is available")
        print("   4. Billing is set up (if required)")
    print("=" * 60)

if __name__ == "__main__":
    main()
