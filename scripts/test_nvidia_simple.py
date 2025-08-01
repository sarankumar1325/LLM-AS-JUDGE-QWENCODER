"""
Simple test script to test NVIDIA embedder connection only.
"""

import os
import sys
from pathlib import Path

# Add both src and root to path for proper imports
current_dir = Path(__file__).parent
root_dir = current_dir.parent
src_dir = root_dir / "src"
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(src_dir))

def test_basic_imports():
    """Test basic imports."""
    try:
        from config.settings import settings
        print(f"✓ Settings imported: {settings.NVIDIA_API_KEY[:10]}..." if settings.NVIDIA_API_KEY else "✗ No NVIDIA API key")
        
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        print("✓ Logger imported successfully")
        
        from embeddings.nvidia_embedder import NVIDIAEmbedder
        print("✓ NVIDIAEmbedder imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_nvidia_connection():
    """Test NVIDIA API connection."""
    try:
        from embeddings.nvidia_embedder import NVIDIAEmbedder
        
        print("Initializing NVIDIA embedder...")
        embedder = NVIDIAEmbedder()
        
        print("Testing API connection...")
        success = embedder.test_connection()
        
        if success:
            print("✓ NVIDIA API connection successful!")
            return True
        else:
            print("✗ NVIDIA API connection failed!")
            return False
            
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("=== Simple NVIDIA Embedder Test ===")
    
    print("\n1. Testing imports...")
    if not test_basic_imports():
        return 1
    
    print("\n2. Testing NVIDIA API connection...")
    if not test_nvidia_connection():
        return 1
    
    print("\n✅ All tests passed!")
    return 0

if __name__ == "__main__":
    exit(main())
