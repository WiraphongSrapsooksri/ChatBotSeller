# test_imports.py
import sys
import os
from zoneinfo import ZoneInfo
import tensorflow as tf
import fastapi
import pydantic
import pythainlp

def test_imports():
    print("\nTesting imports:")
    print("-" * 50)
    
    # Test TensorFlow
    print(f"TensorFlow version: {tf.__version__}")
    
    # Test FastAPI
    print(f"FastAPI version: {fastapi.__version__}")
    
    # Test Pydantic
    print(f"Pydantic version: {pydantic.__version__}")
    
    # Test PyThaiNLP
    print(f"PyThaiNLP version: {pythainlp.__version__}")
    
    # Test timezone
    try:
        tz = ZoneInfo("Asia/Bangkok")
        print("Timezone test: Success")
    except Exception as e:
        print(f"Timezone test failed: {str(e)}")
    
    print("\nPython Information:")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    print("\nEnvironment:")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")

if __name__ == "__main__":
    test_imports()