"""
Streamlit Cloud entrypoint.
Ensures `src` is importable without PYTHONPATH manipulation.
"""
import sys
import os

# Ensure the project root is in sys.path for clean imports
sys.path.insert(0, os.path.dirname(__file__))

# Delegate to the actual dashboard module
from src.interfaces.dashboard import main

main()
