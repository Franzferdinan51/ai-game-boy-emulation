"""
AI Game Server - Main entry point
"""
import sys
import os
import warnings

# Suppress non-critical SDL2 warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Unable to preload all dependencies for SDL2")

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.server import main

if __name__ == "__main__":
    main()