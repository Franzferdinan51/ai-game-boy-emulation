#!/usr/bin/env python3
"""
Final verification script for the complete AI Game Playing System
This script verifies that all components are properly installed and configured
"""
import sys
import os
import subprocess

def verify_complete_setup():
    """Verify that the complete setup is working correctly"""
    print("=== AI Game Playing System - Final Verification ===\n")
    
    # Check 1: Directory structure
    print("Checking directory structure...")
    required_dirs = [
        "ai-game-server",
        "ai-game-assistant"
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"[FAIL] Missing directory: {directory}")
            return False
        print(f"[OK] Found directory: {directory}")
    
    # Check 2: Backend server files
    print("\nChecking backend server files...")
    backend_files = [
        "ai-game-server/src/backend/server.py",
        "ai-game-server/src/backend/emulators/__init__.py",
        "ai-game-server/src/backend/ai_apis/__init__.py"
    ]
    
    for file_path in backend_files:
        if not os.path.exists(file_path):
            print(f"[FAIL] Missing file: {file_path}")
            return False
        print(f"[OK] Found file: {file_path}")
    
    # Check 3: Frontend files
    print("\nChecking frontend files...")
    frontend_files = [
        "ai-game-assistant/App.tsx",
        "ai-game-assistant/components/AIPanel.tsx",
        "ai-game-assistant/components/SettingsModal.tsx"
    ]
    
    for file_path in frontend_files:
        if not os.path.exists(file_path):
            print(f"[FAIL] Missing file: {file_path}")
            return False
        print(f"[OK] Found file: {file_path}")
    
    # Check 4: Python dependencies
    print("\nChecking Python dependencies...")
    try:
        import flask
        import flask_cors
        import numpy
        import PIL
        import requests
        print("[OK] All required Python packages are installed")
    except ImportError as e:
        print(f"[FAIL] Missing Python package: {e}")
        print("Please run: pip install -r ai-game-server/requirements.txt")
        return False
    
    # Check 5: Node.js dependencies
    print("\nChecking Node.js dependencies...")
    if os.path.exists("ai-game-assistant/node_modules"):
        print("[OK] Node.js dependencies are installed")
    else:
        print("[WARN] Node.js dependencies not found (will be installed on first run)")
    
    # Check 6: Startup scripts
    print("\nChecking startup scripts...")
    startup_scripts = [
        "start_system.bat",
        "start_system.ps1",
        "start_system.sh"
    ]
    
    for script in startup_scripts:
        if os.path.exists(script):
            print(f"[OK] Found startup script: {script}")
        else:
            print(f"[FAIL] Missing startup script: {script}")
            return False
    
    # Check 7: Installation scripts
    print("\nChecking installation scripts...")
    install_scripts = [
        "install_dependencies.bat",
        "install_dependencies.ps1",
        "install_dependencies.sh"
    ]
    
    for script in install_scripts:
        if os.path.exists(script):
            print(f"[OK] Found installation script: {script}")
        else:
            print(f"[FAIL] Missing installation script: {script}")
            return False
    
    # Check 8: Documentation
    print("\nChecking documentation files...")
    docs = [
        "STARTUP_GUIDE.md",
        "SETUP_COMPLETE.md"
    ]
    
    for doc in docs:
        if os.path.exists(doc):
            print(f"[OK] Found documentation: {doc}")
        else:
            print(f"[FAIL] Missing documentation: {doc}")
            return False
    
    print("\n[SUCCESS] All verification checks completed!")
    print("\nYour AI Game Playing System is properly set up and ready to use.")
    print("\nNext steps:")
    print("1. Set your API keys as environment variables")
    print("2. Run the system using one of the startup scripts:")
    print("   - start_system.bat (Windows)")
    print("   - start_system.ps1 (PowerShell)")
    print("   - start_system.sh (Unix/Linux/Mac)")
    print("\nAccess the application at: http://localhost:5173")
    
    return True

if __name__ == "__main__":
    success = verify_complete_setup()
    sys.exit(0 if success else 1)