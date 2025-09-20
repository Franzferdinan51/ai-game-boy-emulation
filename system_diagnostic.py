#!/usr/bin/env python3
"""
Diagnostic script to identify issues with the AI Game Playing System
"""
import sys
import os
import subprocess
import importlib.util

def diagnose_system():
    """Diagnose issues with the system"""
    print("=== AI Game Playing System - Diagnostic Check ===\n")
    
    # Check 1: Directory structure
    print("1. Checking directory structure...")
    required_dirs = [
        "ai-game-server",
        "ai-game-assistant"
    ]
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"   [OK] Found: {directory}")
        else:
            print(f"   [ERROR] Missing: {directory}")
            return False
    
    # Check 2: Required files
    print("\n2. Checking required files...")
    required_files = [
        "ai-game-server/src/main.py",
        "ai-game-server/src/backend/server.py",
        "ai-game-assistant/package.json",
        "ai-game-assistant/App.tsx"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   [OK] Found: {file_path}")
        else:
            print(f"   [ERROR] Missing: {file_path}")
            return False
    
    # Check 3: Python dependencies
    print("\n3. Checking Python dependencies...")
    required_packages = [
        ("flask", "Flask"),
        ("flask_cors", "Flask-CORS"),
        ("numpy", "numpy"),
        ("PIL", "Pillow"),
        ("requests", "requests")
    ]
    
    missing_packages = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"   [OK] {package_name} is available")
        except ImportError:
            print(f"   [MISSING] {package_name} is not installed")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n   To install missing packages, run:")
        print(f"   pip install {' '.join(missing_packages)}")
    
    # Check 4: Node.js and npm availability
    print("\n4. Checking Node.js and npm...")
    try:
        node_result = subprocess.run(["node", "--version"], 
                                    capture_output=True, text=True)
        if node_result.returncode == 0:
            print(f"   [OK] Node.js: {node_result.stdout.strip()}")
        else:
            print("   [ERROR] Node.js not found")
    except FileNotFoundError:
        print("   [ERROR] Node.js not found - please install Node.js")
    
    try:
        npm_result = subprocess.run(["npm", "--version"], 
                                   capture_output=True, text=True)
        if npm_result.returncode == 0:
            print(f"   [OK] npm: {npm_result.stdout.strip()}")
        else:
            print("   [ERROR] npm not found")
    except FileNotFoundError:
        print("   [ERROR] npm not found - please install Node.js")
    
    # Check 5: Check if requirements.txt exists and readable
    print("\n5. Checking requirements.txt...")
    req_file = "ai-game-server/requirements.txt"
    if os.path.exists(req_file):
        try:
            with open(req_file, 'r') as f:
                content = f.read()
                print(f"   [OK] requirements.txt found ({len(content)} characters)")
                print("   Required packages:")
                for line in content.strip().split('\n'):
                    if line and not line.startswith('#'):
                        print(f"     - {line}")
        except Exception as e:
            print(f"   [ERROR] Could not read requirements.txt: {e}")
    else:
        print(f"   [ERROR] {req_file} not found")
    
    # Check 6: Check package.json
    print("\n6. Checking package.json...")
    pkg_file = "ai-game-assistant/package.json"
    if os.path.exists(pkg_file):
        try:
            import json
            with open(pkg_file, 'r') as f:
                data = json.load(f)
                print(f"   [OK] package.json found")
                if 'scripts' in data:
                    print("   Available scripts:")
                    for script_name in data['scripts']:
                        print(f"     - npm run {script_name}")
                if 'dependencies' in data:
                    print(f"   Dependencies: {len(data['dependencies'])} packages")
                if 'devDependencies' in data:
                    print(f"   Dev Dependencies: {len(data['devDependencies'])} packages")
        except Exception as e:
            print(f"   [ERROR] Could not read package.json: {e}")
    else:
        print(f"   [ERROR] {pkg_file} not found")
    
    print("\n=== Diagnostic Check Complete ===")
    return True

if __name__ == "__main__":
    try:
        diagnose_system()
    except Exception as e:
        print(f"Unexpected error during diagnosis: {e}")
        import traceback
        traceback.print_exc()