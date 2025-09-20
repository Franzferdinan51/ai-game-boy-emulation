#!/usr/bin/env python3
"""
Simple verification script to check if the frontend dependencies are installed
"""
import sys
import os
import subprocess

def verify_frontend_setup():
    """Verify that the frontend is properly set up"""
    print("=== Frontend Setup Verification ===\n")
    
    # Check if we're in the right directory
    frontend_path = os.path.join(os.path.dirname(__file__), 'ai-game-assistant')
    
    if not os.path.exists(frontend_path):
        print("❌ Cannot find ai-game-assistant directory")
        print("Please run this script from the root directory containing ai-game-assistant")
        return False
    
    if not os.path.exists(os.path.join(frontend_path, 'package.json')):
        print("❌ Cannot find package.json in ai-game-assistant")
        return False
    
    print("✅ Found ai-game-assistant directory and package.json")
    
    # Check if node_modules exists
    node_modules_path = os.path.join(frontend_path, 'node_modules')
    if os.path.exists(node_modules_path):
        print("✅ node_modules directory exists")
    else:
        print("⚠️  node_modules directory not found - you may need to run 'npm install'")
    
    # Try to check if npm is available
    try:
        result = subprocess.run(['npm', '--version'], 
                              capture_output=True, 
                              text=True, 
                              cwd=frontend_path)
        if result.returncode == 0:
            print(f"✅ npm is available (version {result.stdout.strip()})")
        else:
            print("❌ npm is not available - please install Node.js")
            return False
    except FileNotFoundError:
        print("❌ npm is not available - please install Node.js")
        return False
    
    # Check for key dependencies in package.json
    try:
        import json
        with open(os.path.join(frontend_path, 'package.json'), 'r') as f:
            package_json = json.load(f)
        
        required_deps = ['react', '@google/genai']
        found_deps = []
        
        if 'dependencies' in package_json:
            for dep in required_deps:
                if dep in package_json['dependencies']:
                    found_deps.append(dep)
                    print(f"✅ Found dependency: {dep}")
                else:
                    print(f"⚠️  Missing dependency: {dep}")
        
        if 'scripts' in package_json and 'dev' in package_json['scripts']:
            print("✅ Found 'dev' script in package.json")
        else:
            print("❌ Missing 'dev' script in package.json")
            
    except Exception as e:
        print(f"⚠️  Could not read package.json: {e}")
    
    print("\n🎉 Frontend setup verification completed!")
    print("You should be able to run 'npm install' and 'npm run dev' in the ai-game-assistant directory")
    return True

if __name__ == "__main__":
    success = verify_frontend_setup()
    sys.exit(0 if success else 1)