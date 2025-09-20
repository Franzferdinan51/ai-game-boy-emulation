#!/usr/bin/env python3
"""
Simple verification script to test if the backend server can start properly
"""
import sys
import os

def verify_backend_startup():
    """Verify that the backend server can start without errors"""
    print("=== Backend Server Startup Verification ===\n")
    
    # Add the backend directory to the path
    backend_path = os.path.join(os.path.dirname(__file__), 'ai-game-server', 'src')
    sys.path.insert(0, backend_path)
    
    try:
        print("Attempting to import backend server components...")
        
        # Try to import the main server module
        from backend.server import app, emulators, ai_apis, action_history, game_state
        print("✅ Backend server components imported successfully")
        
        # Check that required components are present
        required_vars = [
            ('app', app),
            ('emulators', emulators),
            ('ai_apis', ai_apis),
            ('action_history', action_history),
            ('game_state', game_state)
        ]
        
        for var_name, var_value in required_vars:
            if var_value is None:
                print(f"❌ {var_name} is None")
                return False
            print(f"✅ {var_name} initialized successfully")
        
        # Check that emulators dict has required keys
        expected_emulators = ['gb', 'gba']
        for key in expected_emulators:
            if key not in emulators:
                print(f"❌ Missing emulator: {key}")
                return False
            print(f"✅ Emulator '{key}' available")
        
        # Check that game_state has required keys
        expected_game_state_keys = ['active_emulator', 'rom_loaded', 'ai_running', 'current_goal']
        for key in expected_game_state_keys:
            if key not in game_state:
                print(f"❌ Missing key in game_state: {key}")
                return False
        print("✅ Game state structure verified")
        
        # Check that action_history is a list
        if not isinstance(action_history, list):
            print("❌ action_history is not a list")
            return False
        print("✅ Action history structure verified")
        
        print("\n🎉 All backend startup verification tests passed!")
        print("The backend server should start successfully.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This usually means required dependencies are not installed.")
        print("Try running: pip install -r ai-game-server/requirements.txt")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_backend_startup()
    sys.exit(0 if success else 1)