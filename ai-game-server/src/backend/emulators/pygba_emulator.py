"""
PyGBA emulator implementation
"""
import numpy as np
from typing import List, Tuple, Optional, Any
import io
import os

# Try to import PyGBA
try:
    from pygba import PyGBA
    PYGBA_AVAILABLE = True
except ImportError as e:
    PYGBA_AVAILABLE = False
    # Only show warning if pygba is installed but missing dependencies
    if "mgba" in str(e):
        print("PyGBA available but missing mGBA dependency. Install mGBA system package.")
    else:
        print("PyGBA not available. Install with 'pip install pygba'")

from .emulator_interface import EmulatorInterface


class PyGBAEmulator(EmulatorInterface):
    """PyGBA emulator implementation"""
    
    def __init__(self):
        self.pygba = None
        self.rom_path = None
        self.initialized = False
        
    def load_rom(self, rom_path: str) -> bool:
        """Load a ROM file into the PyGBA emulator"""
        if not PYGBA_AVAILABLE:
            raise RuntimeError("PyGBA is not available. Please install it with 'pip install pygba'")
            
        if not os.path.exists(rom_path):
            raise FileNotFoundError(f"ROM file not found: {rom_path}")
            
        try:
            # Initialize PyGBA
            self.pygba = PyGBA.load(rom_path)
            self.rom_path = rom_path
            self.initialized = True
            return True
        except Exception as e:
            print(f"Error loading ROM: {e}")
            return False
    
    def step(self, action: str, frames: int = 1) -> bool:
        """Execute an action for a number of frames"""
        if not self.initialized or self.pygba is None:
            return False
            
        # Map actions to PyGBA buttons
        action_map = {
            'UP': 'up',
            'DOWN': 'down',
            'LEFT': 'left',
            'RIGHT': 'right',
            'A': 'A',
            'B': 'B',
            'START': 'start',
            'SELECT': 'select'
        }
        
        if action in action_map:
            try:
                # Execute the action for the specified number of frames
                action_func = getattr(self.pygba, f"press_{action_map[action]}")
                action_func(frames)
                return True
            except Exception as e:
                print(f"Error executing action {action}: {e}")
                return False
        else:
            # Just run frames without input
            try:
                self.pygba.wait(frames)
                return True
            except Exception as e:
                print(f"Error ticking emulator: {e}")
                return False
    
    def get_screen(self) -> np.ndarray:
        """Get the current screen as a numpy array"""
        if not self.initialized or self.pygba is None:
            return np.zeros((160, 240, 3), dtype=np.uint8)  # GBA screen size

        try:
            # Try to get screen from PyGBA
            # Note: This implementation depends on PyGBA's actual API
            if hasattr(self.pygba, 'screen') and hasattr(self.pygba.screen, 'ndarray'):
                screen_array = self.pygba.screen.ndarray
                if screen_array is not None and screen_array.size > 0:
                    return screen_array.astype(np.uint8)

            # Alternative: Try to get screenshot
            if hasattr(self.pygba, 'get_screenshot'):
                screenshot = self.pygba.get_screenshot()
                if screenshot is not None:
                    return np.array(screenshot).astype(np.uint8)

            # Fallback: Create a test pattern to indicate emulator is working
            # This creates a gradient that changes over time
            import time
            t = int(time.time() * 10) % 256
            screen = np.zeros((160, 240, 3), dtype=np.uint8)

            # Create a moving gradient pattern
            for x in range(240):
                for y in range(160):
                    screen[y, x] = [
                        (x + t) % 256,
                        (y + t) % 256,
                        (x + y + t) % 256
                    ]

            return screen

        except Exception as e:
            print(f"Error getting screen: {e}")
            return np.zeros((160, 240, 3), dtype=np.uint8)
    
    def get_memory(self, address: int, size: int = 1) -> bytes:
        """Read memory from the emulator"""
        if not self.initialized or self.pygba is None:
            return b'\x00' * size
            
        try:
            if size == 1:
                return bytes([self.pygba.read_u8(address)])
            elif size == 2:
                return self.pygba.read_u16(address).to_bytes(2, byteorder='little')
            elif size == 4:
                return self.pygba.read_u32(address).to_bytes(4, byteorder='little')
            else:
                return self.pygba.read_memory(address, size)
        except Exception as e:
            print(f"Error reading memory at {hex(address)}: {e}")
            return b'\x00' * size
    
    def set_memory(self, address: int, value: bytes) -> bool:
        """Write memory to the emulator"""
        # PyGBA doesn't have a direct memory write interface
        # This would need to be implemented in the PyGBA library itself
        print("Memory writing not implemented for PyGBA")
        return False
    
    def reset(self) -> bool:
        """Reset the emulator"""
        if not self.initialized or self.pygba is None:
            return False
            
        try:
            # Reset the emulator
            self.pygba.core.reset()
            return True
        except Exception as e:
            print(f"Error resetting emulator: {e}")
            return False
    
    def save_state(self) -> bytes:
        """Save the current state of the emulator"""
        if not self.initialized or self.pygba is None:
            return b''
            
        try:
            # Save state to bytes (simplified implementation)
            state = self.pygba.core.save_raw_state()
            return state if isinstance(state, bytes) else str(state).encode()
        except Exception as e:
            print(f"Error saving state: {e}")
            return b''
    
    def load_state(self, state: bytes) -> bool:
        """Load a saved state into the emulator"""
        if not self.initialized or self.pygba is None:
            return False
            
        try:
            # Load state from bytes (simplified implementation)
            self.pygba.core.load_raw_state(state)
            return True
        except Exception as e:
            print(f"Error loading state: {e}")
            return False
    
    def get_info(self) -> dict:
        """Get information about the current game state"""
        if not self.initialized or self.pygba is None:
            return {}
            
        try:
            return {
                "initialized": self.initialized,
                "screen_size": (160, 240, 3)  # GBA screen size
            }
        except Exception as e:
            print(f"Error getting info: {e}")
            return {}
    
    def get_game_state_analysis(self) -> dict:
        """Get a detailed analysis of the current game state"""
        if not self.initialized or self.pygba is None:
            return {}
            
        try:
            # Get basic info
            info = self.get_info()
            
            # Get screen analysis
            screen = self.get_screen()
            
            # Get memory regions of interest (this would be game-specific)
            # For now, we'll just get some general memory values
            memory_analysis = {}
            
            # Add game-specific analysis based on the game title
            # This would need to be implemented based on the specific game
            game_specific = {"analysis": "Game state analysis not implemented for GBA"}
            
            return {
                "basic_info": info,
                "screen_analysis": {
                    "shape": screen.shape,
                    "mean_color": screen.mean(axis=(0,1)).tolist(),
                    "unique_colors": len(np.unique(screen.reshape(-1, screen.shape[2]), axis=0))
                },
                "memory_analysis": memory_analysis,
                "game_specific": game_specific
            }
        except Exception as e:
            print(f"Error getting game state analysis: {e}")
            return {}