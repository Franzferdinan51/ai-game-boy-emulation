"""
Abstract interface for game emulators
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any
import numpy as np
import io


class EmulatorInterface(ABC):
    """Abstract base class for game emulators"""
    
    @abstractmethod
    def load_rom(self, rom_path: str) -> bool:
        """Load a ROM file into the emulator"""
        pass
    
    @abstractmethod
    def step(self, action: str, frames: int = 1) -> bool:
        """Execute an action for a number of frames"""
        pass
    
    @abstractmethod
    def get_screen(self) -> np.ndarray:
        """Get the current screen as a numpy array"""
        pass
    
    @abstractmethod
    def get_memory(self, address: int, size: int = 1) -> bytes:
        """Read memory from the emulator"""
        pass
    
    @abstractmethod
    def set_memory(self, address: int, value: bytes) -> bool:
        """Write memory to the emulator"""
        pass
    
    @abstractmethod
    def reset(self) -> bool:
        """Reset the emulator"""
        pass
    
    @abstractmethod
    def save_state(self) -> bytes:
        """Save the current state of the emulator"""
        pass
    
    @abstractmethod
    def load_state(self, state: bytes) -> bool:
        """Load a saved state into the emulator"""
        pass
    
    @abstractmethod
    def get_info(self) -> dict:
        """Get information about the current game state"""
        pass
    
    def get_game_state_analysis(self) -> dict:
        """Get a detailed analysis of the current game state"""
        # Default implementation
        return {
            "basic_info": self.get_info(),
            "screen_analysis": {},
            "memory_analysis": {},
            "game_specific": {}
        }

    # Audio Methods
    def get_audio_buffer(self) -> np.ndarray:
        """Get current audio buffer as numpy array"""
        # Default implementation - returns empty array
        return np.array([], dtype=np.int8)

    def get_audio_info(self) -> dict:
        """Get audio configuration information"""
        # Default implementation
        return {
            "enabled": False,
            "sample_rate": 0,
            "buffer_size": 0,
            "format": "none",
            "channels": 0
        }

    def export_audio_wav(self, duration_frames: int = 60) -> bytes:
        """Export audio as WAV file bytes"""
        # Default implementation - returns empty WAV header
        return b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00>\x00\x00\x01\x00\x08\x00data\x00\x00\x00\x00'

    def set_audio_enabled(self, enabled: bool) -> bool:
        """Enable or disable audio emulation"""
        # Default implementation - audio cannot be enabled
        return False

    def set_audio_volume(self, volume: int) -> bool:
        """Set audio volume (0-100)"""
        # Default implementation - volume cannot be set
        return False