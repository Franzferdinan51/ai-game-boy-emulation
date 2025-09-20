"""
Base class for AI API connectors
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import base64
import logging
import time
import random


class AIAPIConnector(ABC):
    """Abstract base class for AI API connectors"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = None  # Model selection - can be set by the provider manager
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_retries = 3
        self.retry_delay = 1.0  # Base delay in seconds

        # Validate API key
        if not api_key or not api_key.strip():
            self.logger.warning(f"API key is empty or None for {self.__class__.__name__}")
        elif len(api_key.strip()) < 10:
            self.logger.warning(f"API key appears to be too short for {self.__class__.__name__}")

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry logic with exponential backoff"""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt == self.max_retries - 1:
                    # Last attempt failed
                    self.logger.error(f"All {self.max_retries} retry attempts failed")
                    raise last_exception

                # Calculate delay with exponential backoff and jitter
                delay = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f} seconds: {e}")

                time.sleep(delay)

        raise last_exception  # This line should never be reached
    
    @abstractmethod
    def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
        """Get the next action from the AI based on the current game state"""
        pass
    
    def chat_with_ai(self, message: str, image_data: bytes, context: dict) -> str:
        """Chat with the AI about the current game state"""
        # Default implementation returns a generic response
        return "I'm an AI assistant. I can see the game screen and help you with your questions."

    @abstractmethod
    def get_models(self) -> List[str]:
        """Get a list of available models from the provider"""
        pass
    
    def encode_image(self, image_data: bytes) -> str:
        """Encode image data to base64 string"""
        return base64.b64encode(image_data).decode('utf-8')