"""
NVIDIA NIM AI API connector
"""
import base64
import os
import requests
from typing import List, Optional, Dict, Any
import io
from PIL import Image
from .ai_api_base import AIAPIConnector
from openai import OpenAI


class NVIDIAAPIConnector(AIAPIConnector):
    """NVIDIA NIM AI API connector"""

    def __init__(self, api_key: str, model: str = None):
        super().__init__(api_key)
        self.api_url = "https://integrate.api.nvidia.com/v1"
        # Allow custom model selection via environment variable or parameter
        self.model = model or os.environ.get('NVIDIA_MODEL', "nvidia/llama-3.1-nemotron-70b-instruct")
        self.valid_actions = {'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT'}
        self.timeout = int(os.environ.get('AI_TIMEOUT', '60'))
        self.max_retries = int(os.environ.get('AI_MAX_RETRIES', '3'))
        self.logger.info(f"NVIDIA API initialized with model: {self.model}")

        # Initialize client with robust error handling
        self.client = self._initialize_client()

        # Test connection during initialization
        self._test_connection()

    def _initialize_client(self) -> Optional[OpenAI]:
        """Initialize OpenAI client for NVIDIA API with error handling"""
        try:
            if not self.api_key or not self.api_key.strip():
                self.logger.error("NVIDIA API key is empty or invalid")
                return None

            client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_url,
                timeout=self.timeout
            )
            self.logger.info(f"Initialized NVIDIA API client with URL: {self.api_url}")
            return client

        except Exception as e:
            self.logger.error(f"Failed to initialize NVIDIA API client: {e}")
            return None

    def _test_connection(self):
        """Test connection to the NVIDIA API"""
        if not self.client:
            self.logger.warning("Cannot test connection - no client initialized")
            return

        try:
            # Simple test request
            response = self.client.models.list()
            self.logger.info(f"Connection test successful. Available models: {len(response.data) if hasattr(response, 'data') else 'unknown'}")
        except Exception as e:
            self.logger.warning(f"Connection test failed: {e}")
            # Don't raise exception - allow the connector to work even if test fails

    def get_models(self) -> List[str]:
        """Get a list of available models from the NVIDIA NIM API"""
        if not self.client:
            self.logger.warning("Cannot fetch models - client not initialized")
            return []

        try:
            response = self.client.models.list()
            models = [model.id for model in response.data]
            self.logger.info(f"Found {len(models)} NVIDIA models")
            return models
        except Exception as e:
            self.logger.error(f"Failed to fetch models from NVIDIA NIM API: {e}")
            return []

    def get_next_action(self, image_bytes: bytes, goal: str, action_history: List[str]) -> str:
        """Get the next action from NVIDIA NIM AI based on the current game state"""
        if not self.client:
            self.logger.error("NVIDIA API client not initialized")
            return self._get_fallback_action(action_history)

        try:
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')

            prompt = self._create_action_prompt(goal, action_history)

            self.logger.debug(f"Making request to NVIDIA NIM API - Model: {self.model}")

            # Use the specified model or default to first available model
            model_to_use = self.model if self.model else (self.get_models()[0] if self.get_models() else "meta/llama3-8b-instruct")
            
            def make_request():
                return self.client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                        ]}
                    ],
                    max_tokens=10,
                    temperature=0.7,
                    timeout=self.timeout
                )

            response = self._retry_with_backoff(make_request)

            # Parse and validate response
            action = self._parse_action_response(response)
            if action in self.valid_actions:
                self.logger.info(f"NVIDIA NIM returned valid action: {action}")
                return action
            else:
                self.logger.warning(f"NVIDIA NIM returned an invalid action: '{action}'. Defaulting to SELECT.")
                return self._get_fallback_action(action_history)

        except Exception as e:
            self.logger.error(f"Error calling NVIDIA NIM API: {e}", exc_info=True)
            # Re-raise exception to trigger fallback system
            raise e

    def _create_action_prompt(self, goal: str, action_history: List[str]) -> str:
        """Create an enhanced prompt for action selection"""
        recent_actions = ', '.join(action_history[-5:]) if action_history else "none"
        return f"""You are an expert AI playing a Game Boy game. Your current objective is: "{goal}".

GAME BOY CONTROLS:
- D-PAD: UP, DOWN, LEFT, RIGHT (move character, navigate menus)
- ACTION BUTTONS: A (primary action/jump/confirm), B (secondary action/cancel/back)
- SYSTEM BUTTONS: START (pause/start game), SELECT (menu/option selection)

Recent actions: {recent_actions}.

Analyze the game screen and choose the BEST single action to progress toward your objective.
Consider game context: character position, enemies, items, menus, dialogue boxes, etc.

Your response MUST be exactly one of: UP, DOWN, LEFT, RIGHT, A, B, START, SELECT.
No explanation - just the action word."""

    def _get_system_prompt(self) -> str:
        """Get system prompt for the AI"""
        return "You are an expert AI playing a retro video game. Respond with only the action name in uppercase."

    def _parse_action_response(self, response) -> str:
        """Parse and clean action response from AI"""
        try:
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content.strip().upper()
                # Clean up common response issues
                content = content.replace('.', '').replace(',', '').strip()
                return content
            return "SELECT"
        except Exception as e:
            self.logger.error(f"Error parsing action response: {e}")
            return "SELECT"

    def _get_fallback_action(self, action_history: List[str]) -> str:
        """Get a fallback action when AI fails"""
        # Simple strategy: if no recent actions, try UP, otherwise try different action
        if not action_history:
            return "UP"

        last_action = action_history[-1]
        if last_action == "UP":
            return "RIGHT"
        elif last_action == "RIGHT":
            return "DOWN"
        elif last_action == "DOWN":
            return "LEFT"
        else:
            return "A"

    def chat_with_ai(self, user_message: str, image_bytes: bytes, context: dict) -> str:
        """Chat with the NVIDIA NIM AI about the current game state"""
        if not self.client:
            return "I'm sorry, the AI service is not available right now."

        try:
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Use the specified model or default to first available model
            model_to_use = self.model if self.model else (self.get_models()[0] if self.get_models() else "meta/llama3-8b-instruct")
            
            prompt = self._create_chat_prompt(user_message, context)

            def make_request():
                return self.client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": "You are a helpful game assistant with expertise in retro video games."},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                        ]}
                    ],
                    max_tokens=500,
                    temperature=0.7,
                    timeout=self.timeout
                )

            response = self._retry_with_backoff(make_request)
            return response.choices[0].message.content.strip()

        except Exception as e:
            self.logger.error(f"Error calling NVIDIA NIM API for chat: {e}", exc_info=True)
            return "I'm sorry, I encountered an error processing your request."

    def _create_chat_prompt(self, user_message: str, context: dict) -> str:
        """Create an enhanced chat prompt with context"""
        current_goal = context.get('current_goal', 'None')
        action_history = ', '.join(context.get('action_history', [])[-5:]) if context.get('action_history') else "none"
        game_type = context.get('game_type', 'Unknown')

        return f"""User Message: {user_message}

Game Context:
- Current Goal: {current_goal}
- Recent Actions: {action_history}
- Game Type: {game_type}

Please provide a helpful response based on the game screen and the user's message. 
You can see the current game screen, so you can provide specific advice about what's happening in the game."""