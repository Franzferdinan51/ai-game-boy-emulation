"""
OpenAI Compatible AI API connector
"""
import base64
import os
import time
import requests
from typing import List, Optional, Dict, Any
import io
from PIL import Image
from .ai_api_base import AIAPIConnector
from openai import OpenAI


class OpenAICompatibleConnector(AIAPIConnector):
    """OpenAI Compatible AI API connector"""
    
    def __init__(self, api_key: str, base_url: str = None):
        super().__init__(api_key)
        # Support for LM Studio and other local providers
        self.base_url = self._validate_base_url(base_url)
        self.model = None
        self.valid_actions = {'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT'}
        self.timeout = int(os.environ.get('AI_TIMEOUT', '60'))
        self.max_retries = int(os.environ.get('AI_MAX_RETRIES', '3'))

        # Initialize client with robust error handling
        self.client = self._initialize_client()

        # Test connection during initialization
        self._test_connection()

    def _validate_base_url(self, base_url: Optional[str]) -> str:
        """Validate and normalize base URL"""
        if not base_url:
            # Check environment variables
            base_url = os.environ.get('OPENAI_ENDPOINT') or os.environ.get('AI_ENDPOINT')

        if not base_url:
            # Default to OpenAI
            return "https://api.openai.com/v1"

        # Normalize URL
        base_url = base_url.rstrip('/')

        # Common LM Studio endpoints
        if base_url in ['localhost', '127.0.0.1', 'lm-studio']:
            base_url = "http://localhost:1234/v1"
        elif 'lm-studio' in base_url and not base_url.startswith(('http://', 'https://')):
            base_url = f"http://{base_url}:1234/v1"
        elif not base_url.startswith(('http://', 'https://')):
            base_url = f"http://{base_url}"

        return base_url

    def _initialize_client(self) -> Optional[OpenAI]:
        """Initialize OpenAI client with error handling"""
        try:
            # For local providers, we might not need an API key
            if 'localhost' in self.base_url or '127.0.0.1' in self.base_url:
                api_key = self.api_key or "not-needed"
            else:
                api_key = self.api_key

            if not api_key:
                # For local providers, this is not an error
                if 'localhost' in self.base_url or '127.0.0.1' in self.base_url:
                    api_key = "not-needed"
                else:
                    self.logger.warning("No API key provided for OpenAI-compatible provider")
                    return None

            client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=self.timeout)
            self.logger.info(f"Initialized OpenAI-compatible client with URL: {self.base_url}")
            return client

        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI-compatible client: {e}")
            return None

    def _test_connection(self):
        """Test connection to the AI provider"""
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
        """Get a list of available models from the OpenAI-compatible API"""
        if not self.client:
            self.logger.warning("Cannot fetch models - client not initialized")
            return []

        try:
            response = self.client.models.list()
            models = [model.id for model in response.data]
            self.logger.info(f"Found {len(models)} models at {self.base_url}")
            return models
        except Exception as e:
            self.logger.error(f"Failed to fetch models from {self.base_url}: {e}")
            return []

    def get_next_action(self, image_bytes: bytes, goal: str, action_history: List[str]) -> str:
        """Get the next action from OpenAI-compatible API based on the current game state"""
        # Check if client is initialized
        if not self.client:
            self.logger.error("OpenAI-compatible client not initialized")
            return self._get_fallback_action(action_history)

        try:
            # Convert image to base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')

            # Create enhanced prompt with better context
            prompt = self._create_action_prompt(goal, action_history)

            self.logger.debug(f"Making request to OpenAI-compatible API - Model: {self.model}, URL: {self.base_url}")

            # Use the specified model or default to first available model
            model_to_use = self.model if self.model else (self.get_models()[0] if self.get_models() else "gpt-3.5-turbo")
            
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
                self.logger.info(f"OpenAI-compatible API returned valid action: {action}")
                return action
            else:
                self.logger.warning(f"OpenAI-compatible API returned invalid action: '{action}'. Using fallback.")
                return self._get_fallback_action(action_history)

        except Exception as e:
            self.logger.error(f"Error calling OpenAI-compatible API: {e}", exc_info=True)
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
        """Chat with the AI about the current game state"""
        if not self.client:
            return "I'm sorry, the AI service is not available right now."

        try:
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Use the specified model or default to first available model
            model_to_use = self.model if self.model else (self.get_models()[0] if self.get_models() else "gpt-3.5-turbo")
            
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
            self.logger.error(f"Error calling OpenAI-compatible API for chat: {e}", exc_info=True)
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

Please provide a helpful response based on the game screen and the user's message. You can see the current game state and provide specific advice about what's happening in the game."""