"""
Gemini AI API connector
"""
import os
import requests
import json
from typing import List
from .ai_api_base import AIAPIConnector

try:
    import google.generativeai as genai
except ImportError:
    genai = None


class GeminiAPIConnector(AIAPIConnector):
    """Gemini AI API connector"""

    def __init__(self, api_key: str, model: str = None):
        super().__init__(api_key)
        # Allow custom model selection via environment variable or parameter
        self.model = model or os.environ.get('GEMINI_MODEL', "models/gemini-1.5-pro-latest")
        self.valid_actions = {'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT'}
        if genai:
            genai.configure(api_key=self.api_key)
        self.logger.info(f"Gemini API initialized with model: {self.model}")

    def get_models(self) -> List[str]:
        """Get a list of available models from the Gemini API"""
        if not genai:
            self.logger.warning("google.generativeai library not found. Falling back to hardcoded list.")
            return ["models/gemini-1.5-pro-latest", "models/gemini-pro-vision", "models/gemini-pro"]

        # Only validate API key, don't call list_models during startup
        if not self.api_key or not self.api_key.strip():
            self.logger.warning("Gemini API key not configured")
            return ["models/gemini-1.5-pro-latest", "models/gemini-pro-vision", "models/gemini-pro"]

        # Return hardcoded list during startup to avoid API calls
        return ["models/gemini-1.5-pro-latest", "models/gemini-pro-vision", "models/gemini-pro"]
    
    def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
        """Get the next action from Gemini AI based on the current game state"""
        # Check API key
        if not self.api_key or not self.api_key.strip():
            self.logger.error("Gemini API key is empty or invalid")
            return "SELECT"

        try:
            # Encode image to base64
            image_base64 = self.encode_image(image_data)
            self.logger.info(f"Processing AI request for goal: '{goal}' with history: {history[-5:]}")

            # Create the prompt with detailed control instructions
            prompt = f"""You are an expert AI playing a Game Boy game. Your current objective is: "{goal}".

GAME BOY CONTROLS:
- D-PAD: UP, DOWN, LEFT, RIGHT (move character, navigate menus)
- ACTION BUTTONS: A (primary action/jump/confirm), B (secondary action/cancel/back)
- SYSTEM BUTTONS: START (pause/start game), SELECT (menu/option selection)

Recent actions: {', '.join(history[-5:])} if history else 'None yet'.

Analyze the game screen and choose the BEST single action to progress toward your objective.
Consider game context: character position, enemies, items, menus, dialogue boxes, etc.

Your response MUST be exactly one of: UP, DOWN, LEFT, RIGHT, A, B, START, SELECT.
No explanation - just the action word."""

            # Use the specified model or default to models/gemini-1.5-pro-latest
            model_to_use = self.model if self.model else "models/gemini-1.5-pro-latest"
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_to_use}:generateContent"

            # Prepare the request
            headers = {
                "Content-Type": "application/json"
            }

            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": image_base64
                                }
                            },
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            }

            # Make the request with timeout and retry logic
            def make_request():
                self.logger.debug("Making request to Gemini API")
                response = requests.post(
                    f"{api_url}?key={self.api_key}",
                    headers=headers,
                    json=data,
                    timeout=30  # 30 second timeout
                )
                response.raise_for_status()
                return response

            response = self._retry_with_backoff(make_request)

            # Parse the response
            result = response.json()
            self.logger.debug("Gemini API response received successfully")

            # Extract the action from the response
            if "candidates" in result and len(result["candidates"]) > 0:
                content = result["candidates"][0]["content"]
                if "parts" in content and len(content["parts"]) > 0:
                    action = content["parts"][0]["text"].strip().upper()

                    # Validate the action
                    if action in self.valid_actions:
                        self.logger.info(f"Gemini returned valid action: {action}")
                        return action
                    else:
                        self.logger.warning(f"Gemini returned an invalid action: '{action}'. Defaulting to SELECT.")
                        return "SELECT"

            # Default action if we can't parse the response
            self.logger.warning("Could not parse action from Gemini response. Defaulting to SELECT.")
            self.logger.debug(f"Response structure: {result}")
            return "SELECT"

        except requests.exceptions.Timeout:
            self.logger.error("Gemini API request timed out")
            return "SELECT"
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Gemini API request failed: {e}")
            # Re-raise exception to trigger fallback system
            raise e
        except Exception as e:
            self.logger.error(f"Unexpected error calling Gemini API: {e}", exc_info=True)
            self.logger.error(f"Error type: {type(e).__name__}")
            # Re-raise exception to trigger fallback system
            raise e
    
    def chat_with_ai(self, message: str, image_data: bytes, context: dict) -> str:
        """Chat with the Gemini AI about the current game state"""
        try:
            # Encode image to base64
            image_base64 = self.encode_image(image_data)
            
            # Create the prompt with context
            prompt = f"""You are an expert AI assistant helping a user play a retro video game. 
The user has sent you a message: "{message}"

Context information:
- Current goal: {context.get('current_goal', 'None')}
- Recent actions taken: {', '.join(context.get('action_history', [])[-10:])}
- Game type: {context.get('game_type', 'Unknown')}

Please provide a helpful response based on the game screen and the user's message. 
You can see the current game screen, so you can provide specific advice about what's happening in the game."""

            # Use the specified model or default to models/gemini-1.5-pro-latest
            model_to_use = self.model if self.model else "models/gemini-1.5-pro-latest"
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_to_use}:generateContent"

            # Prepare the request
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": image_base64
                                }
                            },
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            }
            
            # Make the request
            response = requests.post(
                f"{api_url}?key={self.api_key}",
                headers=headers,
                json=data
            )
            
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Extract the response text
            if "candidates" in result and len(result["candidates"]) > 0:
                content = result["candidates"][0]["content"]
                if "parts" in content and len(content["parts"]) > 0:
                    return content["parts"][0]["text"].strip()
            
            return "I'm sorry, I couldn't process your request at the moment."
            
        except Exception as e:
            print(f"Error calling Gemini API for chat: {e}")
            return "I'm sorry, I encountered an error processing your request."