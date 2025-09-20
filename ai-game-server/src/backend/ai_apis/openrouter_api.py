"""
OpenRouter AI API connector
"""
import os
import requests
import json
from typing import List, Dict
from .ai_api_base import AIAPIConnector


class OpenRouterAPIConnector(AIAPIConnector):
    """OpenRouter AI API connector"""

    def __init__(self, api_key: str, model: str = None):
        super().__init__(api_key)
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        # Allow custom model selection via environment variable or parameter
        self.model = model or os.environ.get('OPENROUTER_MODEL', "anthropic/claude-3.5-sonnet")
        self.valid_actions = {'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT'}
        self.logger.info(f"OpenRouter API initialized with model: {self.model}")
    
    def get_models(self) -> List[str]:
        """Get a list of available models from the OpenRouter API"""
        try:
            response = requests.get("https://openrouter.ai/api/v1/models")
            response.raise_for_status()
            models = response.json()['data']
            return [model['id'] for model in models]
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to fetch models from OpenRouter API: {e}")
            return []
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while fetching models from OpenRouter: {e}")
            return []

    def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
        """Get the next action from OpenRouter AI based on the current game state"""
        # Check API key
        if not self.api_key or not self.api_key.strip():
            self.logger.error("OpenRouter API key is empty or invalid")
            return "SELECT"

        try:
            # Encode image to base64
            image_base64 = self.encode_image(image_data)

            # Create the prompt
            prompt = f"""You are an expert AI playing a Game Boy game. Your current objective is: "{goal}".

GAME BOY CONTROLS:
- D-PAD: UP, DOWN, LEFT, RIGHT (move character, navigate menus)
- ACTION BUTTONS: A (primary action/jump/confirm), B (secondary action/cancel/back)
- SYSTEM BUTTONS: START (pause/start game), SELECT (menu/option selection)

Recent actions: {', '.join(history[-5:])}.

Analyze the game screen and choose the BEST single action to progress toward your objective.
Consider game context: character position, enemies, items, menus, dialogue boxes, etc.

Your response MUST be exactly one of: UP, DOWN, LEFT, RIGHT, A, B, START, SELECT.
No explanation - just the action word."""

            # Prepare the request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/your-repo/ai-game-server",  # Required by OpenRouter
                "X-Title": "AI Game Server"
            }

            # Use the specified model or default to openai/gpt-4-vision-preview
            model_to_use = self.model if self.model else "openai/gpt-4-vision-preview"
            
            data = {
                "model": model_to_use,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "max_tokens": 10,
                "temperature": 0.7
            }

            # Make the request with timeout and retry logic
            def make_request():
                self.logger.debug("Making request to OpenRouter API")
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=data,
                    timeout=30  # 30 second timeout
                )
                response.raise_for_status()
                return response

            response = self._retry_with_backoff(make_request)

            # Parse the response
            result = response.json()
            self.logger.debug("OpenRouter API response received successfully")

            # Extract the action from the response
            if "choices" in result and len(result["choices"]) > 0:
                action = result["choices"][0]["message"]["content"].strip().upper()

                # Validate the action
                if action in self.valid_actions:
                    self.logger.info(f"OpenRouter returned valid action: {action}")
                    return action
                else:
                    self.logger.warning(f"OpenRouter returned an invalid action: '{action}'. Defaulting to SELECT.")
                    return "SELECT"

            # Default action if we can't parse the response
            self.logger.warning("Could not parse action from OpenRouter response. Defaulting to SELECT.")
            self.logger.debug(f"Response structure: {result}")
            return "SELECT"

        except requests.exceptions.Timeout:
            self.logger.error("OpenRouter API request timed out")
            return "SELECT"
        except requests.exceptions.RequestException as e:
            self.logger.error(f"OpenRouter API request failed: {e}")
            # Re-raise exception to trigger fallback system
            raise e
        except Exception as e:
            self.logger.error(f"Unexpected error calling OpenRouter API: {e}")
            # Return a default safe action
            return "SELECT"
    
    def chat_with_ai(self, message: str, image_data: bytes, context: Dict) -> str:
        """Chat with the OpenRouter AI about the current game state"""
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

            # Prepare the request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Use the specified model or default to openai/gpt-4-vision-preview
            model_to_use = self.model if self.model else "openai/gpt-4-vision-preview"
            
            data = {
                "model": model_to_use,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "max_tokens": 500
            }
            
            # Make the request
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data
            )
            
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Extract the response text
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            
            return "I'm sorry, I couldn't process your request at the moment."
            
        except Exception as e:
            print(f"Error calling OpenRouter API for chat: {e}")
            return "I'm sorry, I encountered an error processing your request."