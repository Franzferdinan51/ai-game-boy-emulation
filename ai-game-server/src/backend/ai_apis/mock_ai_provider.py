"""
Mock AI Provider - Simple rule-based AI that requires no API keys
"""
import random
import time
from typing import List, Optional
from .ai_api_base import AIAPIConnector


class MockAIProvider(AIAPIConnector):
    """Mock AI provider that generates simple, rule-based game actions"""

    def __init__(self, api_key: str = "mock-key-not-needed"):
        super().__init__(api_key)
        self.logger.info("Mock AI Provider initialized - no API key required")
        self.valid_actions = {'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT'}
        self.action_patterns = {
            'explore': ['UP', 'RIGHT', 'DOWN', 'LEFT'],
            'interact': ['A', 'B', 'START'],
            'navigate': ['RIGHT', 'UP', 'LEFT', 'DOWN', 'A'],
            'default': ['UP', 'RIGHT', 'DOWN', 'LEFT', 'A', 'B']
        }
        self.last_actions = []

    def get_models(self) -> List[str]:
        """Get available models (mock provider has virtual models)"""
        return [
            "mock-basic-v1",
            "mock-adaptive-v2",
            "mock-explorer-v3"
        ]

    def get_next_action(self, image_data: bytes, goal: str, history: List[str]) -> str:
        """Get the next action using simple rule-based logic"""
        try:
            # Simple goal-based action selection
            goal_lower = goal.lower() if goal else ""

            # Determine action pattern based on goal
            if any(word in goal_lower for word in ['explore', 'move', 'go', 'walk']):
                pattern = self.action_patterns['explore']
            elif any(word in goal_lower for word in ['talk', 'interact', 'use', 'open', 'select']):
                pattern = self.action_patterns['interact']
            elif any(word in goal_lower for word in ['navigate', 'find', 'reach', 'get']):
                pattern = self.action_patterns['navigate']
            else:
                pattern = self.action_patterns['default']

            # Avoid repeating the same action too many times
            if len(history) >= 2:
                last_action = history[-1]
                second_last_action = history[-2] if len(history) >= 2 else None

                # If we just did the same action twice, try something different
                if last_action == second_last_action:
                    available_actions = [action for action in pattern if action != last_action]
                    if available_actions:
                        action = random.choice(available_actions)
                        self.logger.info(f"Mock AI: Avoiding repeat of {last_action}, chose {action}")
                        return action

            # Intelligently choose next action based on history
            if history:
                last_action = history[-1]

                # Simple state machine-like behavior
                if last_action == "UP":
                    # After going up, try right or interact
                    action = random.choice(["RIGHT", "A", "B"])
                elif last_action == "RIGHT":
                    # After going right, try down or interact
                    action = random.choice(["DOWN", "A", "B"])
                elif last_action == "DOWN":
                    # After going down, try left or interact
                    action = random.choice(["LEFT", "A", "B"])
                elif last_action == "LEFT":
                    # After going left, try up or interact
                    action = random.choice(["UP", "A", "B"])
                elif last_action in ["A", "B"]:
                    # After interacting, try moving
                    action = random.choice(["UP", "RIGHT", "DOWN", "LEFT"])
                else:
                    # Default: random from pattern
                    action = random.choice(pattern)
            else:
                # No history, start with UP
                action = "UP"

            # Add some randomness to make it feel more natural
            if random.random() < 0.2:  # 20% chance of random action
                action = random.choice(list(self.valid_actions))
                self.logger.info(f"Mock AI: Adding randomness, chose {action}")

            self.logger.info(f"Mock AI action: {action} (goal: '{goal[:30]}...' if len(goal) > 30 else goal)")
            return action

        except Exception as e:
            self.logger.error(f"Error in mock AI action generation: {e}")
            return "UP"  # Safe fallback

    def chat_with_ai(self, message: str, image_data: bytes, context: dict) -> str:
        """Generate a simple chat response"""
        try:
            message_lower = message.lower()

            # Simple response patterns based on keywords
            if any(word in message_lower for word in ['hello', 'hi', 'hey']):
                responses = [
                    "Hello! I'm helping you play this game. What would you like to do?",
                    "Hi there! I can see the game screen and help you make decisions.",
                    "Hey! Ready to play? What's your goal?"
                ]
            elif any(word in message_lower for word in ['help', 'how', 'what']):
                responses = [
                    "I can help you play this game! Tell me what you want to achieve and I'll suggest actions.",
                    "I'm your AI game assistant. I can see the screen and suggest button presses.",
                    "I analyze the game screen and suggest actions like UP, DOWN, LEFT, RIGHT, A, B, START, SELECT."
                ]
            elif any(word in message_lower for word in ['goal', 'objective', 'do']):
                responses = [
                    f"Your current goal is: {context.get('current_goal', 'Explore the game')}. I'll help you achieve it!",
                    "Let's work towards your goal. I'll analyze the screen and suggest the best actions.",
                    "I'll help you reach your objective by suggesting the right button presses."
                ]
            elif any(word in message_lower for word in ['action', 'move', 'press']):
                responses = [
                    "I'll suggest actions based on what I see on screen. Common actions are UP, DOWN, LEFT, RIGHT, A, B, START, SELECT.",
                    "I analyze the game state and recommend the best button press for your current goal."
                ]
            else:
                responses = [
                    "I can see the game screen and help you play. What would you like me to help you with?",
                    "I'm your AI game assistant. I can see what's happening and suggest actions.",
                    "Let me analyze the current game state and help you decide what to do next."
                ]

            response = random.choice(responses)
            self.logger.info(f"Mock AI chat response: {response[:50]}...")
            return response

        except Exception as e:
            self.logger.error(f"Error in mock AI chat: {e}")
            return "I'm here to help you play the game! What would you like to do?"