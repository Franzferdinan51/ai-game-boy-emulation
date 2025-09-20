#!/bin/bash
# Startup script for AI Game Server

# Set environment variables (modify as needed)
export GEMINI_API_KEY="your-gemini-api-key-here"
export OPENROUTER_API_KEY="your-openrouter-api-key-here"
export NVIDIA_API_KEY="your-nvidia-api-key-here"

# Install dependencies if not already installed
echo "Installing dependencies..."
pip install -r requirements.txt

# Install emulator packages
echo "Installing emulator packages..."
pip install pyboy
# pip install pygba  # Uncomment if you need GBA support

# Start the server
echo "Starting AI Game Server..."
python src/main.py