@echo off
REM Startup script for AI Game Server (Windows)

REM Set environment variables (modify as needed)
set GEMINI_API_KEY=your-gemini-api-key-here
set OPENROUTER_API_KEY=your-openrouter-api-key-here
set NVIDIA_API_KEY=your-nvidia-api-key-here

REM Install dependencies if not already installed
echo Installing dependencies...
pip install -r requirements.txt

REM Install emulator packages
echo Installing emulator packages...
pip install pyboy
REM pip install pygba  # Uncomment if you need GBA support

REM Start the server
echo Starting AI Game Server...
python src/main.py