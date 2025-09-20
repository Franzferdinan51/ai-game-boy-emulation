@echo off
echo Setting up AI Game Server environment...

:: Set default environment variables for development
:: Replace these with your actual API keys

:: Gemini API Key (Get from https://makersuite.google.com/app/apikey)
set GEMINI_API_KEY=your_gemini_api_key_here

:: OpenRouter API Key (Get from https://openrouter.ai/keys)
set OPENROUTER_API_KEY=your_openrouter_api_key_here

:: NVIDIA API Key (Get from https://build.nvidia.com/)
set NVIDIA_API_KEY=your_nvidia_api_key_here
set NVIDIA_MODEL=nvidia/llama3-llm-70b

:: OpenAI API Key (Get from https://platform.openai.com/api-keys)
set OPENAI_API_KEY=your_openai_api_key_here
set OPENAI_ENDPOINT=https://api.openai.com/v1

:: Server configuration
set HOST=0.0.0.0
set PORT=5000
set DEBUG=True

echo Environment variables set successfully!
echo.
echo IMPORTANT: Replace the placeholder API keys with your actual keys in this script.
echo.
echo To get API keys:
echo   - Gemini: https://makersuite.google.com/app/apikey
echo   - OpenRouter: https://openrouter.ai/keys
echo   - NVIDIA: https://build.nvidia.com/
echo   - OpenAI: https://platform.openai.com/api-keys
echo.
echo Starting server...
cd /d "%~dp0"
python src/backend/server.py

pause