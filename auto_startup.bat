@echo off
title AI Game System - Auto Startup (Google AI Studio UI)
echo [STARTUP] Launching AI Game System with Google AI Studio UI...

REM Basic checks
if not exist "ai-game-server\src\main.py" (
    echo [ERROR] ai-game-server\src\main.py not found!
    pause
    exit /b 1
)
if not exist "google-ai-studio-ui\package.json" (
    echo [ERROR] google-ai-studio-ui\package.json not found!
    pause
    exit /b 1
)
echo [OK] Directories OK

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install from python.org
    pause
    exit /b 1
)
echo [OK] Python OK

REM Check Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js not found. Install from nodejs.org
    pause
    exit /b 1
)
echo [OK] Node.js OK

REM Install Python dependencies only if not present
echo [BACKEND DEP] Checking Python dependencies...
pushd "%~dp0ai-game-server"
pip show flask >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Flask and dependencies...
    pip install flask flask-cors pillow numpy psutil openai
) else (
    echo Flask already installed
)
pip show pyboy >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing PyBoy...
    pip install pyboy
) else (
    echo PyBoy already installed
)

echo [INFO] Checking for PyGBA...
pip show pygba
if %errorlevel% neq 0 (
    echo [INFO] PyGBA not found. Attempting installation...
    pip install pygba
    if %errorlevel% neq 0 (
        echo [WARNING] PyGBA installation failed. GBA support will be disabled.
    ) else (
        echo [SUCCESS] PyGBA installed successfully.
    )
) else (
    echo [INFO] PyGBA is already installed.
)
popd

REM Install Node dependencies only if node_modules doesn't exist
echo [FRONTEND DEP] Checking Node dependencies for Google AI Studio UI...
pushd "%~dp0google-ai-studio-ui"
if not exist "node_modules" (
    echo [INFO] 'node_modules' not found. Running 'npm install'...
    npm install
) else (
    echo [INFO] 'node_modules' already exists. Skipping 'npm install'.
)
popd

REM Start backend with LM Studio support
echo [BACKEND] Starting server on http://localhost:5000...
echo [BACKEND] Configuring LM Studio environment variables...
start "AI Game Server" cmd /k "cd /d %~dp0ai-game-server\src && set LM_STUDIO_URL=http://localhost:1234/v1 && set OPENAI_ENDPOINT=http://localhost:1234/v1 && set AI_ENDPOINT=http://localhost:1234/v1 && set OPENAI_API_KEY=not-needed && python main.py"

REM Wait 5 seconds for backend to start
timeout /t 5 >nul

REM Start Google AI Studio UI
echo [FRONTEND] Starting Google AI Studio UI...
echo [FRONTEND] UI will be available on http://localhost:3000 (or next available port)
start "Google AI Studio UI" cmd /k "cd /d %~dp0google-ai-studio-ui && npm run dev"

echo [SUCCESS] System started!
echo ==========================================
echo  🎮 AI GAME SYSTEM WITH GOOGLE AI STUDIO UI
echo ==========================================
echo Backend Server: http://localhost:5000
echo Google AI Studio UI: http://localhost:3000 (check UI window for actual port)
echo ==========================================
echo 🎯 Features:
echo   - Load GB/GBC/GBA ROM files
echo   - Real-time game streaming
echo   - AI-powered game assistance
echo   - Modern chat interface
echo   - Session management
echo ==========================================
echo 💡 Tips:
echo   - Drag ROM files to the game screen
echo   - Chat with AI for game strategy
echo   - Enable AI auto-play mode
echo   - Use settings to customize experience
echo ==========================================
echo [INFO] LM Studio integration enabled at http://localhost:1234/v1
pause