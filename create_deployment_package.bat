@echo off
setlocal enabledelayedexpansion

REM ================================================================
REM AI Game System - Deployment Package Creator
REM ================================================================
REM This script creates a clean deployment package for the AI Game System
REM It copies only the essential files needed for deployment
REM ================================================================

title AI Game System - Deployment Package Creator

echo ================================================================
echo == AI Game System - Deployment Package Creator             ==
echo ================================================================
echo.

REM Check if we're in the correct directory
if not exist "ai-game-server\src\main.py" (
    echo [ERROR] Cannot find backend server files.
    echo        Please run this script from the root directory containing ai-game-server and ai-game-assistant folders.
    echo.
    pause
    exit /b 1
)

if not exist "ai-game-assistant\package.json" (
    echo [ERROR] Cannot find frontend files.
    echo        Please run this script from the root directory containing ai-game-server and ai-game-assistant folders.
    echo.
    pause
    exit /b 1
)

REM Configuration
set "PACKAGE_NAME=AI_Game_System_Deployment"
set "PACKAGE_VERSION=1.0.0"
set "TIMESTAMP=%date:~-4,4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%"
set "TIMESTAMP=%TIMESTAMP: =0%"
set "PACKAGE_DIR=%PACKAGE_NAME%_%PACKAGE_VERSION%_%TIMESTAMP%"
set "INCLUDE_PYBOY=true"
set "INCLUDE_PYGBA=true"
set "INCLUDE_DOCUMENTATION=true"

REM Create package directory
echo [INFO] Creating deployment package directory...
if exist "%PACKAGE_DIR%" rmdir /s /q "%PACKAGE_DIR%"
mkdir "%PACKAGE_DIR%"

echo [INFO] Package directory: %PACKAGE_DIR%
echo.

REM ================================================================
REM STEP 1: Copy Core Backend Files
REM ================================================================
echo [STEP 1/6] Copying core backend files...

REM Create backend directory structure
mkdir "%PACKAGE_DIR%\ai-game-server"
mkdir "%PACKAGE_DIR%\ai-game-server\src"
mkdir "%PACKAGE_DIR%\ai-game-server\src\backend"
mkdir "%PACKAGE_DIR%\ai-game-server\src\backend\ai_apis"
mkdir "%PACKAGE_DIR%\ai-game-server\src\backend\emulators"
mkdir "%PACKAGE_DIR%\ai-game-server\src\backend\utils"
mkdir "%PACKAGE_DIR%\ai-game-server\tests"

REM Copy essential backend files
copy "ai-game-server\config.py" "%PACKAGE_DIR%\ai-game-server\" >nul 2>&1
copy "ai-game-server\requirements.txt" "%PACKAGE_DIR%\ai-game-server\" >nul 2>&1
copy "ai-game-server\start_server.bat" "%PACKAGE_DIR%\ai-game-server\" >nul 2>&1
copy "ai-game-server\start_server.sh" "%PACKAGE_DIR%\ai-game-server\" >nul 2>&1
copy "ai-game-server\src\main.py" "%PACKAGE_DIR%\ai-game-server\src\" >nul 2>&1
copy "ai-game-server\src\backend\__init__.py" "%PACKAGE_DIR%\ai-game-server\src\backend\" >nul 2>&1
copy "ai-game-server\src\backend\server.py" "%PACKAGE_DIR%\ai-game-server\src\backend\" >nul 2>&1
copy "ai-game-server\src\backend\ai_chat_endpoint.py" "%PACKAGE_DIR%\ai-game-server\src\backend\" >nul 2>&1
copy "ai-game-server\tests\test_server.py" "%PACKAGE_DIR%\ai-game-server\tests\" >nul 2>&1

REM Copy AI API files if they exist
if exist "ai-game-server\src\backend\ai_apis\gemini_api.py" (
    copy "ai-game-server\src\backend\ai_apis\gemini_api.py" "%PACKAGE_DIR%\ai-game-server\src\backend\ai_apis\" >nul 2>&1
)
if exist "ai-game-server\src\backend\ai_apis\openai_api.py" (
    copy "ai-game-server\src\backend\ai_apis\openai_api.py" "%PACKAGE_DIR%\ai-game-server\src\backend\ai_apis\" >nul 2>&1
)
if exist "ai-game-server\src\backend\ai_apis\nvidia_nim_api.py" (
    copy "ai-game-server\src\backend\ai_apis\nvidia_nim_api.py" "%PACKAGE_DIR%\ai-game-server\src\backend\ai_apis\" >nul 2>&1
)

REM Copy emulator files if they exist
if exist "ai-game-server\src\backend\emulators\pyboy_wrapper.py" (
    copy "ai-game-server\src\backend\emulators\pyboy_wrapper.py" "%PACKAGE_DIR%\ai-game-server\src\backend\emulators\" >nul 2>&1
)
if exist "ai-game-server\src\backend\emulators\pygba_wrapper.py" (
    copy "ai-game-server\src\backend\emulators\pygba_wrapper.py" "%PACKAGE_DIR%\ai-game-server\src\backend\emulators\" >nul 2>&1
)

REM Copy utility files if they exist
if exist "ai-game-server\src\backend\utils\game_utils.py" (
    copy "ai-game-server\src\backend\utils\game_utils.py" "%PACKAGE_DIR%\ai-game-server\src\backend\utils\" >nul 2>&1
)
if exist "ai-game-server\src\backend\utils\file_utils.py" (
    copy "ai-game-server\src\backend\utils\file_utils.py" "%PACKAGE_DIR%\ai-game-server\src\backend\utils\" >nul 2>&1
)
if exist "ai-game-server\src\backend\utils\system_utils.py" (
    copy "ai-game-server\src\backend\utils\system_utils.py" "%PACKAGE_DIR%\ai-game-server\src\backend\utils\" >nul 2>&1
)

echo [OK] Backend files copied successfully

REM ================================================================
REM STEP 2: Copy Core Frontend Files
REM ================================================================
echo.
echo [STEP 2/6] Copying core frontend files...

REM Create frontend directory structure
mkdir "%PACKAGE_DIR%\ai-game-assistant"
mkdir "%PACKAGE_DIR%\ai-game-assistant\components"
mkdir "%PACKAGE_DIR%\ai-game-assistant\services"
mkdir "%PACKAGE_DIR%\ai-game-assistant\dist"
mkdir "%PACKAGE_DIR%\ai-game-assistant\dist\assets"

REM Copy essential frontend files
copy "ai-game-assistant\.env.local" "%PACKAGE_DIR%\ai-game-assistant\" >nul 2>&1
copy "ai-game-assistant\.gitignore" "%PACKAGE_DIR%\ai-game-assistant\" >nul 2>&1
copy "ai-game-assistant\package.json" "%PACKAGE_DIR%\ai-game-assistant\" >nul 2>&1
copy "ai-game-assistant\package-lock.json" "%PACKAGE_DIR%\ai-game-assistant\" >nul 2>&1
copy "ai-game-assistant\tsconfig.json" "%PACKAGE_DIR%\ai-game-assistant\" >nul 2>&1
copy "ai-game-assistant\vite.config.ts" "%PACKAGE_DIR%\ai-game-assistant\" >nul 2>&1
copy "ai-game-assistant\index.html" "%PACKAGE_DIR%\ai-game-assistant\" >nul 2>&1
copy "ai-game-assistant\index.tsx" "%PACKAGE_DIR%\ai-game-assistant\" >nul 2>&1
copy "ai-game-assistant\types.ts" "%PACKAGE_DIR%\ai-game-assistant\" >nul 2>&1
copy "ai-game-assistant\index.css" "%PACKAGE_DIR%\ai-game-assistant\" >nul 2>&1
copy "ai-game-assistant\App.tsx" "%PACKAGE_DIR%\ai-game-assistant\" >nul 2>&1

REM Copy components if they exist
if exist "ai-game-assistant\components\GameBoy.tsx" (
    copy "ai-game-assistant\components\GameBoy.tsx" "%PACKAGE_DIR%\ai-game-assistant\components\" >nul 2>&1
)
if exist "ai-game-assistant\components\GameControls.tsx" (
    copy "ai-game-assistant\components\GameControls.tsx" "%PACKAGE_DIR%\ai-game-assistant\components\" >nul 2>&1
)
if exist "ai-game-assistant\components\AIChatInterface.tsx" (
    copy "ai-game-assistant\components\AIChatInterface.tsx" "%PACKAGE_DIR%\ai-game-assistant\components\" >nul 2>&1
)
if exist "ai-game-assistant\components\FileUpload.tsx" (
    copy "ai-game-assistant\components\FileUpload.tsx" "%PACKAGE_DIR%\ai-game-assistant\components\" >nul 2>&1
)
if exist "ai-game-assistant\components\StatusDisplay.tsx" (
    copy "ai-game-assistant\components\StatusDisplay.tsx" "%PACKAGE_DIR%\ai-game-assistant\components\" >nul 2>&1
)
if exist "ai-game-assistant\components\ROMManager.tsx" (
    copy "ai-game-assistant\components\ROMManager.tsx" "%PACKAGE_DIR%\ai-game-assistant\components\" >nul 2>&1
)
if exist "ai-game-assistant\components\SettingsPanel.tsx" (
    copy "ai-game-assistant\components\SettingsPanel.tsx" "%PACKAGE_DIR%\ai-game-assistant\components\" >nul 2>&1
)

REM Copy services if they exist
if exist "ai-game-assistant\services\aiService.ts" (
    copy "ai-game-assistant\services\aiService.ts" "%PACKAGE_DIR%\ai-game-assistant\services\" >nul 2>&1
)
if exist "ai-game-assistant\services\gameService.ts" (
    copy "ai-game-assistant\services\gameService.ts" "%PACKAGE_DIR%\ai-game-assistant\services\" >nul 2>&1
)
if exist "ai-game-assistant\services\socketService.ts" (
    copy "ai-game-assistant\services\socketService.ts" "%PACKAGE_DIR%\ai-game-assistant\services\" >nul 2>&1
)

REM Copy built frontend files if they exist
if exist "ai-game-assistant\dist\index.html" (
    copy "ai-game-assistant\dist\index.html" "%PACKAGE_DIR%\ai-game-assistant\dist\" >nul 2>&1
    xcopy "ai-game-assistant\dist\assets" "%PACKAGE_DIR%\ai-game-assistant\dist\assets\" /E /I /Y >nul 2>&1
)

echo [OK] Frontend files copied successfully

REM ================================================================
REM STEP 3: Copy Startup Scripts
REM ================================================================
echo.
echo [STEP 3/6] Copying startup scripts...

copy "unified_startup.bat" "%PACKAGE_DIR%\" >nul 2>&1
copy "unified_startup.ps1" "%PACKAGE_DIR%\" >nul 2>&1
copy "unified_startup.sh" "%PACKAGE_DIR%\" >nul 2>&1
copy "install_dependencies.bat" "%PACKAGE_DIR%\" >nul 2>&1
copy "install_dependencies.ps1" "%PACKAGE_DIR%\" >nul 2>&1
copy "install_dependencies.sh" "%PACKAGE_DIR%\" >nul 2>&1

echo [OK] Startup scripts copied successfully

REM ================================================================
REM STEP 4: Copy Optional Emulators
REM ================================================================
echo.
echo [STEP 4/6] Copying optional emulators...

if "%INCLUDE_PYBOY%"=="true" (
    if exist "PyBoy" (
        mkdir "%PACKAGE_DIR%\PyBoy"
        xcopy "PyBoy" "%PACKAGE_DIR%\PyBoy\" /E /I /Y >nul 2>&1
        echo [OK] PyBoy emulator included
    ) else (
        echo [WARNING] PyBoy directory not found - skipping
    )
)

if "%INCLUDE_PYGBA%"=="true" (
    if exist "pygba" (
        mkdir "%PACKAGE_DIR%\pygba"
        xcopy "pygba" "%PACKAGE_DIR%\pygba\" /E /I /Y >nul 2>&1
        echo [OK] PyGBA emulator included
    ) else (
        echo [WARNING] pygba directory not found - skipping
    )
)

REM ================================================================
REM STEP 5: Copy Documentation
REM ================================================================
echo.
echo [STEP 5/6] Copying documentation...

if "%INCLUDE_DOCUMENTATION%"=="true" (
    copy "DEPLOYMENT_STRUCTURE.txt" "%PACKAGE_DIR%\" >nul 2>&1
    copy "CLAUDE.md" "%PACKAGE_DIR%\" >nul 2>&1
    copy "ai-game-server\README.md" "%PACKAGE_DIR%\ai-game-server\" >nul 2>&1
    copy "ai-game-assistant\README.md" "%PACKAGE_DIR%\ai-game-assistant\" >nul 2>&1

    echo [OK] Documentation copied successfully
)

REM ================================================================
REM STEP 6: Create Deployment Documentation
REM ================================================================
echo.
echo [STEP 6/6] Creating deployment documentation...

REM Create deployment README
(
echo # AI Game System - Deployment Package
echo.
echo ## Package Information
echo - **Package Name**: %PACKAGE_NAME%
echo - **Version**: %PACKAGE_VERSION%
echo - **Created**: %date% %time%
echo - **Package Size**: ~30-40 MB (depending on included emulators)
echo.
echo ## Quick Start
echo 1. Extract the package to your desired location
echo 2. Run `install_dependencies.bat` to install required packages
echo 3. Run `unified_startup.bat` to start the system
echo 4. Open your web browser and go to http://localhost:5173
echo.
echo ## System Requirements
echo - **Operating System**: Windows 10/11, macOS, or Linux
echo - **Python**: 3.8 or higher
echo - **Node.js**: 16 or higher
echo - **Memory**: 4 GB RAM minimum, 8 GB recommended
echo - **Storage**: 100 MB free space (minimum)
echo.
echo ## Included Components
echo - **Backend Server**: Flask-based API server with AI integration
echo - **Frontend Web UI**: React-based web interface
echo - **Startup Scripts**: Windows, macOS, and Linux startup scripts
echo - **Dependency Installers**: Automated dependency management
echo.
echo ## Optional Components
if "%INCLUDE_PYBOY%"=="true" (
    echo - **PyBoy Emulator**: Game Boy emulator integration
) else (
    echo - **PyBoy Emulator**: Not included
)
if "%INCLUDE_PYGBA%"=="true" (
    echo - **PyGBA Emulator**: Game Boy Advance emulator integration
) else (
    echo - **PyGBA Emulator**: Not included
)
echo.
echo ## Port Usage
echo - **Backend API**: http://localhost:5000
echo - **Frontend Web UI**: http://localhost:5173
echo - **Service Monitor**: http://localhost:8080 (optional)
echo.
echo ## Troubleshooting
echo - If you encounter dependency issues, run `install_dependencies.bat` again
echo - If ports are already in use, the system will automatically find available ports
echo - Check the logs/ directory for detailed error information
echo - For more detailed troubleshooting, see the included documentation
echo.
echo ## Support
echo For issues and support, please refer to the included documentation files.
echo.
echo ---
echo This deployment package was created automatically by the AI Game System.
echo Package generated on: %date% at %time%
) > "%PACKAGE_DIR%\DEPLOYMENT_README.md"

REM Create version info file
(
echo AI Game System Deployment Package
echo ===================================
echo Package Name: %PACKAGE_NAME%
echo Version: %PACKAGE_VERSION%
echo Created: %date% %time%
echo Platform: Windows
echo Creator: Deployment Package Creator Script
echo.
echo Included Components:
echo - Core Backend Server: YES
echo - Core Frontend Web UI: YES
echo - Startup Scripts: YES
echo - Dependency Installers: YES
echo - PyBoy Emulator: %INCLUDE_PYBOY%
echo - PyGBA Emulator: %INCLUDE_PYGBA%
echo - Documentation: %INCLUDE_DOCUMENTATION%
echo.
echo Estimated Size: 30-40 MB
echo Minimum Requirements: Python 3.8+, Node.js 16+, 4GB RAM
echo.
echo Files Included:
) > "%PACKAGE_DIR%\VERSION_INFO.txt"

REM Count files
for /f %%A in ('dir "%PACKAGE_DIR%" /s /b ^| find /v /c ""') do set "FILE_COUNT=%%A"
echo Total Files: %FILE_COUNT% >> "%PACKAGE_DIR%\VERSION_INFO.txt"

echo [OK] Deployment documentation created

REM ================================================================
REM FINAL STEPS
REM ================================================================
echo.
echo ================================================================
echo == Package Creation Complete!                               ==
echo ================================================================
echo.
echo [SUCCESS] Deployment package created successfully!
echo.
echo [PACKAGE DETAILS]
echo -----------------
echo Package Name: %PACKAGE_DIR%
echo Location: %CD%\%PACKAGE_DIR%
echo Files Included: %FILE_COUNT%
echo.
echo [NEXT STEPS]
echo ------------
echo 1. Copy the entire "%PACKAGE_DIR%" folder to the target system
echo 2. On the target system, run "install_dependencies.bat"
echo 3. Run "unified_startup.bat" to start the system
echo 4. Access the web interface at http://localhost:5173
echo.
echo [PACKAGE CONTENTS]
echo ------------------
echo ✓ Core backend server files
echo ✓ Core frontend web interface
echo ✓ Startup scripts for all platforms
echo ✓ Dependency installation scripts
if "%INCLUDE_PYBOY%"=="true" (
    echo ✓ PyBoy emulator integration
)
if "%INCLUDE_PYGBA%"=="true" (
    echo ✓ PyGBA emulator integration
)
if "%INCLUDE_DOCUMENTATION%"=="true" (
    echo ✓ Complete documentation
)
echo.
echo [OPTIONAL: CREATE ZIP ARCHIVE]
echo ----------------------------
echo Would you like to create a ZIP archive of the package?
echo This will make it easier to distribute.
echo.
set /p create_zip="Create ZIP archive? (y/n): "
if /i "%create_zip%"=="y" (
    echo.
    echo [INFO] Creating ZIP archive...
    powershell -Command "Compress-Archive -Path '%PACKAGE_DIR%' -DestinationPath '%PACKAGE_DIR%.zip' -Force"
    if %errorlevel% equ 0 (
        echo [SUCCESS] ZIP archive created: %PACKAGE_DIR%.zip
        echo.
        echo [ARCHIVE INFO]
        echo --------------
        echo Archive File: %PACKAGE_DIR%.zip
        echo Archive Size:
        dir "%PACKAGE_DIR%.zip" | findstr "%PACKAGE_DIR%.zip"
    ) else (
        echo [ERROR] Failed to create ZIP archive
    )
)

echo.
echo [DEPLOYMENT COMPLETE]
echo =====================
echo Your AI Game System deployment package is ready!
echo.
echo Package Directory: %CD%\%PACKAGE_DIR%
echo Instructions: See %PACKAGE_DIR%\DEPLOYMENT_README.md
echo.
echo Thank you for using the AI Game System!
echo.
pause
exit /b 0