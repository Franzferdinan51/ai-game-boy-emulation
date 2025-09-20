@echo off
echo Starting AI Game Server with LM Studio support...

REM Set environment variables for LM Studio
set LM_STUDIO_URL=http://localhost:1234/v1
set OPENAI_ENDPOINT=http://localhost:1234/v1
set AI_ENDPOINT=http://localhost:1234/v1
set OPENAI_API_KEY=not-needed

echo Environment variables set:
echo LM_STUDIO_URL=%LM_STUDIO_URL%
echo OPENAI_API_KEY=%OPENAI_API_KEY%

REM Start the server
echo Starting server...
python src/backend/server.py

pause