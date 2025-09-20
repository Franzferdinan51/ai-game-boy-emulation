<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# AI Game Assistant

A React-based web application that provides an AI assistant for playing Game Boy and Game Boy Advance games on emulators.

## Features

- Web-based interface for game emulation
- AI assistant that can analyze game screens and suggest actions
- Support for both Game Boy and Game Boy Advance modes
- Action history tracking and logging
- Real-time AI decision visualization
- Dynamic model selection for all AI providers
- Chat interface for interacting with the AI

## Technologies Used

- React 19.x
- TypeScript
- Vite for build tooling
- Tailwind CSS for styling

## Run Locally (Simple Method)

**Prerequisites:** Python 3.8+, Node.js 14+

1. Run the unified startup script:
   - Double-click `unified_startup.bat` in the root directory
   - Or run from command prompt: `unified_startup.bat`

This will automatically install dependencies and start both the backend server and frontend application.

## Run Locally (Manual Method)

**Prerequisites:** Node.js

1. Install dependencies:
   `npm install`
2. Configure AI providers in the Settings panel:
   - Add your API keys for the providers you want to use
   - Select your preferred AI provider and model
3. Run the app:
   `npm run dev`

For the backend server, navigate to the `ai-game-server` directory and run:
`python src/main.py`

## AI Provider Configuration

The AI Game Assistant supports multiple AI providers with dynamic model selection:

### Google Gemini
- **Model Selection:** Choose from available Gemini models
- **Requirements:** API key from Google AI Studio
- **Best Models:** `gemini-1.5-pro`, `gemini-1.5-flash`

### OpenRouter
- **Model Selection:** Manual input in format `vendor/model-name:version`
- **Requirements:** API key from OpenRouter
- **Popular Models:** 
  - `openai/gpt-4-vision-preview`
  - `anthropic/claude-3-opus:beta`
  - `google/gemini-pro-vision`

### NVIDIA NIM
- **Model Selection:** Choose from available NVIDIA models
- **Requirements:** API key from NVIDIA
- **Popular Models:**
  - `meta/llama3-8b-instruct`
  - `meta/llama3-70b-instruct`

### OpenAI Compatible (LM Studio, Ollama, etc.)
- **Model Selection:** Auto-detection with "Detect" button or manual input
- **Requirements:** Local server running (LM Studio, Ollama, etc.)
- **Popular Models:**
  - `llava:13b` (vision capable)
  - `bakllava` (vision capable)

## Usage

1. Open your browser and navigate to: http://localhost:5173
2. Configure your AI providers in the Settings panel (gear icon)
3. Load a ROM file using the "Load ROM" button
4. Set your AI goal in the control panel
5. Start the AI with the "Start AI" button
6. Watch as the AI plays the game based on your goal

## Development

### Project Structure

```
ai-game-assistant/
├── components/            # React components
├── services/              # API services and utilities
├── App.tsx                # Main application component
├── index.tsx              # Application entry point
└── types.ts               # TypeScript type definitions
```

### Key Components

- `App.tsx`: Main application component managing state and logic
- `Header`: Navigation and mode selection
- `EmulatorScreen`: Game screen display and ROM loading
- `Controls`: Game control visualization
- `AIPanel`: AI control panel with goal setting and logs
- `SettingsModal`: Provider and model configuration

## Troubleshooting

### Model Selection Issues

1. **NVIDIA models not loading:** 
   - Ensure your NVIDIA API key is correct
   - Check your internet connection
   - Try refreshing the model list

2. **LM Studio models not detected:**
   - Ensure LM Studio is running with a model loaded
   - Check that the endpoint is set correctly (typically `http://localhost:1234/v1`)
   - Click the "Detect" button to refresh the model list

3. **OpenRouter model format errors:**
   - Ensure models are in the correct format: `vendor/model-name:version`
   - Example: `qwen/qwen3-coder:free`

### API Key Issues

1. **Invalid API key errors:**
   - Verify your API keys are correct
   - Check that you've set them in the Settings panel
   - For local providers (LM Studio), API key can often be left blank

2. **Rate limit errors:**
   - Try a different AI provider
   - Wait for rate limits to reset
   - Consider upgrading your API plan if available

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add some AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.