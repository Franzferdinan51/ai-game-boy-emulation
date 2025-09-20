# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive AI-powered Game Boy emulation system featuring real-time game streaming, multiple AI provider integration, and a modern React web interface. The system combines retro gaming with cutting-edge AI assistance for game strategy, automation, and analysis.

## Essential Commands

### Quick Start (Recommended)
```bash
# Automated startup with all features
unified_startup.bat
```

### Manual Development Setup
```bash
# Backend (Python/Flask)
cd ai-game-server
pip install -r requirements.txt
python src/main.py

# Frontend (React/TypeScript)
cd ai-game-assistant
npm install
npm run dev
```

### PyBoy Emulator (Optional - for advanced use)
```bash
cd PyBoy
make build_pyboy             # Build Cython extensions
make test_cython            # Run tests
pip install -e .             # Install in development mode
```

## Architecture Overview

### Backend Server (`ai-game-server/`)
- **Flask REST API**: Main server with SSE streaming for real-time game screens
- **Multi-AI Integration**: Gemini, OpenRouter, NVIDIA NIM, OpenAI-compatible APIs
- **Emulator Wrappers**: PyBoy integration with secure subprocess handling
- **Performance Optimized**: OpenCV/PyTorch acceleration when available
- **Thread-Safe State**: Global state management with proper locking

### Frontend Application (`ai-game-assistant/`)
- **React + TypeScript**: Modern component-based architecture
- **Real-time Streaming**: SSE for game screen updates
- **Mobile Responsive**: Touch controls and adaptive layouts
- **Multiple Panels**: Game view, AI chat, settings, ROM management
- **Enhanced UI**: Framer Motion animations, accessibility features

### Key Integration Points
- **Screen Streaming**: Real-time game screen capture via SSE endpoints
- **AI Chat Interface**: Vision-capable AI analysis of game screenshots
- **ROM Management**: Upload and manage Game Boy ROM files
- **Service Monitoring**: Health checks and auto-restart capabilities

## Development Commands

### Backend Development
```bash
# Start server with development settings
cd ai-game-server
set FLASK_ENV=development
python src/main.py

# Test individual components
python demo_ai_features.py        # Test AI integration
python demo_ui_integration.py     # Test emulator controls
```

### Frontend Development
```bash
# Development server with hot reload
cd ai-game-assistant
npm run dev                       # Starts on localhost:5173

# Build for production
npm run build

# Type checking
npx tsc --noEmit
```

### Testing and Validation
```bash
# System health check
python verify_system.py          # Check all dependencies
python verify_backend.py         # Test backend components
python verify_frontend.py         # Test frontend components

# Service monitoring
python service_monitor.py        # Start monitoring dashboard
python unified_logger.py         # Enhanced logging system
```

## Security Considerations

### Critical Security Features
- **Input Validation**: Comprehensive validation for all user inputs
- **Subprocess Security**: Secure template-based emulator launching
- **Rate Limiting**: API endpoint protection
- **File Upload Security**: ROM file validation and sanitization

### When Working with Security
- Always validate string inputs with `validate_string_input()`
- Use secure subprocess templates for any emulator operations
- Implement proper error handling without exposing sensitive information
- Follow the principle of least privilege for file operations

## Performance Optimization

### Backend Optimizations
- **Screen Capture**: Caching and OpenCV acceleration when available
- **GPU Processing**: PyTorch integration for image processing
- **Memory Management**: Proper cleanup of ThreadPoolExecutor
- **Adaptive FPS**: Dynamic frame rate adjustment based on system load

### Frontend Optimizations
- **Virtual Scrolling**: For large message histories
- **Image Optimization**: Efficient screen capture handling
- **Animation Performance**: Framer Motion with hardware acceleration
- **Mobile Performance**: Touch-optimized controls and responsive design

## Multi-Platform Support

### Windows Development (Primary)
- Use `unified_startup.bat` for comprehensive startup options
- Multiple startup modes: Basic, Enhanced Logging, Development, Ultimate
- Service monitoring with web dashboard at http://localhost:8080

### Cross-Platform Considerations
- Shell scripts available for Linux/Mac (`unified_startup.sh`)
- PowerShell alternative (`unified_startup.ps1`)
- Path handling works across platforms with proper escaping

## Configuration

### Environment Variables
```env
# AI API Keys (required for AI features)
GEMINI_API_KEY=your_gemini_key
OPENROUTER_API_KEY=your_openrouter_key
NVIDIA_API_KEY=your_nvidia_key

# Server Configuration
FLASK_ENV=development
BACKEND_PORT=5000
FRONTEND_PORT=5173
MONITOR_PORT=8080
```

### Port Configuration
- **Backend Server**: localhost:5000 (Flask API)
- **Frontend Dev**: localhost:5173 (Vite dev server)
- **Monitor Dashboard**: localhost:8080 (Service monitoring)

## Common Development Tasks

### Adding New AI Providers
1. Create new API module in `ai-game-server/src/backend/ai_apis/`
2. Implement the standard AI interface with vision capabilities
3. Add provider configuration to the AI router system
4. Test with the demo AI features script

### Extending Emulator Support
1. Create new emulator wrapper in `ai-game-server/src/backend/emulators/`
2. Implement secure subprocess handling for the emulator
3. Add screen capture and control interfaces
4. Update the UI configuration for new emulator options

### Frontend Component Development
1. Use TypeScript with proper type definitions
2. Follow the established component patterns in `/components/`
3. Implement proper accessibility features
4. Test with both desktop and mobile layouts

## Troubleshooting

### Common Issues
- **Port Conflicts**: Use different ports in environment variables
- **AI API Issues**: Verify API keys and network connectivity
- **Emulator Problems**: Check ROM file integrity and emulator installation
- **Build Failures**: Clear cache and reinstall dependencies

### Debug Tools
- **Service Monitor**: Real-time health monitoring and auto-restart
- **Unified Logger**: Comprehensive logging with color coding
- **System Verification**: Automated dependency and configuration checking
- **Development Mode**: Verbose logging for troubleshooting

## File Structure Highlights

```
├── ai-game-server/              # Flask backend
│   ├── src/backend/
│   │   ├── server.py            # Main server with SSE streaming
│   │   ├── ai_apis/            # AI provider implementations
│   │   ├── emulators/          # Emulator wrappers
│   │   └── utils/              # Utility functions
│   └── requirements.txt         # Python dependencies
├── ai-game-assistant/           # React frontend
│   ├── components/             # TypeScript React components
│   ├── services/               # API services
│   └── package.json            # Node.js dependencies
├── PyBoy/                      # Game Boy emulator (optional)
├── unified_startup.bat          # Main startup script
└── service_monitor.py          # System monitoring
```

## Dependencies

### Backend Dependencies
- **Flask**: Web framework with CORS support
- **NumPy/Pillow**: Array operations and image processing
- **OpenCV/PyTorch**: Performance optimizations (optional)
- **OpenAI**: Multi-AI provider compatibility

### Frontend Dependencies
- **React 19**: Modern React with concurrent features
- **TypeScript**: Type-safe development
- **Vite**: Fast build tool and dev server
- **Framer Motion**: Animation library
- **Lucide React**: Icon library