# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive AI-powered Game Boy emulation system featuring real-time game streaming, multiple AI provider integration, and modern React web interfaces. The system combines retro gaming with cutting-edge AI assistance for game strategy, automation, and analysis. The architecture includes a Flask backend with SSE streaming, multiple Next.js/React frontends, and optional PyBoy emulator integration.

## Essential Commands

### Primary Startup (Recommended)
```bash
# Unified startup script with 11 different modes and comprehensive features
unified_startup.bat
```

### Manual Development Setup
```bash
# Backend (Python/Flask)
cd ai-game-server
pip install -r requirements.txt
set FLASK_ENV=development
python src/main.py

# Main Frontend (Next.js 15 + React 19 + TypeScript)
cd ai-game-assistant
npm install
npm run dev                  # Starts on localhost:5173 with hot reload

# Alternative Frontend (GLM4.5-UI)
cd GLM4.5-UI
npm install
npm run dev                  # Starts on localhost:3000
```

### PyBoy Emulator (Optional - for advanced use)
```bash
cd PyBoy
pip install -r requirements.txt
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
- **Next.js 15 + React 19**: Modern framework with App Router
- **TypeScript**: Type-safe development with strict configuration
- **Modern UI Stack**: Tailwind CSS 4, shadcn/ui, Framer Motion
- **Real-time Streaming**: SSE for game screen updates
- **State Management**: Zustand for client state, TanStack Query for server state
- **Database**: Prisma ORM with TypeScript integration
- **Authentication**: NextAuth.js ready
- **Mobile Responsive**: Touch controls and adaptive layouts
- **Advanced Features**: Drag & drop, charts, forms with validation

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
# Development server with hot reload (includes nodemon + logging)
cd ai-game-assistant
npm run dev                       # Starts on localhost:5173

# Build for production
npm run build

# Start production server (includes logging)
npm start

# Type checking
npx tsc --noEmit

# Linting
npm run lint

# Database operations (Prisma)
npm run db:push                   # Push schema to database
npm run db:generate              # Generate Prisma client
npm run db:migrate               # Run migrations
npm run db:reset                 # Reset database

# GLM4.5-UI Frontend Development
cd GLM4.5-UI
npm run dev                       # Alternative UI on localhost:3000
```

### Testing and Validation
```bash
# System health checks
python verify_system.py          # Check all dependencies
python verify_backend.py         # Test backend components
python final_verification.py     # Final system validation

# Service monitoring and logging
python service_monitor.py        # Start monitoring dashboard (localhost:8080)
python unified_logger.py         # Enhanced logging system
python start_monitor.py          # Start monitoring service

# Specialized startup scripts
python start_with_monitor.bat    # Start with monitoring
python start_with_logging.bat    # Start with enhanced logging
python start_optimized_system.bat # Optimized system startup
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
- Use `unified_startup.bat` for comprehensive startup options (11 modes)
- All modes provide continuous monitoring and never exit automatically
- Service monitoring with web dashboard at http://localhost:8080
- Real-time log streaming and interactive controls in monitoring modes

### Cross-Platform Considerations
- Shell scripts available for Linux/Mac (`unified_startup.sh`, `start_system.sh`)
- PowerShell alternative (`unified_startup.ps1`)
- Path handling works across platforms with proper escaping
- All startup scripts provide similar functionality across platforms

## Configuration

### Environment Variables
```env
# AI API Keys (required for AI features)
GEMINI_API_KEY=your_gemini_key
OPENROUTER_API_KEY=your_openrouter_key
NVIDIA_API_KEY=your_nvidia_key
OPENAI_API_KEY=your_openai_key

# Server Configuration
FLASK_ENV=development
BACKEND_PORT=5000
FRONTEND_PORT=5173
GLM_UI_PORT=3000
MONITOR_PORT=8080

# Database (Prisma)
DATABASE_URL="file:./dev.db"
```

### Port Configuration
- **Backend Server**: localhost:5000 (Flask API)
- **Frontend Dev**: localhost:5173 (Next.js dev server)
- **GLM4.5 UI**: localhost:3000 (Alternative Next.js UI)
- **Monitor Dashboard**: localhost:8080 (Service monitoring)

### Startup Modes
The `unified_startup.bat` script provides 11 comprehensive startup options:

1. **Basic Startup** - Quick start with minimal logging
2. **Enhanced Logging** - Terminal logging + file logging with unified_logger.py
3. **Monitoring Console** - Real-time monitoring with interactive controls
4. **Service Monitor** - Auto-restart + web dashboard at localhost:8080
5. **Development Mode** - Verbose logging + debugging + hot reload
6. **Clean Start** - Kill existing processes + fresh start
7. **System Health Check** - Check dependencies and system status
8. **Recovery Mode** - Advanced troubleshooting and recovery
9. **GLM4.5-UI Mode** - Start Next.js UI with modern components
10. **Ultimate Mode** - Start EVERYTHING with all features (recommended)
11. **Complete System Shutdown** - Stops ALL services and processes

**Note**: The unified startup script never exits automatically - it always returns to the main menu for continuous operation.

## Development Workflows

### Daily Development Routine
1. **Startup**: Use `unified_startup.bat` → Choose Development Mode (5) for hot reload
2. **Backend Changes**: Edit in `ai-game-server/src/backend/` - auto-reloads
3. **Frontend Changes**: Edit in `ai-game-assistant/` - hot reloads via Vite
4. **Testing**: Use built-in monitoring controls (L for logs, R for restart)
5. **Database**: Use Prisma commands for schema changes
6. **Shutdown**: Choose option 0 for clean system shutdown

### Code Quality Workflow
```bash
# Backend
cd ai-game-server
python -m pytest tests/  # Run tests (if available)
python -m flake8 src/   # Linting (if configured)

# Frontend
cd ai-game-assistant
npm run lint            # ESLint checking
npx tsc --noEmit        # TypeScript type checking
npm run build           # Production build test
```

### Database Development
```bash
cd ai-game-assistant
npm run db:push         # Push schema changes
npm run db:generate     # Regenerate Prisma client
npm run db:migrate      # Run migrations
npm run db:studio       # Open Prisma Studio (if available)
```

### Production Deployment
1. Use Ultimate Mode (10) for full system testing
2. Run `npm run build` in both frontend directories
3. Test all AI providers and emulator functionality
4. Verify monitoring dashboard functionality
5. Use deployment package creation scripts if available

## Common Development Tasks

### Adding New AI Providers
1. Create new API module in `ai-game-server/src/backend/ai_apis/`
2. Implement the standard AI interface with vision capabilities
3. Add provider configuration to the AI router system
4. Update environment variables for API keys
5. Test with the demo AI features script

### Extending Emulator Support
1. Create new emulator wrapper in `ai-game-server/src/backend/emulators/`
2. Implement secure subprocess handling for the emulator
3. Add screen capture and control interfaces
4. Update the UI configuration for new emulator options
5. Test with different ROM files and game scenarios

### Frontend Component Development
1. Use TypeScript with proper type definitions
2. Follow the established component patterns in `/components/`
3. Implement proper accessibility features
4. Test with both desktop and mobile layouts
5. Use shadcn/ui components for consistency

### Monitoring and Debugging
1. Use Service Monitor mode (4) for auto-restart capabilities
2. Access web dashboard at localhost:8080 for real-time metrics
3. Use monitoring controls: L (logs), H (health), T (connectivity test)
4. Export logs for debugging with E key in monitoring modes
5. Use unified_logger.py for enhanced logging with color coding

## Troubleshooting

### Common Issues
- **Port Conflicts**: Use different ports in environment variables
- **AI API Issues**: Verify API keys and network connectivity
- **Emulator Problems**: Check ROM file integrity and emulator installation
- **Build Failures**: Clear cache and reinstall dependencies

### Debug Tools
- **Service Monitor**: Real-time health monitoring and auto-restart with web dashboard
- **Unified Logger**: Comprehensive logging with color coding and real-time streaming
- **System Verification**: Automated dependency and configuration checking
- **Development Mode**: Verbose logging with hot reload for debugging
- **Interactive Controls**: Built-in monitoring controls (L-logs, R-restart, H-health, T-test, E-export)
- **Real-time Log Streaming**: Live terminal output during monitoring sessions
- **Web Dashboard**: Service monitoring at localhost:8080 with metrics visualization
- **Log Export**: Export all logs for debugging and analysis

### System Recovery
- **Recovery Mode**: Advanced troubleshooting and automatic system recovery
- **Clean Start**: Kill existing processes and fresh start
- **Complete Shutdown**: Proper cleanup of all services and processes
- **Auto-restart**: Service monitor automatically restarts failed services

## File Structure Highlights

```
├── ai-game-server/              # Flask backend
│   ├── src/backend/
│   │   ├── server.py            # Main server with SSE streaming
│   │   ├── optimized_server.py  # Performance-optimized server
│   │   ├── ai_apis/            # AI provider implementations
│   │   │   ├── ai_provider_manager.py
│   │   │   ├── gemini_api.py
│   │   │   ├── nvidia_api.py
│   │   │   └── openrouter_api.py
│   │   ├── emulators/          # Emulator wrappers
│   │   │   ├── pyboy_emulator.py
│   │   │   ├── enhanced_pyboy_emulator.py
│   │   │   └── game_analyzer.py
│   │   └── utils/              # Utility functions
│   └── requirements.txt         # Python dependencies
├── ai-game-assistant/           # Main React frontend
│   ├── components/             # TypeScript React components
│   │   ├── AIPanel.tsx
│   │   ├── EmulatorScreen.tsx
│   │   ├── Controls.tsx
│   │   └── SettingsModal.tsx
│   ├── services/               # API services
│   │   ├── backendService.ts
│   │   └── geminiService.ts
│   ├── src/                    # Next.js App Router structure
│   ├── prisma/                 # Database schema
│   ├── server.ts               # Next.js server
│   └── package.json            # Node.js dependencies
├── GLM4.5-UI/                  # Alternative Next.js frontend
│   ├── src/                    # GLM4.5-specific components
│   └── package.json            # Node.js dependencies
├── PyBoy/                      # Game Boy emulator (optional)
├── unified_startup.bat          # Main startup script (11 modes)
├── unified_startup.sh/.ps1      # Cross-platform startup scripts
├── service_monitor.py          # System monitoring dashboard
├── unified_logger.py           # Enhanced logging system
├── verify_*.py                 # System validation scripts
└── deployment_package_*/       # Deployment archives
```

## Dependencies

### Backend Dependencies
- **Flask 2.3.2**: Web framework with CORS support
- **Flask-CORS 4.0.0**: Cross-origin resource sharing
- **NumPy 1.24.3**: Array operations and numerical computing
- **Pillow 9.5.0**: Image processing and manipulation
- **Requests 2.31.0**: HTTP library for API calls
- **OpenAI >=1.0.0**: Multi-AI provider compatibility
- **PyBoy (Optional)**: Game Boy emulator integration
- **PyGBA (Optional)**: Alternative Game Boy Advance emulator

### Frontend Dependencies
- **Next.js 15.3.5**: React framework with App Router
- **React 19.0.0**: Modern React with concurrent features
- **TypeScript 5**: Type-safe development
- **Tailwind CSS 4**: Utility-first CSS framework
- **shadcn/ui**: High-quality accessible components (Radix UI based)
- **Framer Motion 12.23.2**: Production-ready motion library
- **Zustand 5.0.6**: Simple state management
- **TanStack Query 5.82.0**: Server state management
- **Prisma 6.11.1**: Next-generation ORM
- **NextAuth.js 4.24.11**: Authentication solution
- **React Hook Form 7.60.0 + Zod 4.0.2**: Forms with validation
- **Lucide React 0.525.0**: Icon library
- **Socket.IO 4.8.1**: Real-time communication
- **Axios 1.10.0**: HTTP client
- **tsx 4.20.3**: TypeScript execution
- **nodemon 3.1.10**: Development auto-restart
- **@dnd-kit**: Drag and drop functionality
- **Recharts**: Data visualization
- **AI Function Helper 3.5.2** (GLM4.5-UI only): AI integration helpers

### Development Tools
- **ESLint 9**: Code linting and formatting
- **Prisma**: Database schema management
- **Sharp**: Image optimization
- **uuid**: Unique identifier generation

## Architecture Patterns

### Backend Patterns
- **SSE Streaming**: Real-time game screen updates using Server-Sent Events
- **Multi-AI Router**: Fallback system between different AI providers
- **Secure Subprocess**: Template-based emulator launching with security validation
- **Thread Pool Management**: Efficient handling of concurrent requests
- **State Management**: Thread-safe global state with proper locking

### Frontend Patterns
- **App Router**: Next.js 15 with modern routing structure
- **Component Architecture**: Reusable components with TypeScript types
- **State Management**: Zustand for client state, TanStack Query for API state
- **Real-time Updates**: SSE integration for live game streaming
- **Form Handling**: React Hook Form with Zod validation
- **Accessibility**: WCAG-compliant components with ARIA support

### Integration Patterns
- **API Layer**: Consistent RESTful API design with OpenAPI documentation
- **Event-Driven**: Event system for game state changes and AI responses
- **Error Handling**: Comprehensive error management with user-friendly messages
- **Security**: Input validation, rate limiting, and secure file handling