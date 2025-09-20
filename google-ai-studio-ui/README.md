# Google AI Studio UI - AI Game Boy Emulator Interface

A modern, Google AI Studio-inspired interface for AI-powered Game Boy emulation. This UI is designed to be a separate, clonable component that can be updated independently from the main application.

## 🎯 Features

### Core Functionality
- **Modern Chat Interface**: Clean, Google AI Studio-style messaging with AI assistant
- **Game Screen Integration**: Real-time Game Boy/GBC/GBA emulator display with SSE streaming
- **AI Provider Support**: Multiple AI providers (Gemini, OpenRouter, NVIDIA, OpenAI, Mock)
- **Session Management**: Persistent chat sessions with game state tracking
- **Responsive Design**: Works on desktop and mobile devices

### UI Components
- **Message Bubbles**: Rich text messaging with attachments and reactions
- **Streaming Indicator**: Real-time connection status and FPS counter
- **Game Controls**: Virtual D-pad and action buttons
- **Settings Modal**: Comprehensive configuration options
- **Dark/Light Theme**: Automatic theme switching

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- Your AI Game Boy emulator server running on `http://localhost:5000`

### Installation

1. **Navigate to the UI directory:**
   ```bash
   cd google-ai-studio-ui
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start development server:**
   ```bash
   npm run dev
   ```

4. **Open browser:**
   Navigate to `http://localhost:3000`

### Building for Production

```bash
npm run build
npm run preview
```

## 🏗️ Project Structure

```
google-ai-studio-ui/
├── src/
│   ├── components/           # React components
│   │   ├── App.tsx          # Main application component
│   │   ├── ChatInterface.tsx # Chat messaging interface
│   │   ├── GameScreen.tsx   # Emulator display component
│   │   ├── MessageBubble.tsx # Individual message component
│   │   ├── Header.tsx       # Application header
│   │   ├── SettingsModal.tsx # Settings configuration
│   │   └── ...              # Supporting components
│   ├── types.ts             # TypeScript type definitions
│   ├── main.tsx             # Application entry point
│   └── index.css            # Global styles
├── package.json             # Dependencies and scripts
├── vite.config.ts          # Vite configuration
├── tailwind.config.js      # Tailwind CSS configuration
└── README.md               # This file
```

## 🔧 Configuration

### Backend Connection
The UI automatically connects to your emulator server at `http://localhost:5000`. To change this:

1. Update `API_BASE_URL` in `src/App.tsx`
2. Or configure proxy in `vite.config.ts`

### Environment Variables
Create a `.env` file for custom configuration:

```env
VITE_API_BASE_URL=http://localhost:5000
VITE_APP_TITLE=AI Game Studio
```

## 🎮 Usage

### Loading Games
1. **Drag & Drop**: Drag ROM files (.gb, .gbc, .gba, .zip) onto the game screen
2. **File Browser**: Click "Browse ROM Files" button
3. **Auto-connect**: SSE streaming starts automatically when ROM is loaded

### AI Interaction
1. **Chat**: Send messages to AI assistant for game help and strategy
2. **Auto-play**: Enable AI to automatically play the game
3. **Manual Control**: Use on-screen controls or keyboard (Arrow keys, Z/X)

### Session Management
- **Create New Session**: Start fresh conversation
- **Switch Sessions**: Access previous conversations
- **Auto-save**: Sessions automatically saved (configurable)

## 🎨 Customization

### Theming
- **Automatic**: Follows system preference
- **Manual**: Dark/Light mode in settings
- **Custom**: Modify Tailwind CSS configuration

### Styling
- **Components**: Individual component styling
- **Global**: Override in `src/index.css`
- **Theme**: Colors and spacing in `tailwind.config.js`

## 🔄 Updates & Maintenance

### Independent Updates
This UI is designed to be updated separately from your main emulator:

1. **UI Changes**: Modify components in `src/components/`
2. **Styling**: Update CSS and Tailwind configuration
3. **Dependencies**: Update via `package.json`

### Integration Points
The UI integrates with your backend through standardized API endpoints:
- `/api/chat` - AI messaging
- `/api/upload-rom` - ROM file upload
- `/api/stream` - Screen streaming via SSE
- `/api/action` - Game control inputs
- `/api/providers/status` - AI provider availability

## 🛠️ Development

### Adding New Features
1. **New Component**: Create in `src/components/`
2. **Types**: Update `src/types.ts`
3. **Styling**: Use Tailwind classes
4. **Testing**: Test with backend server

### Code Style
- **TypeScript**: Strict typing enabled
- **React**: Functional components with hooks
- **Tailwind**: Utility-first CSS
- **Accessibility**: ARIA labels and keyboard navigation

## 📱 Mobile Support

The interface is fully responsive and includes:
- **Touch Controls**: Virtual gamepad for mobile
- **Adaptive Layout**: Resizes for different screens
- **Gesture Support**: Swipe actions where appropriate

## 🔮 Future Enhancements

This clonable UI structure supports easy addition of:
- **New AI Providers**: Add to provider selection
- **Game Plugins**: Extend for specific games
- **Analytics**: Track AI performance and game progress
- **Multiplayer**: Real-time collaborative features
- **Cloud Save**: Sync sessions across devices

## 🤝 Contributing

When updating this UI:
1. **Test**: Verify backend integration works
2. **Document**: Update README and comments
3. **Style**: Follow existing patterns
4. **Access**: Ensure keyboard and screen reader support

---

*This UI is designed to be easily maintainable and updatable as your Google AI Studio interface evolves.*