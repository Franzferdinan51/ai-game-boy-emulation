import React, { useState, useEffect, useCallback } from 'react';
import {
  ChatInterface,
  GameScreen,
  SidePanel,
  Header,
  SettingsModal
} from './components';
import {
  ChatMessage,
  EmulatorState,
  AISession,
  Settings,
  AIThinkingState,
  GameAction
} from './types';
import {
  ChatBubbleLeftRightIcon,
  Cog6ToothIcon,
  ArrowsPointingOutIcon,
  PauseIcon,
  PlayIcon,
  StopIcon
} from '@heroicons/react/24/outline';
import { Sparkles } from 'lucide-react';

const API_BASE_URL = 'http://localhost:5000';

const GoogleAIStudioApp: React.FC = () => {
  // Core state
  const [sessions, setSessions] = useState<AISession[]>([]);
  const [currentSession, setCurrentSession] = useState<AISession | null>(null);
  const [emulatorState, setEmulatorState] = useState<EmulatorState>({
    type: 'gb',
    romName: null,
    isRunning: false,
    screenData: null,
    streamingStatus: 'disconnected',
    fps: 0
  });
  const [aiThinking, setAiThinking] = useState<AIThinkingState>({
    isThinking: false,
    progress: 0
  });

  // UI state
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [activeView, setActiveView] = useState<'chat' | 'game' | 'both'>('both');

  // Settings
  const [settings, setSettings] = useState<Settings>({
    theme: 'dark',
    aiProvider: 'gemini',
    aiModel: 'gemini-pro',
    autoSave: true,
    notifications: true,
    streamingQuality: 'high'
  });

  // Event source for streaming
  const [eventSource, setEventSource] = useState<EventSource | null>(null);

  // Initialize or create new session
  useEffect(() => {
    if (!currentSession) {
      createNewSession();
    }
  }, [currentSession]);

  // Setup SSE connection when ROM is loaded
  useEffect(() => {
    if (emulatorState.romName && emulatorState.streamingStatus === 'disconnected') {
      setupSSEConnection();
    }

    return () => {
      if (eventSource) {
        eventSource.close();
        setEventSource(null);
      }
    };
  }, [emulatorState.romName, emulatorState.streamingStatus]);

  const createNewSession = useCallback(() => {
    const newSession: AISession = {
      id: `session-${Date.now()}`,
      provider: settings.aiProvider,
      model: settings.aiModel,
      messages: [],
      createdAt: new Date(),
      lastActive: new Date()
    };

    setCurrentSession(newSession);
    setSessions(prev => [newSession, ...prev]);
  }, [settings.aiProvider, settings.aiModel]);

  const setupSSEConnection = useCallback(() => {
    if (eventSource) {
      eventSource.close();
    }

    try {
      const source = new EventSource(`${API_BASE_URL}/api/stream`);

      source.onopen = () => {
        setEmulatorState(prev => ({ ...prev, streamingStatus: 'connecting' }));
      };

      source.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.image) {
            setEmulatorState(prev => ({
              ...prev,
              screenData: `data:image/jpeg;base64,${data.image}`,
              streamingStatus: 'connected',
              fps: data.fps || prev.fps
            }));
          }

          if (data.status === 'stream_started') {
            setEmulatorState(prev => ({ ...prev, streamingStatus: 'connected' }));
          }
        } catch (error) {
          console.error('SSE parse error:', error);
          setEmulatorState(prev => ({ ...prev, streamingStatus: 'error' }));
        }
      };

      source.onerror = () => {
        setEmulatorState(prev => ({ ...prev, streamingStatus: 'error' }));
        source.close();
      };

      setEventSource(source);
    } catch (error) {
      console.error('SSE connection error:', error);
      setEmulatorState(prev => ({ ...prev, streamingStatus: 'error' }));
    }
  }, [eventSource]);

  const handleSendMessage = useCallback(async (message: string, attachments?: File[]) => {
    if (!currentSession || !message.trim()) return;

    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: message,
      timestamp: new Date(),
      attachments: attachments?.map(file => ({
        type: file.type.startsWith('image/') ? 'image' : 'file',
        url: URL.createObjectURL(file),
        name: file.name
      }))
    };

    // Update session with user message
    const updatedSession = {
      ...currentSession,
      messages: [...currentSession.messages, userMessage],
      lastActive: new Date()
    };

    setCurrentSession(updatedSession);
    setSessions(prev =>
      prev.map(s => s.id === currentSession.id ? updatedSession : s)
    );

    // Show AI thinking state
    setAiThinking({
      isThinking: true,
      progress: 0,
      currentAction: 'Processing your message...'
    });

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          api_name: settings.aiProvider,
          model: settings.aiModel,
        })
      });

      if (!response.ok) {
        throw new Error('Failed to get AI response');
      }

      const data = await response.json();

      const aiMessage: ChatMessage = {
        id: `msg-${Date.now()}`,
        role: 'assistant',
        content: data.response,
        timestamp: new Date()
      };

      // Update session with AI response
      const finalSession = {
        ...updatedSession,
        messages: [...updatedSession.messages, aiMessage],
        lastActive: new Date()
      };

      setCurrentSession(finalSession);
      setSessions(prev =>
        prev.map(s => s.id === currentSession.id ? finalSession : s)
      );

    } catch (error) {
      console.error('Chat error:', error);

      const errorMessage: ChatMessage = {
        id: `msg-${Date.now()}`,
        role: 'assistant',
        content: 'I apologize, but I encountered an error. Please try again.',
        timestamp: new Date()
      };

      const errorSession = {
        ...updatedSession,
        messages: [...updatedSession.messages, errorMessage],
        lastActive: new Date()
      };

      setCurrentSession(errorSession);
      setSessions(prev =>
        prev.map(s => s.id === currentSession.id ? errorSession : s)
      );
    } finally {
      setAiThinking({ isThinking: false, progress: 0 });
    }
  }, [currentSession, settings]);

  const handleLoadROM = useCallback(async (file: File) => {
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append('rom_file', file);
      formData.append('emulator_type', emulatorState.type);

      const response = await fetch(`${API_BASE_URL}/api/upload-rom`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Failed to load ROM');
      }

      const result = await response.json();

      setEmulatorState(prev => ({
        ...prev,
        romName: file.name,
        streamingStatus: 'disconnected' // Will trigger SSE connection
      }));

      // Update current session with game context
      if (currentSession) {
        const updatedSession = {
          ...currentSession,
          gameState: {
            step: 0,
            location: 'Starting new game',
            objective: 'Exploring the game world'
          },
          lastActive: new Date()
        };

        setCurrentSession(updatedSession);
        setSessions(prev =>
          prev.map(s => s.id === currentSession.id ? updatedSession : s)
        );
      }

    } catch (error) {
      console.error('ROM load error:', error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, [currentSession, emulatorState.type]);

  const handleGameAction = useCallback(async (action: GameAction) => {
    if (!emulatorState.romName) return;

    try {
      const response = await fetch(`${API_BASE_URL}/api/action`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          action: action,
          frames: 10
        })
      });

      if (!response.ok) {
        throw new Error('Failed to execute action');
      }

      // Update game state in current session
      if (currentSession && currentSession.gameState) {
        const updatedSession = {
          ...currentSession,
          gameState: {
            ...currentSession.gameState,
            step: currentSession.gameState.step + 1
          },
          lastActive: new Date()
        };

        setCurrentSession(updatedSession);
        setSessions(prev =>
          prev.map(s => s.id === currentSession.id ? updatedSession : s)
        );
      }

    } catch (error) {
      console.error('Game action error:', error);
    }
  }, [emulatorState.romName, currentSession]);

  const handleStartAI = useCallback(() => {
    if (!currentSession || !emulatorState.romName) return;

    setEmulatorState(prev => ({ ...prev, isRunning: true }));

    // Add AI start message
    const startMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'assistant',
      content: '🎮 I\'m now analyzing the game and will start playing automatically!',
      timestamp: new Date()
    };

    const updatedSession = {
      ...currentSession,
      messages: [...currentSession.messages, startMessage],
      lastActive: new Date()
    };

    setCurrentSession(updatedSession);
    setSessions(prev =>
      prev.map(s => s.id === currentSession.id ? updatedSession : s)
    );
  }, [currentSession, emulatorState.romName]);

  const handleStopAI = useCallback(() => {
    setEmulatorState(prev => ({ ...prev, isRunning: false }));
    setAiThinking({ isThinking: false, progress: 0 });
  }, []);

  const handleSettingsSave = useCallback((newSettings: Settings) => {
    setSettings(newSettings);
    setIsSettingsOpen(false);

    // Update current session provider if changed
    if (currentSession && newSettings.aiProvider !== currentSession.provider) {
      const updatedSession = {
        ...currentSession,
        provider: newSettings.aiProvider,
        model: newSettings.aiModel,
        lastActive: new Date()
      };

      setCurrentSession(updatedSession);
      setSessions(prev =>
        prev.map(s => s.id === currentSession.id ? updatedSession : s)
      );
    }
  }, [currentSession]);

  return (
    <div className={`min-h-screen ${settings.theme === 'dark' ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'}`}>
      {/* Header */}
      <Header
        title="AI Game Studio"
        subtitle="Intelligent Game Boy Emulation"
        onOpenSettings={() => setIsSettingsOpen(true)}
        onCreateSession={createNewSession}
        currentSession={currentSession}
        sessions={sessions}
        onSessionSelect={setCurrentSession}
        theme={settings.theme}
      />

      {/* Main Content */}
      <div className="h-[calc(100vh-64px)] flex">
        {/* Chat Interface - Always visible on left */}
        <div className={`transition-all duration-300 ${
          activeView === 'chat' ? 'w-full' :
          activeView === 'both' ? 'w-1/2' : 'w-96'
        }`}>
          <ChatInterface
            session={currentSession}
            onSendMessage={handleSendMessage}
            aiThinking={aiThinking}
            theme={settings.theme}
          />
        </div>

        {/* Right Panel - Game or Controls */}
        {activeView !== 'chat' && (
          <div className={`flex-1 transition-all duration-300 ${
            activeView === 'game' ? 'w-full' : 'w-1/2'
          }`}>
            {activeView === 'both' ? (
              <div className="h-full flex flex-col">
                {/* Game Screen */}
                <div className="flex-1 p-4">
                  <GameScreen
                    emulatorState={emulatorState}
                    onLoadROM={handleLoadROM}
                    onGameAction={handleGameAction}
                    isFullscreen={isFullscreen}
                    onToggleFullscreen={() => setIsFullscreen(!isFullscreen)}
                    theme={settings.theme}
                  />
                </div>

                {/* AI Controls */}
                <div className="h-32 border-t border-gray-700 p-4">
                  <div className="flex items-center justify-between h-full">
                    <div className="flex items-center gap-4">
                      {!emulatorState.isRunning ? (
                        <button
                          onClick={handleStartAI}
                          disabled={!emulatorState.romName}
                          className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded-lg transition-colors"
                        >
                          <PlayIcon className="w-5 h-5" />
                          Start AI
                        </button>
                      ) : (
                        <button
                          onClick={handleStopAI}
                          className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-colors"
                        >
                          <StopIcon className="w-5 h-5" />
                          Stop AI
                        </button>
                      )}

                      <div className="flex items-center gap-2 text-sm">
                        <Sparkles className={`w-4 h-4 ${emulatorState.isRunning ? 'text-green-400 animate-pulse' : 'text-gray-400'}`} />
                        <span>{emulatorState.isRunning ? 'AI is playing' : 'AI is idle'}</span>
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => setActiveView('game')}
                        className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
                        title="Fullscreen Game"
                      >
                        <ArrowsPointingOutIcon className="w-5 h-5" />
                      </button>

                      <button
                        onClick={() => setActiveView('chat')}
                        className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
                        title="Chat Only"
                      >
                        <ChatBubbleLeftRightIcon className="w-5 h-5" />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <GameScreen
                emulatorState={emulatorState}
                onLoadROM={handleLoadROM}
                onGameAction={handleGameAction}
                isFullscreen={true}
                onToggleFullscreen={() => setIsFullscreen(!isFullscreen)}
                theme={settings.theme}
              />
            )}
          </div>
        )}
      </div>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        onSave={handleSettingsSave}
        theme={settings.theme}
      />

      {/* Loading Overlay */}
      {isLoading && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="text-white text-lg">Loading...</div>
        </div>
      )}
    </div>
  );
};

export default GoogleAIStudioApp;