import React, { useState, useRef, useEffect, useCallback } from 'react';
import Header from './components/Header';
import EmulatorScreen from './components/EmulatorScreen';
import Controls from './components/Controls';
import AIPanel from './components/AIPanel';
import EnhancedChatInterface from './components/EnhancedChatInterface';
import CollapsibleSidebar from './components/CollapsibleSidebar';
import AIPersonalityAvatar, { AIPersonalityState } from './components/AIPersonalityAvatar';
import { GameStateDisplay } from './components/PokemonGameUI';
import SettingsModal from './components/SettingsModal';
import type { EmulatorMode, AIState, GameAction, AILog, ChatMessage, AppSettings } from './types';
import { EmulatorMode as EmulatorModeEnum, AIState as AIStateEnum } from './types';

const SERVER_URL = 'http://localhost:5000';

const App: React.FC = () => {
  const [emulatorMode, setEmulatorMode] = useState<EmulatorMode>(EmulatorModeEnum.GB);
  const [romName, setRomName] = useState<string | null>(null);
  const [aiState, setAiState] = useState<AIState>(AIStateEnum.IDLE);
  const [aiGoal, setAiGoal] = useState<string>('');
  const [aiLogs, setAiLogs] = useState<AILog[]>([]);
  const [lastAIAction, setLastAIAction] = useState<GameAction | null>(null);
  const [screenImage, setScreenImage] = useState<string>('');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState<string>('');
  const [isChatting, setIsChatting] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);
  const [eventSource, setEventSource] = useState<EventSource | null>(null);
  const [streamingStatus, setStreamingStatus] = useState<'disconnected' | 'connecting' | 'connected' | 'error' | 'failed'>('disconnected');
  const [streamingInfo, setStreamingInfo] = useState<{ fps: number; frameCount: number }>({ fps: 0, frameCount: 0 });
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);
  const [appSettings, setAppSettings] = useState<AppSettings>(() => {
    const savedSettings = localStorage.getItem('appSettings');
    if (savedSettings) {
      return JSON.parse(savedSettings);
    }
    return {
      aiActionInterval: 5000,
      apiProvider: 'gemini',
    };
  });

  // Enhanced state for new features
  const [useEnhancedUI, setUseEnhancedUI] = useState<boolean>(true);

  // Toggle for enhanced UI
  const toggleEnhancedUI = useCallback(() => {
    setUseEnhancedUI(prev => !prev);
  }, []);
  const [aiPersonality, setAiPersonality] = useState<AIPersonalityState>('neutral');
  const [gameStats, setGameStats] = useState({
    step: 0,
    location: 'Unknown',
    isPlaying: false
  });
  const [enhancedMessages, setEnhancedMessages] = useState<Array<{
    id: string;
    role: 'user' | 'ai';
    content: string;
    timestamp?: Date;
    aiState?: AIPersonalityState;
  }>>([]);

  
  const gameLoopRef = useRef<number | null>(null);
  const actionHistoryRef = useRef<string[]>([]);
  const logIdCounter = useRef<number>(0);
  const chatIdCounter = useRef<number>(0);
  const enhancedMessageIdCounter = useRef<number>(0);

  const handleOpenSettings = () => setIsSettingsOpen(true);
  const handleCloseSettings = () => setIsSettingsOpen(false);

  const handleSaveSettings = (newSettings: AppSettings) => {
    setAppSettings(newSettings);
    localStorage.setItem('appSettings', JSON.stringify(newSettings));
    if (gameLoopRef.current) {
      stopAI();
    }
    handleCloseSettings();
  };

  // Enhanced functions
  const determineAIState = (content: string): AIPersonalityState => {
    const lowerContent = content.toLowerCase();
    if (lowerContent.includes('think') || lowerContent.includes('considering') || lowerContent.includes('maybe')) {
      return 'thinking';
    } else if (lowerContent.includes('read') || lowerContent.includes('check') || lowerContent.includes('analyze')) {
      return 'reading';
    } else if (lowerContent.includes('hehe') || lowerContent.includes('trick') || lowerContent.includes('surprise')) {
      return 'mischievous';
    } else if (lowerContent.includes('fun') || lowerContent.includes('play') || lowerContent.includes('game')) {
      return 'playful';
    } else if (lowerContent.includes('definitely') || lowerContent.includes('certain') || lowerContent.includes('sure')) {
      return 'confident';
    }
    return 'neutral';
  };

  const handleEnhancedSendMessage = useCallback(async (message: string) => {
    if (!message.trim()) return;

    const userMessage = {
      id: `msg-${enhancedMessageIdCounter.current++}`,
      role: 'user' as const,
      content: message,
      timestamp: new Date()
    };

    setEnhancedMessages(prev => [...prev, userMessage]);

    try {
      const response = await fetch(`${SERVER_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          api_name: appSettings.apiProvider,
          api_endpoint: appSettings.apiEndpoint,
          api_key: appSettings.apiKey,
          model: appSettings.model,
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        if (errorData.error && errorData.error.includes('not available') && errorData.available_providers) {
          const firstAvailableProvider = errorData.available_providers[0];
          const newSettings = {
            ...appSettings,
            apiProvider: firstAvailableProvider as AppSettings['apiProvider']
          };
          setAppSettings(newSettings);
          localStorage.setItem('appSettings', JSON.stringify(newSettings));

          const retryResponse = await fetch(`${SERVER_URL}/api/chat`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              message: message,
              api_name: firstAvailableProvider,
              api_endpoint: newSettings.apiEndpoint,
              api_key: newSettings.apiKey,
              model: newSettings.model,
            })
          });

          if (!retryResponse.ok) {
            throw new Error('Failed to get chat response');
          }

          const retryData = await retryResponse.json();
          const aiMessage = {
            id: `msg-${enhancedMessageIdCounter.current++}`,
            role: 'ai' as const,
            content: retryData.response,
            timestamp: new Date(),
            aiState: determineAIState(retryData.response)
          };
          setEnhancedMessages(prev => [...prev, aiMessage]);
        } else {
          throw new Error(errorData.error || 'Failed to get chat response');
        }
      } else {
        const data = await response.json();
        const aiMessage = {
          id: `msg-${enhancedMessageIdCounter.current++}`,
          role: 'ai' as const,
          content: data.response,
          timestamp: new Date(),
          aiState: determineAIState(data.response)
        };
        setEnhancedMessages(prev => [...prev, aiMessage]);
      }
    } catch (error) {
      console.error('Error sending chat message:', error);
      const errorMessage = {
        id: `msg-${enhancedMessageIdCounter.current++}`,
        role: 'ai' as const,
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
        aiState: 'mischievous' as AIPersonalityState
      };
      setEnhancedMessages(prev => [...prev, errorMessage]);
    }
  }, [appSettings]);

  // Sidebar configuration
  const sidebarSections = [
    {
      title: 'Game Controls',
      items: [
        { id: 'rom', icon: '📁', label: 'Load ROM', onClick: () => {} },
        { id: 'start', icon: '▶️', label: 'Start AI', onClick: () => aiState === AIStateEnum.IDLE && startAI() },
        { id: 'stop', icon: '⏹️', label: 'Stop AI', onClick: () => aiState !== AIStateEnum.IDLE && stopAI() },
        { id: 'reset', icon: '🔄', label: 'Reset Game', onClick: () => {} },
      ]
    },
    {
      title: 'AI Features',
      items: [
        { id: 'chat', icon: '💬', label: 'Chat Interface', onClick: () => {} },
        { id: 'analysis', icon: '🧠', label: 'Game Analysis', onClick: () => {} },
        { id: 'strategy', icon: '🎯', label: 'Strategy Mode', onClick: () => {} },
        { id: 'learning', icon: '📚', label: 'Learning Hub', onClick: () => {} },
      ]
    },
    {
      title: 'External',
      items: [
        { id: 'discord', icon: '💎', label: 'Discord', onClick: () => {} },
      ]
    }
  ];

  useEffect(() => {
    let source: EventSource | null = null;
    let reconnectTimeout: NodeJS.Timeout | null = null;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;

    const connectSSE = () => {
      if (!romName) return;

      try {
        source = new EventSource(`${SERVER_URL}/api/stream`);

        source.onopen = () => {
          console.log('SSE connection established');
          reconnectAttempts = 0;
        };

        source.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);

            if (data.error) {
              console.error('SSE stream error:', data.error);
              setStreamingStatus('error');
              return;
            }

            if (data.image) {
              setScreenImage(`data:image/jpeg;base64,${data.image}`);
              setStreamingStatus('connected');

              // Update FPS counter if available
              if (data.fps) {
                setStreamingInfo(prev => ({ ...prev, fps: data.fps }));
              }
            }

            if (data.status === 'stream_started') {
              console.log('SSE stream started successfully');
              setStreamingStatus('connected');
            }
          } catch (e) {
            console.error('SSE parse error:', e);
            setStreamingStatus('error');
          }
        };

        source.onerror = (err) => {
          console.error('SSE error:', err);
          setStreamingStatus('disconnected');
          source?.close();

          // Attempt to reconnect
          if (reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            console.log(`Attempting to reconnect... (${reconnectAttempts}/${maxReconnectAttempts})`);
            reconnectTimeout = setTimeout(connectSSE, Math.min(1000 * Math.pow(2, reconnectAttempts), 30000));
          } else {
            console.error('Max reconnection attempts reached');
            setStreamingStatus('failed');
          }
        };

        setEventSource(source);
      } catch (error) {
        console.error('Failed to create SSE connection:', error);
        setStreamingStatus('failed');
      }
    };

    connectSSE();

    return () => {
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
      source?.close();
      setEventSource(null);
      setStreamingStatus('disconnected');
    };
  }, [romName]);

  const addLog = useCallback((message: string, type: AILog['type']) => {
    setAiLogs(prev => [...prev, { id: logIdCounter.current++, message, type }]);
  }, []);

  const addChatMessage = useCallback((message: string, sender: 'user' | 'ai') => {
    setChatHistory(prev => [...prev, { id: chatIdCounter.current++, text: message, sender }]);
  }, []);

  // Validate and fix provider settings on startup
  useEffect(() => {
    const validateProvider = async () => {
      try {
        const response = await fetch(`${SERVER_URL}/api/providers/status`);
        if (response.ok) {
          const providerStatus = await response.json();
          const availableProviders = Object.entries(providerStatus)
            .filter(([_, info]: [string, any]) => info.available)
            .map(([name, _]: [string, any]) => name);

          if (availableProviders.length > 0 && !availableProviders.includes(appSettings.apiProvider)) {
            // Current provider is not available, switch to the first available one
            const newProvider = availableProviders[0] as AppSettings['apiProvider'];
            const newSettings = {
              ...appSettings,
              apiProvider: newProvider
            };
            setAppSettings(newSettings);
            localStorage.setItem('appSettings', JSON.stringify(newSettings));
            addLog(`Auto-switched from unavailable provider to '${newProvider}'`, 'info');
          }
        }
      } catch (error) {
        console.error('Error validating provider:', error);
      }
    };

    validateProvider();
  }, [addLog]); // Run only once on startup

  // Function to update the screen from the server
  const updateScreen = useCallback(async () => {
    try {
      const response = await fetch(`${SERVER_URL}/api/screen`);
      if (!response.ok) {
        throw new Error('Failed to get screen');
      }
      
      const data = await response.json();
      setScreenImage(`data:image/jpeg;base64,${data.image}`);
    } catch (error) {
      console.error('Error updating screen:', error);
    }
  }, []);

  // Function to load a ROM
  const loadRom = useCallback(async (file: File) => {
    try {
      const formData = new FormData();
      formData.append('rom_file', file);
      formData.append('emulator_type', emulatorMode);
      
      setLoading(true);
      addLog(`Uploading ROM: ${file.name}`, 'info');
      
      const response = await fetch(`${SERVER_URL}/api/upload-rom`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Upload failed');
      }
      
      const result = await response.json();
      setRomName(file.name);
      addLog(`Loaded ROM: ${file.name}`, 'info');
      console.log('ROM loaded:', result);
      
      await updateScreen();
      setLoading(false);
    } catch (error) {
      console.error('Error loading ROM:', error);
      addLog(`Error loading ROM: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
      setLoading(false);
    }
  }, [emulatorMode, addLog, updateScreen]);

  // Function to execute an action
  const executeAction = useCallback(async (action: string) => {
    try {
      const response = await fetch(`${SERVER_URL}/api/action`, {
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
      
      await updateScreen();
    } catch (error) {
      console.error('Error executing action:', error);
      addLog(`Error executing action: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
    }
  }, [addLog, updateScreen]);

  // Function to get AI next move
  const getAINextMove = useCallback(async (goal: string) => {
    try {
      const response = await fetch(`${SERVER_URL}/api/ai-action`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          api_name: appSettings.apiProvider,
          api_endpoint: appSettings.apiEndpoint,
          api_key: appSettings.apiKey,
          model: appSettings.model,
          goal: goal
        })
      });

      if (!response.ok) {
        const errorData = await response.json();

        // Check if the error is about an unavailable provider
        if (errorData.error && errorData.error.includes('not available') && errorData.available_providers) {
          // Automatically switch to the first available provider
          const firstAvailableProvider = errorData.available_providers[0];
          addLog(`Provider '${appSettings.apiProvider}' not available. Switching to '${firstAvailableProvider}'`, 'info');

          // Update settings to use the available provider
          const newSettings = {
            ...appSettings,
            apiProvider: firstAvailableProvider as AppSettings['apiProvider']
          };
          setAppSettings(newSettings);
          localStorage.setItem('appSettings', JSON.stringify(newSettings));

          // Retry with the new provider
          const retryResponse = await fetch(`${SERVER_URL}/api/ai-action`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              api_name: firstAvailableProvider,
              api_endpoint: newSettings.apiEndpoint,
              api_key: newSettings.apiKey,
              model: newSettings.model,
              goal: goal
            })
          });

          if (!retryResponse.ok) {
            throw new Error('Failed to get AI action even with available provider');
          }

          const retryData = await retryResponse.json();
          return retryData.action;
        }

        throw new Error(errorData.error || 'Failed to get AI action');
      }

      const data = await response.json();
      return data.action;
    } catch (error) {
      console.error('Error getting AI action:', error);
      addLog(`Error getting AI action: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
      return 'SELECT'; // Default safe action
    }
  }, [addLog, appSettings]);

  const runAI = useCallback(async () => {
    if (!romName) {
      addLog("Cannot start AI: No ROM loaded.", 'error');
      setAiState(AIStateEnum.IDLE);
      return;
    }
    setAiState(AIStateEnum.THINKING);
    setAiPersonality('thinking');
    addLog("AI is thinking...", 'thought');

    try {
      const nextAction = await getAINextMove(aiGoal);

      actionHistoryRef.current.push(nextAction);
      addLog(`AI chose action: ${nextAction}`, 'action');
      setLastAIAction(nextAction as GameAction);

      await executeAction(nextAction);

      setTimeout(() => setLastAIAction(null), 500);
      setAiState(AIStateEnum.RUNNING);
      setAiPersonality('confident');

    } catch (error) {
      console.error(error);
      const errorMessage = error instanceof Error ? error.message : "An unknown error occurred.";
      addLog(`Error: ${errorMessage}`, 'error');
      setAiState(AIStateEnum.ERROR);
      setAiPersonality('mischievous');
      if (gameLoopRef.current) {
        clearInterval(gameLoopRef.current);
        gameLoopRef.current = null;
      }
    }
  }, [romName, aiGoal, addLog, getAINextMove, executeAction]);

  const startAI = () => {
    if (gameLoopRef.current) return;
    setAiLogs([]);
    actionHistoryRef.current = [];
    addLog(`AI started with objective: "${aiGoal}"`, 'info');
    setAiState(AIStateEnum.RUNNING);
    setAiPersonality('confident');
    runAI();
    gameLoopRef.current = window.setInterval(runAI, appSettings.aiActionInterval);
  };

  const stopAI = () => {
    if (gameLoopRef.current) {
      clearInterval(gameLoopRef.current);
      gameLoopRef.current = null;
    }
    addLog("AI stopped by user.", 'info');
    setAiState(AIStateEnum.IDLE);
    setAiPersonality('neutral');
  };
  
  const handleRomLoad = (file: File) => {
    setRomName(file.name);
    addLog(`Loaded ROM: ${file.name}`, 'info');
    loadRom(file);
  };

  const handleSendMessage = useCallback(async (message: string) => {
    if (!message.trim()) return;

    addChatMessage(message, 'user');
    setChatInput('');
    setIsChatting(true);

    try {
      const response = await fetch(`${SERVER_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          api_name: appSettings.apiProvider,
          api_endpoint: appSettings.apiEndpoint,
          api_key: appSettings.apiKey,
          model: appSettings.model,
        })
      });

      if (!response.ok) {
        const errorData = await response.json();

        // Check if the error is about an unavailable provider
        if (errorData.error && errorData.error.includes('not available') && errorData.available_providers) {
          // Automatically switch to the first available provider
          const firstAvailableProvider = errorData.available_providers[0];
          addLog(`Provider '${appSettings.apiProvider}' not available. Switching to '${firstAvailableProvider}' for chat`, 'info');

          // Update settings to use the available provider
          const newSettings = {
            ...appSettings,
            apiProvider: firstAvailableProvider as AppSettings['apiProvider']
          };
          setAppSettings(newSettings);
          localStorage.setItem('appSettings', JSON.stringify(newSettings));

          // Retry with the new provider
          const retryResponse = await fetch(`${SERVER_URL}/api/chat`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              message: message,
              api_name: firstAvailableProvider,
              api_endpoint: newSettings.apiEndpoint,
              api_key: newSettings.apiKey,
              model: newSettings.model,
            })
          });

          if (!retryResponse.ok) {
            throw new Error('Failed to get chat response even with available provider');
          }

          const retryData = await retryResponse.json();
          addChatMessage(retryData.response, 'ai');
        } else {
          throw new Error(errorData.error || 'Failed to get chat response');
        }
      } else {
        const data = await response.json();
        addChatMessage(data.response, 'ai');
      }
    } catch (error) {
      console.error('Error sending chat message:', error);
      addChatMessage('Sorry, I encountered an error. Please try again.', 'ai');
    } finally {
      setIsChatting(false);
    }
  }, [addChatMessage, addLog, appSettings]);

  useEffect(() => {
    // Add event listener for manual control input
    const handleControlPress = (event: CustomEvent) => {
      const action = event.detail;
      if (romName) {
        executeAction(action);
        // Briefly show the action as active for visual feedback
        setLastAIAction(action as GameAction);
        setTimeout(() => setLastAIAction(null), 200);
      }
    };

    window.addEventListener('game-control-press', handleControlPress as EventListener);

    return () => {
      if (gameLoopRef.current) {
        clearInterval(gameLoopRef.current);
      }
      window.removeEventListener('game-control-press', handleControlPress as EventListener);
    };
  }, [romName, executeAction]);

  // Update game stats based on AI state
  useEffect(() => {
    if (aiState === AIStateEnum.RUNNING) {
      setGameStats(prev => ({
        ...prev,
        step: prev.step + 1,
        isPlaying: true
      }));
    } else {
      setGameStats(prev => ({
        ...prev,
        isPlaying: false
      }));
    }
  }, [aiState]);

  return (
    <div className="h-screen w-screen flex bg-gray-900 text-white">
      {/* Enhanced Sidebar */}
      {useEnhancedUI && <CollapsibleSidebar sections={sidebarSections} />}

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        <Header
          emulatorMode={emulatorMode}
          onModeChange={setEmulatorMode}
          onOpenSettings={handleOpenSettings}
          useEnhancedUI={useEnhancedUI}
          onToggleEnhancedUI={toggleEnhancedUI}
        />

        <div className="flex-1 flex overflow-hidden">
          {/* Game Area */}
          <div className="flex-1 flex flex-col p-4">
            <div className="h-full flex flex-col bg-gray-800 rounded-lg shadow-xl">
              {/* Game Stats Bar */}
              {useEnhancedUI && (
                <div className="p-4 border-b border-gray-700">
                  <GameStateDisplay
                    step={gameStats.step}
                    location={gameStats.location}
                    isPlaying={aiState === AIStateEnum.RUNNING}
                  />
                </div>
              )}

              {/* Emulator Screen */}
              <div className="flex-1 p-4">
                <EmulatorScreen
                  emulatorMode={emulatorMode}
                  romName={romName}
                  onRomLoad={handleRomLoad}
                  aiState={aiState}
                  screenImage={screenImage}
                  streamingStatus={streamingStatus}
                  streamingInfo={streamingInfo}
                />
              </div>

              {/* Controls */}
              <div className="p-4 border-t border-gray-700">
                <Controls lastAction={lastAIAction} />
              </div>
            </div>
          </div>

          {/* Right Panel - Enhanced Chat or Traditional AIPanel */}
          {useEnhancedUI ? (
            <div className="w-96 bg-gray-900 border-l border-gray-800 flex flex-col">
              {/* AI Personality Header */}
              <div className="p-4 border-b border-gray-800">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <AIPersonalityAvatar state={aiPersonality} size="md" />
                    <div>
                      <h3 className="font-semibold">AI Assistant</h3>
                      <p className="text-sm text-gray-400 capitalize">{aiPersonality}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-xs text-gray-500">Status</div>
                    <div className="text-sm font-medium">
                      {aiState === AIStateEnum.RUNNING ? '🟢 Playing' :
                       aiState === AIStateEnum.THINKING ? '🟡 Thinking' :
                       aiState === AIStateEnum.ERROR ? '🔴 Error' : '⚪ Idle'}
                    </div>
                  </div>
                </div>
              </div>

              {/* Enhanced Chat Interface */}
              <EnhancedChatInterface
                messages={enhancedMessages}
                onSendMessage={handleEnhancedSendMessage}
                gameState={{
                  step: gameStats.step,
                  location: gameStats.location,
                  isPlaying: aiState === AIStateEnum.RUNNING
                }}
              />
            </div>
          ) : (
            <AIPanel
              aiState={aiState}
              aiLogs={aiLogs}
              aiGoal={aiGoal}
              chatHistory={chatHistory}
              chatInput={chatInput}
              isChatting={isChatting}
              onGoalChange={setAiGoal}
              onStart={startAI}
              onStop={stopAI}
              onChatInputChange={setChatInput}
              onSendMessage={() => handleSendMessage(chatInput)}
            />
          )}
        </div>
      </div>

      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={handleCloseSettings}
        currentSettings={appSettings}
        onSave={handleSaveSettings}
      />
    </div>
  );
};

export default App;
