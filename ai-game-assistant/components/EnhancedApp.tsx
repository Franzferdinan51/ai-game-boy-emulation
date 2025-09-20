import React, { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Header from './Header';
import EmulatorScreen from './EmulatorScreen';
import Controls from './Controls';
import EnhancedChatInterface from './EnhancedChatInterface';
import CollapsibleSidebar from './CollapsibleSidebar';
import AIPersonalityAvatar, { AIPersonalityState } from './AIPersonalityAvatar';
import { PokemonCard, TypeBadge, GameStateDisplay } from './PokemonGameUI';
import SettingsModal from './SettingsModal';
import { ErrorBoundary } from './ErrorBoundary';
import { LoadingStates } from './LoadingStates';
import { useTheme } from './ThemeProvider';
import { TouchControls, GestureControls } from './TouchControls';
import type { EmulatorMode, AIState, GameAction, AILog, AppSettings } from '../types';
import { EmulatorMode as EmulatorModeEnum, AIState as AIStateEnum } from '../types';
import { Menu, X, Gamepad2, MessageSquare, Settings, Moon, Sun, Monitor } from 'lucide-react';

const SERVER_URL = 'http://localhost:5000';

interface EnhancedMessage {
  id: string;
  role: 'user' | 'ai';
  content: string;
  timestamp?: Date;
  aiState?: AIPersonalityState;
}

const EnhancedApp: React.FC = () => {
  // Core state
  const [emulatorMode, setEmulatorMode] = useState<EmulatorMode>(EmulatorModeEnum.GB);
  const [romName, setRomName] = useState<string | null>(null);
  const [aiState, setAiState] = useState<AIState>(AIStateEnum.IDLE);
  const [aiGoal, setAiGoal] = useState<string>('');
  const [aiLogs, setAiLogs] = useState<AILog[]>([]);
  const [lastAIAction, setLastAIAction] = useState<GameAction | null>(null);
  const [screenImage, setScreenImage] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [eventSource, setEventSource] = useState<EventSource | null>(null);
  const [streamingStatus, setStreamingStatus] = useState<'disconnected' | 'connecting' | 'connected' | 'error' | 'failed'>('disconnected');
  const [streamingInfo, setStreamingInfo] = useState<{ fps: number; frameCount: number }>({ fps: 0, frameCount: 0 });
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);

  // UI state
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

  // Mobile and accessibility state
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState<boolean>(false);
  const [isChatOpen, setIsChatOpen] = useState<boolean>(false);
  const [activePanel, setActivePanel] = useState<'game' | 'chat'>('game');
  const [keyboardShortcutsEnabled, setKeyboardShortcutsEnabled] = useState<boolean>(true);

  const { currentTheme, theme, toggleTheme } = useTheme();

  // Enhanced state
  const [messages, setMessages] = useState<EnhancedMessage[]>([]);
  const [gameStats, setGameStats] = useState({
    step: 0,
    location: 'Unknown',
    isPlaying: false,
    playerPokemon: null as any,
    enemyPokemon: null as any
  });
  const [aiPersonality, setAiPersonality] = useState<AIPersonalityState>('neutral');

  const gameLoopRef = useRef<number | null>(null);
  const actionHistoryRef = useRef<string[]>([]);
  const logIdCounter = useRef<number>(0);
  const messageIdCounter = useRef<number>(0);

  // Theme toggle handler
  const getThemeIcon = () => {
    switch (theme) {
      case 'light': return <Sun className="w-5 h-5" />;
      case 'dark': return <Moon className="w-5 h-5" />;
      case 'system': return <Monitor className="w-5 h-5" />;
      default: return <Monitor className="w-5 h-5" />;
    }
  };

  // Handlers
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

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (!keyboardShortcutsEnabled) return;

      // Toggle chat panel
      if ((event.ctrlKey || event.metaKey) && event.key === 'c') {
        event.preventDefault();
        setIsChatOpen(prev => !prev);
      }

      // Toggle mobile menu
      if ((event.ctrlKey || event.metaKey) && event.key === 'm') {
        event.preventDefault();
        setIsMobileMenuOpen(prev => !prev);
      }

      // Toggle theme
      if ((event.ctrlKey || event.metaKey) && event.key === 't') {
        event.preventDefault();
        toggleTheme();
      }

      // Quick actions
      if ((event.ctrlKey || event.metaKey) && event.key === 's') {
        event.preventDefault();
        aiState === AIStateEnum.IDLE ? startAI() : stopAI();
      }

      // Escape to close modals
      if (event.key === 'Escape') {
        setIsSettingsOpen(false);
        setIsMobileMenuOpen(false);
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [keyboardShortcutsEnabled, aiState, toggleTheme]);

  // Responsive behavior
  useEffect(() => {
    const handleResize = () => {
      const isMobile = window.innerWidth < 768;
      if (isMobile && isChatOpen) {
        setActivePanel('chat');
      } else if (!isMobile) {
        setActivePanel('game');
        setIsChatOpen(true);
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [isChatOpen]);

  // Enhanced chat message handling
  const handleSendMessage = useCallback(async (message: string) => {
    if (!message.trim()) return;

    const userMessage: EnhancedMessage = {
      id: `msg-${messageIdCounter.current++}`,
      role: 'user',
      content: message,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);

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

          // Update settings
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
          const aiMessage: EnhancedMessage = {
            id: `msg-${messageIdCounter.current++}`,
            role: 'ai',
            content: retryData.response,
            timestamp: new Date(),
            aiState: determineAIState(retryData.response)
          };
          setMessages(prev => [...prev, aiMessage]);
        } else {
          throw new Error(errorData.error || 'Failed to get chat response');
        }
      } else {
        const data = await response.json();
        const aiMessage: EnhancedMessage = {
          id: `msg-${messageIdCounter.current++}`,
          role: 'ai',
          content: data.response,
          timestamp: new Date(),
          aiState: determineAIState(data.response)
        };
        setMessages(prev => [...prev, aiMessage]);
      }
    } catch (error) {
      console.error('Error sending chat message:', error);
      const errorMessage: EnhancedMessage = {
        id: `msg-${messageIdCounter.current++}`,
        role: 'ai',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
        aiState: 'mischievous'
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  }, [appSettings]);

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

  // Enhanced sidebar configuration with mobile support
  const sidebarSections = [
    {
      title: 'Game Controls',
      items: [
        {
          id: 'rom',
          icon: '📁',
          label: 'Load ROM',
          onClick: () => handleRomLoad(new File([], '', { type: 'application/octet-stream' })),
          description: 'Load a game ROM file',
          shortcut: 'Ctrl+L'
        },
        {
          id: 'start',
          icon: '▶️',
          label: 'Start AI',
          onClick: () => aiState === AIStateEnum.IDLE && startAI(),
          description: 'Start AI playing the game',
          shortcut: 'Ctrl+S',
          disabled: aiState !== AIStateEnum.IDLE
        },
        {
          id: 'stop',
          icon: '⏹️',
          label: 'Stop AI',
          onClick: () => aiState !== AIStateEnum.IDLE && stopAI(),
          description: 'Stop AI gameplay',
          shortcut: 'Ctrl+S',
          disabled: aiState === AIStateEnum.IDLE
        },
        {
          id: 'reset',
          icon: '🔄',
          label: 'Reset Game',
          onClick: () => {}, // TODO: Implement reset
          description: 'Reset the current game',
          shortcut: 'Ctrl+R'
        },
      ]
    },
    {
      title: 'AI Features',
      items: [
        {
          id: 'chat',
          icon: '💬',
          label: 'Chat Interface',
          onClick: () => setIsChatOpen(true),
          description: 'Open chat with AI assistant',
          shortcut: 'Ctrl+C'
        },
        {
          id: 'analysis',
          icon: '🧠',
          label: 'Game Analysis',
          onClick: () => {}, // TODO: Implement analysis
          description: 'Analyze current game state',
          shortcut: 'Ctrl+A'
        },
        {
          id: 'strategy',
          icon: '🎯',
          label: 'Strategy Mode',
          onClick: () => {}, // TODO: Implement strategy
          description: 'Configure AI strategy',
          shortcut: 'Ctrl+T'
        },
        {
          id: 'learning',
          icon: '📚',
          label: 'Learning Hub',
          onClick: () => {}, // TODO: Implement learning
          description: 'View AI learning progress',
          shortcut: 'Ctrl+L'
        },
      ]
    },
    {
      title: 'Settings',
      items: [
        {
          id: 'theme',
          icon: getThemeIcon(),
          label: `Theme: ${theme.charAt(0).toUpperCase() + theme.slice(1)}`,
          onClick: toggleTheme,
          description: 'Toggle color theme',
          shortcut: 'Ctrl+T'
        },
        {
          id: 'settings',
          icon: '⚙️',
          label: 'Settings',
          onClick: handleOpenSettings,
          description: 'Open application settings',
          shortcut: 'Ctrl+,'
        },
      ]
    }
  ];

  const addLog = useCallback((message: string, type: AILog['type']) => {
    setAiLogs(prev => [...prev, { id: logIdCounter.current++, message, type }]);
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
  }, [addLog]);

  // SSE connection setup
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

        if (errorData.error && errorData.error.includes('not available') && errorData.available_providers) {
          const firstAvailableProvider = errorData.available_providers[0];
          addLog(`Provider '${appSettings.apiProvider}' not available. Switching to '${firstAvailableProvider}'`, 'info');

          const newSettings = {
            ...appSettings,
            apiProvider: firstAvailableProvider as AppSettings['apiProvider']
          };
          setAppSettings(newSettings);
          localStorage.setItem('appSettings', JSON.stringify(newSettings));

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
      return 'SELECT';
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

  // Handle ROM load from sidebar
  const handleRomLoad = (file: File) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.gb,.gbc';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        setRomName(file.name);
        addLog(`Loaded ROM: ${file.name}`, 'info');
        loadRom(file);
      }
    };
    input.click();
  };

  // Update sidebar item click handlers
  useEffect(() => {
    const sidebarItems = document.querySelectorAll('[data-sidebar-item]');
    sidebarItems.forEach(item => {
      item.addEventListener('click', (e) => {
        const itemId = (e.currentTarget as HTMLElement).dataset.sidebarItem;
        if (itemId === 'rom') {
          handleRomLoad(new File([], '', { type: 'application/octet-stream' }));
        }
      });
    });
  }, []);

  useEffect(() => {
    const handleControlPress = (event: CustomEvent) => {
      const action = event.detail;
      if (romName) {
        executeAction(action);
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

  // Update game stats
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

  // Mobile header component
  const MobileHeader = () => (
    <motion.header
      className="sticky top-0 z-40 bg-gray-900 border-b border-gray-800 px-4 py-3"
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setIsMobileMenuOpen(true)}
            className="p-2 rounded-lg hover:bg-gray-800 touch-target"
            aria-label="Open menu"
          >
            <Menu className="w-6 h-6" />
          </button>
          <h1 className="text-xl font-bold text-cyan-glow">AI Game Assistant</h1>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => setActivePanel(activePanel === 'game' ? 'chat' : 'game')}
            className="p-2 rounded-lg hover:bg-gray-800 touch-target"
            aria-label={activePanel === 'game' ? 'Switch to chat' : 'Switch to game'}
          >
            {activePanel === 'game' ? <MessageSquare className="w-5 h-5" /> : <Gamepad2 className="w-5 h-5" />}
          </button>

          <button
            onClick={toggleTheme}
            className="p-2 rounded-lg hover:bg-gray-800 touch-target"
            aria-label="Toggle theme"
          >
            {getThemeIcon()}
          </button>
        </div>
      </div>
    </motion.header>
  );

  // Mobile navigation panel
  const MobileNav = () => (
    <AnimatePresence>
      {isMobileMenuOpen && (
        <>
          <motion.div
            className="fixed inset-0 bg-black/50 z-50 md:hidden"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setIsMobileMenuOpen(false)}
          />
          <motion.div
            className="fixed left-0 top-0 h-full w-80 bg-gray-800 z-50 md:hidden overflow-y-auto"
            initial={{ x: '-100%' }}
            animate={{ x: 0 }}
            exit={{ x: '-100%' }}
            transition={{ type: 'spring', damping: 25 }}
          >
            <div className="p-4 border-b border-gray-700">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold">Menu</h2>
                <button
                  onClick={() => setIsMobileMenuOpen(false)}
                  className="p-2 rounded-lg hover:bg-gray-700"
                  aria-label="Close menu"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>

            <div className="p-4 space-y-6">
              {sidebarSections.map((section, index) => (
                <div key={index}>
                  <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
                    {section.title}
                  </h3>
                  <div className="space-y-2">
                    {section.items.map((item, itemIndex) => (
                      <button
                        key={itemIndex}
                        onClick={() => {
                          item.onClick();
                          setIsMobileMenuOpen(false);
                        }}
                        disabled={item.disabled}
                        className="w-full flex items-center space-x-3 p-3 rounded-lg hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed text-left transition-colors"
                      >
                        <span className="text-lg">{item.icon}</span>
                        <div className="flex-1">
                          <div className="font-medium">{item.label}</div>
                          <div className="text-xs text-gray-400">{item.description}</div>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );

  return (
    <ErrorBoundary>
      <div className="h-screen w-screen flex bg-gray-900 text-white overflow-hidden">
        {/* Desktop Sidebar */}
        <div className="hidden md:flex">
          <CollapsibleSidebar sections={sidebarSections} />
        </div>

        {/* Mobile Navigation */}
        <MobileNav />

        {/* Main Content */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Desktop Header */}
          <div className="hidden md:block">
            <Header
              emulatorMode={emulatorMode}
              onModeChange={setEmulatorMode}
              onOpenSettings={handleOpenSettings}
            />
          </div>

          {/* Mobile Header */}
          <div className="md:hidden">
            <MobileHeader />
          </div>

          {/* Main Content Area */}
          <div className="flex-1 flex overflow-hidden">
            {/* Desktop Layout */}
            <div className="hidden md:flex flex-1 overflow-hidden">
              {/* Game Area */}
              <div className="flex-1 flex flex-col p-4">
                <motion.div
                  className="h-full flex flex-col bg-gray-800 rounded-lg shadow-xl"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.3 }}
                >
                  {/* Game Stats Bar */}
                  <div className="p-4 border-b border-gray-700">
                    <GameStateDisplay
                      step={gameStats.step}
                      location={gameStats.location}
                      isPlaying={aiState === AIStateEnum.RUNNING}
                    />
                  </div>

                  {/* Emulator Screen */}
                  <div className="flex-1 p-4">
                    <EmulatorScreen
                      emulatorMode={emulatorMode}
                      romName={romName}
                      onRomLoad={() => {}}
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
                </motion.div>
              </div>

              {/* Right Panel - Enhanced Chat */}
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
                  messages={messages}
                  onSendMessage={handleSendMessage}
                  gameState={{
                    step: gameStats.step,
                    location: gameStats.location,
                    isPlaying: aiState === AIStateEnum.RUNNING
                  }}
                />
              </div>
            </div>

            {/* Mobile Layout */}
            <div className="md:hidden flex-1 overflow-hidden">
              <AnimatePresence mode="wait">
                <motion.div
                  key={activePanel}
                  className="h-full flex flex-col"
                  initial={{ x: activePanel === 'game' ? -100 : 100, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  exit={{ x: activePanel === 'game' ? -100 : 100, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                >
                  {activePanel === 'game' ? (
                    // Mobile Game Panel
                    <div className="flex-1 flex flex-col p-4">
                      <div className="h-full flex flex-col bg-gray-800 rounded-lg shadow-xl">
                        {/* Game Stats Bar */}
                        <div className="p-4 border-b border-gray-700">
                          <GameStateDisplay
                            step={gameStats.step}
                            location={gameStats.location}
                            isPlaying={aiState === AIStateEnum.RUNNING}
                          />
                        </div>

                        {/* Emulator Screen */}
                        <div className="flex-1 p-4">
                          <EmulatorScreen
                            emulatorMode={emulatorMode}
                            romName={romName}
                            onRomLoad={() => {}}
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
                  ) : (
                    // Mobile Chat Panel
                    <div className="flex-1 flex flex-col bg-gray-900">
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
                      <div className="flex-1 overflow-hidden">
                        <EnhancedChatInterface
                          messages={messages}
                          onSendMessage={handleSendMessage}
                          gameState={{
                            step: gameStats.step,
                            location: gameStats.location,
                            isPlaying: aiState === AIStateEnum.RUNNING
                          }}
                        />
                      </div>
                    </div>
                  )}
                </motion.div>
              </AnimatePresence>
            </div>
          </div>
        </div>

        {/* Settings Modal */}
        <SettingsModal
          isOpen={isSettingsOpen}
          onClose={handleCloseSettings}
          currentSettings={appSettings}
          onSave={handleSaveSettings}
        />

        {/* Touch Controls */}
        <TouchControls
          onAction={executeAction}
          disabled={!romName}
        />

        {/* Gesture Controls */}
        <GestureControls
          onSwipe={(direction) => {
            // Handle swipe gestures for quick actions
            switch (direction) {
              case 'left':
                setActivePanel('chat');
                break;
              case 'right':
                setActivePanel('game');
                break;
              case 'up':
                // Quick action
                executeAction('UP');
                break;
              case 'down':
                // Quick action
                executeAction('DOWN');
                break;
            }
          }}
          onDoubleTap={() => {
            // Double tap to toggle AI
            aiState === AIStateEnum.IDLE ? startAI() : stopAI();
          }}
          disabled={!romName}
        />
      </div>
    </ErrorBoundary>
  );
};

export default EnhancedApp;