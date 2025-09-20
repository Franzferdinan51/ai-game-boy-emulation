export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp?: Date;
  attachments?: Array<{
    type: 'image' | 'file';
    url: string;
    name: string;
  }>;
}

export interface AIProvider {
  id: string;
  name: string;
  models: string[];
  available: boolean;
  endpoint?: string;
  requiresAuth: boolean;
}

export interface EmulatorState {
  type: 'gb' | 'gba';
  romName: string | null;
  isRunning: boolean;
  screenData: string | null;
  streamingStatus: 'disconnected' | 'connecting' | 'connected' | 'error';
  fps: number;
}

export interface AISession {
  id: string;
  provider: string;
  model: string;
  messages: ChatMessage[];
  gameState?: {
    step: number;
    score?: number;
    location?: string;
    objective?: string;
  };
  createdAt: Date;
  lastActive: Date;
}

export interface Settings {
  theme: 'light' | 'dark' | 'auto';
  aiProvider: string;
  aiModel: string;
  autoSave: boolean;
  notifications: boolean;
  streamingQuality: 'low' | 'medium' | 'high';
}

export type GameAction = 'UP' | 'DOWN' | 'LEFT' | 'RIGHT' | 'A' | 'B' | 'START' | 'SELECT' | 'NONE';

export interface AIThinkingState {
  isThinking: boolean;
  progress: number;
  currentAction?: string;
  confidence?: number;
}