
export interface APIConfig {
  name: string;
  key: string;
  baseUrl?: string;
}

export enum EmulatorMode {
  GB = 'gb',
  GBA = 'gba',
}

export enum AIState {
  IDLE = 'idle',
  RUNNING = 'running',
  THINKING = 'thinking',
  ERROR = 'error',
}

export type GameAction = 'UP' | 'DOWN' | 'LEFT' | 'RIGHT' | 'A' | 'B' | 'START' | 'SELECT';

export interface AILog {
  id: number;
  message: string;
  type: 'info' | 'action' | 'thought' | 'error';
}

export interface ChatMessage {
  id: number;
  sender: 'user' | 'ai';
  text: string;
}

export interface AppSettings {
  aiActionInterval: number; // Time in ms between AI actions
  apiProvider: 'gemini' | 'openrouter' | 'openai-compatible' | 'nvidia';
  apiEndpoint?: string; // Optional: for local/custom models
  apiKey?: string; // Optional: for services that require it
  model?: string; // Optional: specify a model
}
