import React, { useRef, useEffect } from 'react';
import { AIState } from '../types';
import type { AILog, ChatMessage } from '../types';
import PlayIcon from './icons/PlayIcon';
import StopIcon from './icons/StopIcon';
import SendIcon from './icons/SendIcon';

interface AIPanelProps {
  aiState: AIState;
  aiLogs: AILog[];
  aiGoal: string;
  chatHistory: ChatMessage[];
  chatInput: string;
  isChatting: boolean;
  onGoalChange: (goal: string) => void;
  onStart: () => void;
  onStop: () => void;
  onChatInputChange: (value: string) => void;
  onSendMessage: () => void;
}

const AIPanel: React.FC<AIPanelProps> = ({ 
  aiState, 
  aiLogs, 
  aiGoal, 
  chatHistory,
  chatInput,
  isChatting,
  onGoalChange, 
  onStart, 
  onStop,
  onChatInputChange,
  onSendMessage,
}) => {
  const logContainerRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [aiLogs]);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatHistory]);

  const isRunning = aiState === AIState.RUNNING || aiState === AIState.THINKING;

  const getLogColor = (type: AILog['type']) => {
    switch (type) {
      case 'action': return 'text-green-glow';
      case 'thought': return 'text-cyan-glow';
      case 'error': return 'text-red-500';
      default: return 'text-neutral-400';
    }
  };

  const handleChatKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (chatInput.trim() && !isChatting) {
        onSendMessage();
      }
    }
  };

  return (
    <div className="w-full md:w-96 bg-neutral-900 border-l border-neutral-800 flex flex-col">
      <div className="p-3 border-b border-neutral-800">
        <h2 className="font-display font-bold text-lg">AI Controls</h2>
      </div>

      <div className="flex-grow flex flex-col p-3 overflow-hidden">
        {/* AI Objective */}
        <label htmlFor="ai-goal" className="text-sm font-semibold text-neutral-300 mb-2">AI Objective</label>
        <textarea
          id="ai-goal"
          rows={3}
          value={aiGoal}
          onChange={(e) => onGoalChange(e.target.value)}
          placeholder="e.g., 'Defeat the first gym leader'"
          className="w-full p-2 bg-neutral-800 border border-neutral-700 rounded-md text-sm placeholder:text-neutral-600 focus:ring-1 focus:ring-cyan-glow focus:border-cyan-glow outline-none"
          disabled={isRunning}
        />
        
        {/* Container for Log and Chat */}
        <div className="mt-4 flex-grow flex flex-col space-y-4 overflow-hidden">
          {/* AI Log */}
          <div className="flex flex-col bg-black rounded-md overflow-hidden border border-neutral-800 h-1/2">
            <div className="px-3 py-2 bg-neutral-800 text-xs font-mono text-neutral-400">AI LOG</div>
            <div ref={logContainerRef} className="flex-grow p-3 space-y-2 overflow-y-auto font-mono text-xs">
              {aiLogs.length === 0 && <span className="text-neutral-600">Awaiting commands...</span>}
              {aiLogs.map((log) => (
                <div key={log.id}>
                  <span className={getLogColor(log.type)}>{log.message}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Chat Area */}
          <div className="flex flex-col bg-black rounded-md overflow-hidden border border-neutral-800 h-1/2">
            <div className="px-3 py-2 bg-neutral-800 text-xs font-mono text-neutral-400">CHAT WITH AI</div>
            <div ref={chatContainerRef} className="flex-grow p-3 space-y-3 overflow-y-auto text-sm">
              {chatHistory.length === 0 && <span className="text-neutral-600 text-xs">Ask the AI a question...</span>}
              {chatHistory.map((msg) => (
                <div key={msg.id} className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}>
                  <div className={`px-3 py-2 rounded-lg max-w-xs ${msg.sender === 'user' ? 'bg-cyan-800' : 'bg-neutral-700'}`}>
                    {msg.text}
                  </div>
                </div>
              ))}
              {isChatting && (
                  <div className="flex items-start">
                      <div className="px-3 py-2 rounded-lg bg-neutral-700">
                          <span className="animate-pulse">...</span>
                      </div>
                  </div>
              )}
            </div>
          </div>
        </div>
      </div>
      
      {/* Buttons and Chat Input */}
      <div className="p-3 border-t border-neutral-800 space-y-3">
        <button
          onClick={isRunning ? onStop : onStart}
          disabled={!aiGoal}
          className={`w-full flex items-center justify-center p-3 rounded-md font-bold text-lg transition-all duration-300
            ${isRunning ? 'bg-red-600 hover:bg-red-700 text-white' : 'bg-green-glow hover:opacity-80 text-neutral-950'}
            disabled:bg-neutral-700 disabled:text-neutral-500 disabled:cursor-not-allowed`}
        >
          {isRunning ? <StopIcon className="w-6 h-6 mr-2" /> : <PlayIcon className="w-6 h-6 mr-2" />}
          {isRunning ? (aiState === AIState.THINKING ? 'Thinking...' : 'Stop AI') : 'Start AI'}
        </button>
        <div className="flex items-center space-x-2">
            <input 
              type="text"
              value={chatInput}
              onChange={(e) => onChatInputChange(e.target.value)}
              onKeyDown={handleChatKeyDown}
              placeholder="Ask a question..."
              disabled={isChatting}
              className="flex-grow p-2 bg-neutral-800 border border-neutral-700 rounded-md text-sm placeholder:text-neutral-600 focus:ring-1 focus:ring-cyan-glow focus:border-cyan-glow outline-none disabled:bg-neutral-800/50"
            />
            <button 
                onClick={onSendMessage} 
                disabled={!chatInput.trim() || isChatting}
                className="p-2 bg-cyan-glow rounded-md text-neutral-950 disabled:bg-neutral-700 disabled:text-neutral-500"
                aria-label="Send message"
            >
                <SendIcon className="w-6 h-6" />
            </button>
        </div>
      </div>
    </div>
  );
};

export default AIPanel;