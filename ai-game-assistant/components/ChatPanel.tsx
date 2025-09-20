import React, { useState, useRef, useEffect } from 'react';
import { AIState } from '../types';
import type { AILog } from '../types';
import PlayIcon from './icons/PlayIcon';
import StopIcon from './icons/StopIcon';

interface ChatPanelProps {
  aiState: AIState;
  aiLogs: AILog[];
  aiGoal: string;
  onGoalChange: (goal: string) => void;
  onStart: () => void;
  onStop: () => void;
  onSendMessage: (message: string) => void;
  conversation: Array<{ role: 'user' | 'ai'; content: string }>;
}

const ChatPanel: React.FC<ChatPanelProps> = ({ 
  aiState, 
  aiLogs, 
  aiGoal, 
  onGoalChange, 
  onStart, 
  onStop,
  onSendMessage,
  conversation
}) => {
  const [message, setMessage] = useState('');
  const [activeTab, setActiveTab] = useState<'controls' | 'chat' | 'analysis'>('controls');
  const logContainerRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [aiLogs]);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [conversation]);

  const isRunning = aiState === AIState.RUNNING || aiState === AIState.THINKING;

  const getLogColor = (type: AILog['type']) => {
    switch (type) {
      case 'action': return 'text-green-glow';
      case 'thought': return 'text-cyan-glow';
      case 'error': return 'text-red-500';
      default: return 'text-neutral-400';
    }
  };

  const handleSendMessage = () => {
    if (message.trim()) {
      onSendMessage(message);
      setMessage('');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Suggestion system for user guidance
  const getSuggestions = () => {
    const suggestions = [];
    
    if (!aiGoal) {
      suggestions.push({
        title: "Set an objective",
        description: "Tell the AI what you want to accomplish in the game",
        action: () => {}
      });
    }
    
    if (aiState === AIState.IDLE) {
      suggestions.push({
        title: "Start the AI",
        description: "Begin AI control with your objective",
        action: onStart
      });
    }
    
    suggestions.push({
      title: "Ask about the game",
      description: "Ask the AI for help with the current situation",
      action: () => setActiveTab('chat')
    });
    
    return suggestions;
  };

  const suggestions = getSuggestions();

  return (
    <div className="w-full md:w-96 bg-neutral-900 border-l border-neutral-800 flex flex-col">
      <div className="p-3 border-b border-neutral-800">
        <h2 className="font-display font-bold text-lg">AI Assistant</h2>
        <div className="flex mt-2">
          <button
            className={`flex-1 py-2 text-sm font-medium ${activeTab === 'controls' ? 'bg-cyan-glow text-neutral-950' : 'bg-neutral-800 text-neutral-300'}`}
            onClick={() => setActiveTab('controls')}
          >
            Controls
          </button>
          <button
            className={`flex-1 py-2 text-sm font-medium ${activeTab === 'chat' ? 'bg-cyan-glow text-neutral-950' : 'bg-neutral-800 text-neutral-300'}`}
            onClick={() => setActiveTab('chat')}
          >
            Chat
          </button>
          <button
            className={`flex-1 py-2 text-sm font-medium ${activeTab === 'analysis' ? 'bg-cyan-glow text-neutral-950' : 'bg-neutral-800 text-neutral-300'}`}
            onClick={() => setActiveTab('analysis')}
          >
            Analysis
          </button>
        </div>
      </div>

      {activeTab === 'controls' ? (
        <div className="flex-grow flex flex-col p-3 overflow-hidden">
          <label htmlFor="ai-goal" className="text-sm font-semibold text-neutral-300 mb-2">AI Objective</label>
          <textarea
            id="ai-goal"
            rows={3}
            value={aiGoal}
            onChange={(e) => onGoalChange(e.target.value)}
            placeholder="e.g., 'Defeat the first gym leader' or 'Find the hidden sword'"
            className="w-full p-2 bg-neutral-800 border border-neutral-700 rounded-md text-sm placeholder:text-neutral-600 focus:ring-1 focus:ring-cyan-glow focus:border-cyan-glow outline-none"
            disabled={isRunning}
          />
          
          <div className="mt-4 flex-grow flex flex-col bg-black rounded-md overflow-hidden border border-neutral-800">
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
          
          {/* Suggestions */}
          {suggestions.length > 0 && (
            <div className="mt-4">
              <div className="text-xs font-semibold text-neutral-400 mb-2">Suggestions</div>
              <div className="space-y-2">
                {suggestions.map((suggestion, index) => (
                  <button
                    key={index}
                    onClick={suggestion.action}
                    className="w-full text-left p-2 bg-neutral-800 rounded hover:bg-neutral-700 transition-colors"
                  >
                    <div className="font-medium text-sm">{suggestion.title}</div>
                    <div className="text-xs text-neutral-400">{suggestion.description}</div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : activeTab === 'chat' ? (
        <div className="flex-grow flex flex-col p-3 overflow-hidden">
          <div className="flex-grow flex flex-col bg-black rounded-md overflow-hidden border border-neutral-800">
            <div className="px-3 py-2 bg-neutral-800 text-xs font-mono text-neutral-400">CONVERSATION</div>
            <div ref={chatContainerRef} className="flex-grow p-3 space-y-3 overflow-y-auto">
              {conversation.length === 0 && (
                <div className="text-neutral-600 text-sm">
                  Start a conversation with the AI about the game...
                </div>
              )}
              {conversation.map((msg, index) => (
                <div key={index} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[80%] rounded-lg p-3 text-sm ${
                    msg.role === 'user' 
                      ? 'bg-cyan-glow text-neutral-950' 
                      : 'bg-neutral-800 text-neutral-200'
                  }`}>
                    {msg.content}
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          <div className="mt-3 flex">
            <textarea
              ref={inputRef}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask the AI about the game..."
              rows={2}
              className="flex-grow p-2 bg-neutral-800 border border-neutral-700 rounded-l-md text-sm placeholder:text-neutral-600 focus:ring-1 focus:ring-cyan-glow focus:border-cyan-glow outline-none resize-none"
            />
            <button
              onClick={handleSendMessage}
              disabled={!message.trim()}
              className="bg-cyan-glow text-neutral-950 px-4 rounded-r-md font-medium disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Send
            </button>
          </div>
        </div>
      ) : (
        <div className="flex-grow flex flex-col p-3 overflow-hidden">
          <div className="flex-grow flex flex-col bg-black rounded-md overflow-hidden border border-neutral-800">
            <div className="px-3 py-2 bg-neutral-800 text-xs font-mono text-neutral-400">GAME ANALYSIS</div>
            <div className="flex-grow p-3 overflow-y-auto text-sm">
              <div className="text-neutral-600">
                Game analysis will appear here when the AI is running...
              </div>
              <div className="mt-4 space-y-3">
                <div>
                  <div className="font-semibold text-neutral-300">Current Status</div>
                  <div className="text-neutral-400">
                    {aiState === AIState.RUNNING ? "AI is actively playing" : 
                     aiState === AIState.THINKING ? "AI is thinking..." : 
                     "AI is idle"}
                  </div>
                </div>
                <div>
                  <div className="font-semibold text-neutral-300">Objective</div>
                  <div className="text-neutral-400">{aiGoal || "No objective set"}</div>
                </div>
                <div>
                  <div className="font-semibold text-neutral-300">Recent Actions</div>
                  <div className="text-neutral-400">
                    {/* This would be populated with actual action history */}
                    No actions recorded yet
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      
      <div className="p-3 border-t border-neutral-800">
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
      </div>
    </div>
  );
};

export default ChatPanel;