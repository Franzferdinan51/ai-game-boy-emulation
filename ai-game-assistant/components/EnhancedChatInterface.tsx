import React, { useState, useRef, useEffect } from 'react';
import AIPersonalityAvatar, { AIPersonalityState } from './AIPersonalityAvatar';
import { GameStateDisplay } from './PokemonGameUI';

interface EnhancedChatInterfaceProps {
  messages: Array<{
    id: string;
    role: 'user' | 'ai';
    content: string;
    timestamp?: Date;
    aiState?: AIPersonalityState;
  }>;
  onSendMessage: (message: string) => void;
  gameState?: {
    step: number;
    location?: string;
    isPlaying?: boolean;
  };
  className?: string;
}

const EnhancedChatInterface: React.FC<EnhancedChatInterfaceProps> = ({
  messages,
  onSendMessage,
  gameState,
  className = ''
}) => {
  const [message, setMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

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

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

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

  return (
    <div className={`flex flex-col h-full bg-gray-900 ${className}`}>
      {/* Game State Header */}
      {gameState && (
        <div className="p-3 border-b border-gray-800">
          <GameStateDisplay {...gameState} />
        </div>
      )}

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 py-8">
            <div className="text-lg mb-2">🎮</div>
            <div>Start a conversation with the AI about your game!</div>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            {msg.role === 'ai' && (
              <AIPersonalityAvatar
                state={msg.aiState || determineAIState(msg.content)}
                size="md"
                className="flex-shrink-0"
              />
            )}

            <div className={`max-w-[70%] ${msg.role === 'user' ? 'order-1' : ''}`}>
              <div
                className={`rounded-2xl px-4 py-3 ${
                  msg.role === 'user'
                    ? 'bg-cyan-500 text-gray-900 rounded-br-none'
                    : 'bg-gray-800 text-gray-100 rounded-bl-none'
                }`}
              >
                <div className="text-sm whitespace-pre-wrap">{msg.content}</div>
              </div>

              {msg.timestamp && (
                <div className={`text-xs text-gray-500 mt-1 ${
                  msg.role === 'user' ? 'text-right' : 'text-left'
                }`}>
                  {formatTime(msg.timestamp)}
                </div>
              )}
            </div>

            {msg.role === 'user' && (
              <div className="w-12 h-12 bg-cyan-500 rounded-full flex items-center justify-center flex-shrink-0">
                <span className="text-white font-bold">YOU</span>
              </div>
            )}
          </div>
        ))}

        {isTyping && (
          <div className="flex gap-3">
            <AIPersonalityAvatar state="thinking" size="md" />
            <div className="bg-gray-800 rounded-2xl rounded-bl-none px-4 py-3">
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-4 border-t border-gray-800">
        <div className="flex gap-2">
          <textarea
            ref={inputRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask the AI about your game..."
            rows={2}
            className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-cyan-500 resize-none"
          />
          <button
            onClick={handleSendMessage}
            disabled={!message.trim()}
            className="bg-cyan-500 hover:bg-cyan-600 disabled:bg-gray-700 disabled:cursor-not-allowed text-white px-6 py-2 rounded-lg font-medium transition-colors"
          >
            Send
          </button>
        </div>

        <div className="mt-2 flex gap-2">
          <button
            onClick={() => setMessage("What should I do next?")}
            className="text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 px-3 py-1 rounded-full transition-colors"
          >
            What's next?
          </button>
          <button
            onClick={() => setMessage("Analyze my current situation")}
            className="text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 px-3 py-1 rounded-full transition-colors"
          >
            Analyze situation
          </button>
          <button
            onClick={() => setMessage("Help me with this battle")}
            className="text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 px-3 py-1 rounded-full transition-colors"
          >
            Battle help
          </button>
        </div>
      </div>
    </div>
  );
};

export default EnhancedChatInterface;