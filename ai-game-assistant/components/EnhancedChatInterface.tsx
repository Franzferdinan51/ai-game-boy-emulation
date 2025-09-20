import React, { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import AIPersonalityAvatar, { AIPersonalityState } from './AIPersonalityAvatar';
import { GameStateDisplay } from './PokemonGameUI';
import { LazyImage } from './VirtualList';
import { Skeleton } from './LoadingStates';
import { Send, Paperclip, Smile, ThumbsUp, ThumbsDown } from 'lucide-react';

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
  const [isDragging, setIsDragging] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const handleSendMessage = useCallback(() => {
    if (message.trim()) {
      onSendMessage(message);
      setMessage('');
      setIsTyping(false);
    }
  }, [message, onSendMessage]);

  const handleKeyPress = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  }, [handleSendMessage]);

  const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // Handle file upload logic here
      console.log('File uploaded:', file.name);
      setFileError(null);
    }
  }, []);

  const formatTime = useCallback((date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }, []);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = inputRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 120)}px`;
    }
  }, [message]);

  // Focus management
  useEffect(() => {
    if (messages.length === 0) {
      inputRef.current?.focus();
    }
  }, [messages.length]);

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
    <div className={`flex flex-col h-full bg-gray-900 ${className}`} role="log" aria-label="Chat conversation">
      {/* Game State Header */}
      {gameState && (
        <div className="p-3 border-b border-gray-800">
          <GameStateDisplay {...gameState} />
        </div>
      )}

      {/* Messages Container */}
      <div
        className="flex-1 overflow-y-auto p-4 space-y-4"
        role="list"
        aria-label="Messages"
      >
        <AnimatePresence>
          {messages.length === 0 && (
            <motion.div
              className="text-center text-gray-500 py-8"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <motion.div
                className="text-lg mb-2"
                animate={{ rotate: [0, 10, -10, 0] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                🎮
              </motion.div>
              <p className="text-base">Start a conversation with the AI about your game!</p>
              <p className="text-sm mt-2 text-gray-400">Try asking about game strategy or what to do next</p>
            </motion.div>
          )}

          {messages.map((msg, index) => (
            <motion.div
              key={msg.id}
              className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              initial={{ opacity: 0, y: 20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -20, scale: 0.95 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
              role="listitem"
              aria-label={`${msg.role} message`}
            >
              {msg.role === 'ai' && (
                <AIPersonalityAvatar
                  state={msg.aiState || determineAIState(msg.content)}
                  size="md"
                  className="flex-shrink-0"
                  aria-label="AI assistant avatar"
                />
              )}

              <div className={`max-w-[70%] ${msg.role === 'user' ? 'order-1' : ''}`}>
                <motion.div
                  className={`rounded-2xl px-4 py-3 touch-target ${
                    msg.role === 'user'
                      ? 'bg-cyan-500 text-gray-900 rounded-br-none shadow-lg'
                      : 'bg-gray-800 text-gray-100 rounded-bl-none shadow-lg'
                  }`}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="text-sm whitespace-pre-wrap break-words">
                    {msg.content}
                  </div>
                </motion.div>

                {msg.timestamp && (
                  <div className={`text-xs text-gray-500 mt-1 ${
                    msg.role === 'user' ? 'text-right' : 'text-left'
                  }`}>
                    {formatTime(msg.timestamp)}
                  </div>
                )}

                {/* Feedback buttons for AI messages */}
                {msg.role === 'ai' && (
                  <div className="flex gap-2 mt-2">
                    <motion.button
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      className="p-1 rounded hover:bg-gray-700 transition-colors"
                      aria-label="Helpful response"
                      title="This was helpful"
                    >
                      <ThumbsUp className="w-4 h-4 text-gray-400" />
                    </motion.button>
                    <motion.button
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      className="p-1 rounded hover:bg-gray-700 transition-colors"
                      aria-label="Not helpful response"
                      title="This wasn't helpful"
                    >
                      <ThumbsDown className="w-4 h-4 text-gray-400" />
                    </motion.button>
                  </div>
                )}
              </div>

              {msg.role === 'user' && (
                <div className="w-12 h-12 bg-cyan-500 rounded-full flex items-center justify-center flex-shrink-0 touch-target">
                  <span className="text-white font-bold text-sm">YOU</span>
                </div>
              )}
            </motion.div>
          ))}

          {isTyping && (
            <motion.div
              className="flex gap-3"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <AIPersonalityAvatar state="thinking" size="md" />
              <div className="bg-gray-800 rounded-2xl rounded-bl-none px-4 py-3">
                <div className="flex gap-1">
                  <motion.div
                    className="w-2 h-2 bg-gray-500 rounded-full"
                    animate={{ y: [0, -8, 0] }}
                    transition={{ duration: 0.6, repeat: Infinity }}
                  />
                  <motion.div
                    className="w-2 h-2 bg-gray-500 rounded-full"
                    animate={{ y: [0, -8, 0] }}
                    transition={{ duration: 0.6, repeat: Infinity, delay: 0.2 }}
                  />
                  <motion.div
                    className="w-2 h-2 bg-gray-500 rounded-full"
                    animate={{ y: [0, -8, 0] }}
                    transition={{ duration: 0.6, repeat: Infinity, delay: 0.4 }}
                  />
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-4 border-t border-gray-800">
        {/* File upload error */}
        <AnimatePresence>
          {fileError && (
            <motion.div
              className="mb-3 p-3 bg-red-500/20 border border-red-500/50 rounded-lg text-red-200 text-sm"
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
            >
              {fileError}
            </motion.div>
          )}
        </AnimatePresence>

        <div className="flex gap-2 items-end">
          {/* Attachment button */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="p-2 rounded-lg hover:bg-gray-800 transition-colors touch-target"
            onClick={() => fileInputRef.current?.click()}
            aria-label="Attach file"
            title="Attach file"
          >
            <Paperclip className="w-5 h-5 text-gray-400" />
          </motion.button>

          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept="image/*,.txt,.json"
            onChange={handleFileUpload}
            aria-label="File upload"
          />

          {/* Text input */}
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="Ask the AI about your game..."
              rows={1}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 pr-12 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-cyan-500 resize-none mobile-text"
              aria-label="Message input"
              role="textbox"
              aria-multiline="true"
            />

            {/* Character counter */}
            {message.length > 0 && (
              <div className="absolute bottom-1 right-2 text-xs text-gray-500">
                {message.length}/1000
              </div>
            )}
          </div>

          {/* Send button */}
          <motion.button
            onClick={handleSendMessage}
            disabled={!message.trim()}
            className="bg-cyan-500 hover:bg-cyan-600 disabled:bg-gray-700 disabled:cursor-not-allowed text-white p-3 rounded-lg font-medium transition-colors touch-target flex items-center justify-center"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            aria-label="Send message"
            disabled={message.trim().length === 0}
          >
            <Send className="w-5 h-5" />
          </motion.button>
        </div>

        {/* Quick action buttons */}
        <div className="mt-3 flex flex-wrap gap-2">
          {[
            { label: "What's next?", message: "What should I do next?" },
            { label: "Analyze situation", message: "Analyze my current situation" },
            { label: "Battle help", message: "Help me with this battle" },
            { label: "Strategy tips", message: "Give me some strategy tips" }
          ].map((quickAction, index) => (
            <motion.button
              key={index}
              onClick={() => setMessage(quickAction.message)}
              className="text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 px-3 py-1.5 rounded-full transition-colors touch-target"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.1 }}
            >
              {quickAction.label}
            </motion.button>
          ))}
        </div>

        {/* Keyboard shortcuts hint */}
        <div className="mt-2 text-xs text-gray-500 text-center">
          Press Enter to send • Shift+Enter for new line
        </div>
      </div>
    </div>
  );
};

export default EnhancedChatInterface;