import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  ChatMessage,
  AISession,
  AIThinkingState,
  Settings
} from '../types';
import MessageBubble from './MessageBubble';
import {
  PaperAirplaneIcon,
  PaperClipIcon,
  MicrophoneIcon,
  PlusIcon,
  SparklesIcon
} from '@heroicons/react/24/outline';
import { Loader2 } from 'lucide-react';

interface ChatInterfaceProps {
  session: AISession | null;
  onSendMessage: (message: string, attachments?: File[]) => void;
  aiThinking: AIThinkingState;
  theme: 'light' | 'dark';
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  session,
  onSendMessage,
  aiThinking,
  theme
}) => {
  const [message, setMessage] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [attachments, setAttachments] = useState<File[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [session?.messages, aiThinking.isThinking, scrollToBottom]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  }, [message]);

  const handleSendMessage = useCallback(() => {
    if (message.trim() || attachments.length > 0) {
      onSendMessage(message, attachments);
      setMessage('');
      setAttachments([]);
    }
  }, [message, attachments, onSendMessage]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  }, [handleSendMessage]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    setAttachments(prev => [...prev, ...files]);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    const validFiles = files.filter(file => {
      const maxSize = 10 * 1024 * 1024; // 10MB
      return file.size <= maxSize;
    });

    setAttachments(prev => [...prev, ...validFiles]);
  }, []);

  const removeAttachment = useCallback((index: number) => {
    setAttachments(prev => prev.filter((_, i) => i !== index));
  }, []);

  const formatTime = useCallback((date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }, []);

  const bgColor = theme === 'dark' ? 'bg-gray-900' : 'bg-white';
  const borderColor = theme === 'dark' ? 'border-gray-800' : 'border-gray-200';
  const inputBg = theme === 'dark' ? 'bg-gray-800' : 'bg-gray-50';
  const textColor = theme === 'dark' ? 'text-gray-100' : 'text-gray-900';

  return (
    <div className={`h-full flex flex-col ${bgColor}`}>
      {/* Chat Header */}
      <div className={`border-b ${borderColor} p-4`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
              <SparklesIcon className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="font-semibold text-lg">AI Game Assistant</h2>
              <p className="text-sm opacity-70">
                {session?.model ? `Using ${session.model}` : 'Select a model to start'}
              </p>
            </div>
          </div>

          {session && (
            <div className="text-sm opacity-70">
              Session {session.id.slice(-8)}
            </div>
          )}
        </div>
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {session?.messages.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center max-w-md">
              <div className="mb-4">
                <div className="w-16 h-16 mx-auto bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                  <SparklesIcon className="w-8 h-8 text-white" />
                </div>
              </div>
              <h3 className="text-xl font-semibold mb-2">Welcome to AI Game Studio</h3>
              <p className="opacity-70 mb-4">
                Start a conversation with your AI game assistant. Load a ROM and let the AI help you play, analyze the game, or just chat about your gaming experience.
              </p>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className={`p-3 rounded-lg ${theme === 'dark' ? 'bg-gray-800' : 'bg-gray-100'}`}>
                  💬 Chat about game strategy
                </div>
                <div className={`p-3 rounded-lg ${theme === 'dark' ? 'bg-gray-800' : 'bg-gray-100'}`}>
                  🎮 Get AI assistance playing
                </div>
                <div className={`p-3 rounded-lg ${theme === 'dark' ? 'bg-gray-800' : 'bg-gray-100'}`}>
                  📊 Analyze game state
                </div>
                <div className={`p-3 rounded-lg ${theme === 'dark' ? 'bg-gray-800' : 'bg-gray-100'}`}>
                  🎯 Set game objectives
                </div>
              </div>
            </div>
          </div>
        ) : (
          session.messages.map((msg) => (
            <MessageBubble
              key={msg.id}
              message={msg}
              theme={theme}
            />
          ))
        )}

        {/* AI Thinking Indicator */}
        {aiThinking.isThinking && (
          <div className="flex items-start gap-3">
            <div className="flex items-center justify-center w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full">
              <Loader2 className="w-4 h-4 text-white animate-spin" />
            </div>
            <div className={`flex-1 ${theme === 'dark' ? 'bg-gray-800' : 'bg-gray-100'} rounded-2xl p-4`}>
              <div className="flex items-center gap-2 mb-2">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
              </div>
              <p className="text-sm opacity-80">{aiThinking.currentAction || 'Thinking...'}</p>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className={`border-t ${borderColor} p-4`}>
        {/* Attachments */}
        {attachments.length > 0 && (
          <div className="mb-3 flex flex-wrap gap-2">
            {attachments.map((file, index) => (
              <div
                key={index}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm ${
                  theme === 'dark' ? 'bg-gray-800' : 'bg-gray-100'
                }`}
              >
                <PaperClipIcon className="w-4 h-4" />
                <span className="max-w-xs truncate">{file.name}</span>
                <button
                  onClick={() => removeAttachment(index)}
                  className="text-red-500 hover:text-red-600"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Message Input */}
        <div
          className={`relative rounded-2xl border ${borderColor} ${inputBg} transition-all duration-200 ${
            isDragging ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : ''
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message AI Game Assistant..."
            className={`w-full px-4 py-3 pr-24 resize-none focus:outline-none rounded-2xl ${inputBg} ${textColor}`}
            rows={1}
            disabled={aiThinking.isThinking}
          />

          <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-2">
            <button
              onClick={() => fileInputRef.current?.click()}
              className="p-2 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
              disabled={aiThinking.isThinking}
            >
              <PaperClipIcon className="w-5 h-5" />
            </button>

            <button
              className="p-2 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
              disabled={aiThinking.isThinking}
            >
              <MicrophoneIcon className="w-5 h-5" />
            </button>

            <button
              onClick={handleSendMessage}
              disabled={(!message.trim() && attachments.length === 0) || aiThinking.isThinking}
              className={`p-2 rounded-lg transition-colors ${
                (message.trim() || attachments.length > 0) && !aiThinking.isThinking
                  ? 'bg-blue-500 hover:bg-blue-600 text-white'
                  : 'text-gray-400'
              }`}
            >
              <PaperAirplaneIcon className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          multiple
          onChange={handleFileSelect}
          className="hidden"
          accept="image/*,.txt,.json"
        />

        {/* Quick Actions */}
        <div className="mt-2 flex items-center justify-between text-xs opacity-70">
          <div className="flex items-center gap-4">
            <span>Press Enter to send</span>
            <span>Shift + Enter for new line</span>
          </div>
          <div className="flex items-center gap-2">
            <button className="hover:opacity-100 transition-opacity">
              <PlusIcon className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;