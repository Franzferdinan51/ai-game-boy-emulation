import React, { useState } from 'react';
import { ChatMessage } from '../types';
import {
  UserCircleIcon,
  SparklesIcon,
  CheckCircleIcon,
  DocumentTextIcon,
  PhotoIcon
} from '@heroicons/react/24/outline';
import { ThumbsUp, ThumbsDown, Copy, MoreVertical } from 'lucide-react';

interface MessageBubbleProps {
  message: ChatMessage;
  theme: 'light' | 'dark';
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message, theme }) => {
  const [showActions, setShowActions] = useState(false);
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState<'up' | 'down' | null>(null);

  const isUser = message.role === 'user';
  const isAssistant = message.role === 'assistant';

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error('Failed to copy:', error);
    }
  };

  const handleFeedback = (type: 'up' | 'down') => {
    setFeedback(type);
    // Here you would typically send feedback to your backend
  };

  const bgColor = isUser
    ? theme === 'dark' ? 'bg-blue-600' : 'bg-blue-500'
    : theme === 'dark' ? 'bg-gray-800' : 'bg-gray-100';

  const textColor = isUser
    ? 'text-white'
    : theme === 'dark' ? 'text-gray-100' : 'text-gray-900';

  const secondaryTextColor = isUser
    ? 'text-blue-100'
    : theme === 'dark' ? 'text-gray-400' : 'text-gray-600';

  return (
    <div
      className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
    >
      {/* Avatar */}
      <div className={`flex-shrink-0 ${isUser ? 'ml-3' : 'mr-3'}`}>
        <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
          isUser
            ? 'bg-blue-600 text-white'
            : 'bg-gradient-to-br from-blue-500 to-purple-600 text-white'
        }`}>
          {isUser ? (
            <UserCircleIcon className="w-5 h-5" />
          ) : (
            <SparklesIcon className="w-5 h-5" />
          )}
        </div>
      </div>

      {/* Message Content */}
      <div className={`flex-1 max-w-3xl ${isUser ? 'text-right' : 'text-left'}`}>
        {/* Message Header */}
        <div className={`flex items-center gap-2 mb-1 ${isUser ? 'justify-end' : 'justify-start'}`}>
          <span className={`text-sm font-medium ${isUser ? 'text-blue-100' : 'text-purple-400'}`}>
            {isUser ? 'You' : 'AI Assistant'}
          </span>
          <span className={`text-xs ${secondaryTextColor}`}>
            {message.timestamp && formatTime(message.timestamp)}
          </span>
          {message.attachments && message.attachments.length > 0 && (
            <span className={`text-xs ${secondaryTextColor}`}>
              {message.attachments.length} attachment{message.attachments.length > 1 ? 's' : ''}
            </span>
          )}
        </div>

        {/* Message Body */}
        <div className={`inline-block p-4 rounded-2xl ${bgColor} ${
          isUser ? 'rounded-br-sm' : 'rounded-bl-sm'
        }`}>
          {/* Text Content */}
          <div className={`${textColor} whitespace-pre-wrap break-words`}>
            {message.content.split('\n').map((paragraph, index) => (
              <p key={index} className={index > 0 ? 'mt-2' : ''}>
                {paragraph}
              </p>
            ))}
          </div>

          {/* Attachments */}
          {message.attachments && message.attachments.length > 0 && (
            <div className="mt-3 space-y-2">
              {message.attachments.map((attachment, index) => (
                <div
                  key={index}
                  className={`flex items-center gap-2 p-2 rounded-lg ${
                    theme === 'dark' ? 'bg-gray-700' : 'bg-white'
                  }`}
                >
                  {attachment.type === 'image' ? (
                    <div className="flex items-center gap-2">
                      <PhotoIcon className="w-4 h-4 text-blue-400" />
                      <span className="text-sm">{attachment.name}</span>
                      {attachment.url && (
                        <img
                          src={attachment.url}
                          alt={attachment.name}
                          className="w-16 h-16 object-cover rounded cursor-pointer hover:opacity-80 transition-opacity"
                          onClick={() => window.open(attachment.url, '_blank')}
                        />
                      )}
                    </div>
                  ) : (
                    <div className="flex items-center gap-2">
                      <DocumentTextIcon className="w-4 h-4 text-gray-400" />
                      <span className="text-sm">{attachment.name}</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Message Actions */}
        {isAssistant && (showActions || feedback || copied) && (
          <div className={`flex items-center gap-2 mt-2 ${isUser ? 'justify-end' : 'justify-start'}`}>
            <div className={`flex items-center gap-1 p-1 rounded-lg ${
              theme === 'dark' ? 'bg-gray-800' : 'bg-gray-100'
            }`}>
              <button
                onClick={copyToClipboard}
                className={`p-1.5 rounded transition-colors ${
                  theme === 'dark' ? 'hover:bg-gray-700' : 'hover:bg-gray-200'
                }`}
                title="Copy"
              >
                {copied ? (
                  <CheckCircleIcon className="w-4 h-4 text-green-500" />
                ) : (
                  <Copy className="w-4 h-4 text-gray-500" />
                )}
              </button>

              <button
                onClick={() => handleFeedback('up')}
                className={`p-1.5 rounded transition-colors ${
                  feedback === 'up'
                    ? 'text-green-500 bg-green-500/10'
                    : theme === 'dark'
                      ? 'hover:bg-gray-700 text-gray-500'
                      : 'hover:bg-gray-200 text-gray-500'
                }`}
                title="Good response"
              >
                <ThumbsUp className="w-4 h-4" />
              </button>

              <button
                onClick={() => handleFeedback('down')}
                className={`p-1.5 rounded transition-colors ${
                  feedback === 'down'
                    ? 'text-red-500 bg-red-500/10'
                    : theme === 'dark'
                      ? 'hover:bg-gray-700 text-gray-500'
                      : 'hover:bg-gray-200 text-gray-500'
                }`}
                title="Bad response"
              >
                <ThumbsDown className="w-4 h-4" />
              </button>

              <button
                className={`p-1.5 rounded transition-colors ${
                  theme === 'dark' ? 'hover:bg-gray-700' : 'hover:bg-gray-200'
                } text-gray-500`}
                title="More options"
              >
                <MoreVertical className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MessageBubble;