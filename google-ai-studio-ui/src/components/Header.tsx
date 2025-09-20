import React, { useState } from 'react';
import { AISession, Settings } from '../types';
import {
  Cog6ToothIcon,
  PlusIcon,
  ClockIcon,
  SparklesIcon,
  DocumentTextIcon
} from '@heroicons/react/24/outline';

interface HeaderProps {
  title: string;
  subtitle: string;
  onOpenSettings: () => void;
  onCreateSession: () => void;
  currentSession: AISession | null;
  sessions: AISession[];
  onSessionSelect: (session: AISession) => void;
  theme: 'light' | 'dark';
}

const Header: React.FC<HeaderProps> = ({
  title,
  subtitle,
  onOpenSettings,
  onCreateSession,
  currentSession,
  sessions,
  onSessionSelect,
  theme
}) => {
  const [showSessionDropdown, setShowSessionDropdown] = useState(false);

  const bgColor = theme === 'dark' ? 'bg-gray-900' : 'bg-white';
  const borderColor = theme === 'dark' ? 'border-gray-800' : 'border-gray-200';
  const hoverBg = theme === 'dark' ? 'hover:bg-gray-800' : 'hover:bg-gray-50';
  const textColor = theme === 'dark' ? 'text-gray-300' : 'text-gray-700';
  const secondaryTextColor = theme === 'dark' ? 'text-gray-500' : 'text-gray-500';

  const formatTime = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return `${days}d ago`;
  };

  const getSessionPreview = (session: AISession) => {
    if (session.messages.length === 0) return 'New conversation';
    const lastMessage = session.messages[session.messages.length - 1];
    const preview = lastMessage.content.slice(0, 50);
    return lastMessage.content.length > 50 ? `${preview}...` : preview;
  };

  return (
    <header className={`h-16 border-b ${borderColor} ${bgColor}`}>
      <div className="h-full px-4 flex items-center justify-between">
        {/* Left Section - Title and Brand */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
              <SparklesIcon className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold">{title}</h1>
              <p className={`text-sm ${secondaryTextColor}`}>{subtitle}</p>
            </div>
          </div>
        </div>

        {/* Center Section - Session Management */}
        <div className="flex items-center gap-3">
          {/* Session Dropdown */}
          <div className="relative">
            <button
              onClick={() => setShowSessionDropdown(!showSessionDropdown)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg ${hoverBg} transition-colors min-w-[200px]`}
            >
              <DocumentTextIcon className="w-4 h-4" />
              <span className="text-sm truncate">
                {currentSession ? `Session ${currentSession.id.slice(-8)}` : 'No session'}
              </span>
              <svg
                className={`w-4 h-4 transition-transform ${showSessionDropdown ? 'rotate-180' : ''}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>

            {/* Session Dropdown Menu */}
            {showSessionDropdown && (
              <div className={`absolute top-full left-0 mt-2 w-80 rounded-lg shadow-lg border ${borderColor} ${bgColor} z-50`}>
                <div className="p-3 border-b border-gray-700">
                  <div className="flex items-center justify-between">
                    <h3 className="font-medium">Recent Sessions</h3>
                    <button
                      onClick={onCreateSession}
                      className="flex items-center gap-1 px-2 py-1 text-xs bg-blue-500 hover:bg-blue-600 text-white rounded transition-colors"
                    >
                      <PlusIcon className="w-3 h-3" />
                      New
                    </button>
                  </div>
                </div>

                <div className="max-h-64 overflow-y-auto">
                  {sessions.length === 0 ? (
                    <div className="p-4 text-center text-sm opacity-70">
                      No sessions yet. Create your first session!
                    </div>
                  ) : (
                    sessions.map((session) => (
                      <button
                        key={session.id}
                        onClick={() => {
                          onSessionSelect(session);
                          setShowSessionDropdown(false);
                        }}
                        className={`w-full p-3 text-left hover:${hoverBg} transition-colors ${
                          currentSession?.id === session.id
                            ? theme === 'dark' ? 'bg-blue-500/20' : 'bg-blue-100'
                            : ''
                        }`}
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="text-sm font-medium truncate">
                                Session {session.id.slice(-8)}
                              </span>
                              {session.model && (
                                <span className={`text-xs px-1.5 py-0.5 rounded ${
                                  theme === 'dark' ? 'bg-gray-700' : 'bg-gray-200'
                                }`}>
                                  {session.model}
                                </span>
                              )}
                            </div>
                            <p className={`text-xs ${secondaryTextColor} truncate`}>
                              {getSessionPreview(session)}
                            </p>
                          </div>
                          <div className="flex flex-col items-end text-xs opacity-70">
                            <div className="flex items-center gap-1">
                              <ClockIcon className="w-3 h-3" />
                              <span>{formatTime(session.lastActive)}</span>
                            </div>
                            <span>{session.messages.length} messages</span>
                          </div>
                        </div>
                      </button>
                    ))
                  )}
                </div>
              </div>
            )}
          </div>

          {/* New Session Button */}
          <button
            onClick={onCreateSession}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg ${hoverBg} transition-colors`}
            title="New session"
          >
            <PlusIcon className="w-4 h-4" />
            <span className="text-sm">New</span>
          </button>
        </div>

        {/* Right Section - Actions */}
        <div className="flex items-center gap-2">
          {/* Status Indicator */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-green-500/10 text-green-400">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-xs font-medium">Server Online</span>
          </div>

          {/* Settings Button */}
          <button
            onClick={onOpenSettings}
            className={`flex items-center gap-2 p-2 rounded-lg ${hoverBg} transition-colors`}
            title="Settings"
          >
            <Cog6ToothIcon className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Close dropdown when clicking outside */}
      {showSessionDropdown && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setShowSessionDropdown(false)}
        />
      )}
    </header>
  );
};

export default Header;