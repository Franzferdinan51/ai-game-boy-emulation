import React from 'react';

interface StreamingIndicatorProps {
  status: 'disconnected' | 'connecting' | 'connected' | 'error';
  fps?: number;
  theme: 'light' | 'dark';
}

const StreamingIndicator: React.FC<StreamingIndicatorProps> = ({
  status,
  fps,
  theme
}) => {
  if (status === 'disconnected') return null;

  const getStatusConfig = () => {
    switch (status) {
      case 'connecting':
        return {
          text: 'Connecting...',
          color: theme === 'dark' ? 'text-yellow-400' : 'text-yellow-600',
          bgColor: theme === 'dark' ? 'bg-yellow-500/20' : 'bg-yellow-100',
          pulse: true
        };
      case 'connected':
        return {
          text: fps ? `${fps} FPS` : 'LIVE',
          color: theme === 'dark' ? 'text-green-400' : 'text-green-600',
          bgColor: theme === 'dark' ? 'bg-green-500/20' : 'bg-green-100',
          pulse: false
        };
      case 'error':
        return {
          text: 'Stream Error',
          color: theme === 'dark' ? 'text-red-400' : 'text-red-600',
          bgColor: theme === 'dark' ? 'bg-red-500/20' : 'bg-red-100',
          pulse: false
        };
      default:
        return {
          text: 'Unknown',
          color: theme === 'dark' ? 'text-gray-400' : 'text-gray-600',
          bgColor: theme === 'dark' ? 'bg-gray-500/20' : 'bg-gray-100',
          pulse: false
        };
    }
  };

  const config = getStatusConfig();

  return (
    <div className={`px-3 py-1.5 rounded-full ${config.bgColor} flex items-center gap-2`}>
      <div className={`w-2 h-2 rounded-full ${config.color} ${
        config.pulse ? 'animate-pulse' : ''
      }`}></div>
      <span className={`text-xs font-medium ${config.color}`}>
        {config.text}
      </span>
    </div>
  );
};

export default StreamingIndicator;