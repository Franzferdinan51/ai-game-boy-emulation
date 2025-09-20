import React from 'react';

export type AIPersonalityState = 'confident' | 'thinking' | 'reading' | 'mischievous' | 'playful' | 'neutral';

interface AIPersonalityAvatarProps {
  state: AIPersonalityState;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const AIPersonalityAvatar: React.FC<AIPersonalityAvatarProps> = ({
  state,
  size = 'md',
  className = ''
}) => {
  const sizeClasses = {
    sm: 'w-8 h-8',
    md: 'w-12 h-12',
    lg: 'w-16 h-16'
  };

  const getAvatarContent = () => {
    switch (state) {
      case 'confident':
        return (
          <div className="relative w-full h-full bg-gradient-to-br from-yellow-400 to-orange-500 rounded-full flex items-center justify-center">
            <div className="text-white text-xl font-bold">😎</div>
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full border-2 border-white"></div>
          </div>
        );

      case 'thinking':
        return (
          <div className="relative w-full h-full bg-gradient-to-br from-purple-400 to-blue-500 rounded-full flex items-center justify-center">
            <div className="text-white text-xl">🤔</div>
            <div className="absolute top-0 right-0 flex">
              <div className="w-2 h-2 bg-blue-300 rounded-full animate-pulse"></div>
              <div className="w-2 h-2 bg-blue-300 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
              <div className="w-2 h-2 bg-blue-300 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
            </div>
          </div>
        );

      case 'reading':
        return (
          <div className="relative w-full h-full bg-gradient-to-br from-green-400 to-teal-500 rounded-full flex items-center justify-center">
            <div className="text-white text-xl">📚</div>
            <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 translate-y-1">
              <div className="w-6 h-1 bg-white opacity-60 rounded"></div>
              <div className="w-4 h-1 bg-white opacity-40 rounded mt-1"></div>
            </div>
          </div>
        );

      case 'mischievous':
        return (
          <div className="relative w-full h-full bg-gradient-to-br from-red-400 to-pink-500 rounded-full flex items-center justify-center">
            <div className="text-white text-xl">😈</div>
            <div className="absolute top-1 left-1 w-2 h-2 bg-yellow-300 rounded-full animate-bounce"></div>
          </div>
        );

      case 'playful':
        return (
          <div className="relative w-full h-full bg-gradient-to-br from-pink-400 to-purple-500 rounded-full flex items-center justify-center animate-pulse">
            <div className="text-white text-xl">🎮</div>
            <div className="absolute -top-1 -right-1 w-4 h-4 bg-yellow-300 rounded-full animate-spin"></div>
          </div>
        );

      default: // neutral
        return (
          <div className="relative w-full h-full bg-gradient-to-br from-gray-400 to-gray-600 rounded-full flex items-center justify-center">
            <div className="text-white text-xl">🤖</div>
          </div>
        );
    }
  };

  const getStateLabel = () => {
    switch (state) {
      case 'confident': return 'Confident';
      case 'thinking': return 'Thinking';
      case 'reading': return 'Reading';
      case 'mischievous': return 'Mischievous';
      case 'playful': return 'Playful';
      default: return 'Neutral';
    }
  };

  return (
    <div className={`relative ${sizeClasses[size]} ${className}`}>
      {getAvatarContent()}
      <div className="absolute -bottom-6 left-1/2 transform -translate-x-1/2 text-xs text-gray-400 whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity">
        {getStateLabel()}
      </div>
    </div>
  );
};

export default AIPersonalityAvatar;