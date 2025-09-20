import React, { useState } from 'react';
import { GameAction } from '../types';
import {
  ChevronUpIcon,
  ChevronDownIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  PlusIcon,
  MinusIcon,
  PlayIcon,
  StopIcon
} from '@heroicons/react/24/outline';

interface GameControlsProps {
  onAction: (action: GameAction) => void;
  theme: 'light' | 'dark';
  compact?: boolean;
}

const GameControls: React.FC<GameControlsProps> = ({
  onAction,
  theme,
  compact = false
}) => {
  const [activeButton, setActiveButton] = useState<GameAction | null>(null);

  const handleAction = (action: GameAction) => {
    setActiveButton(action);
    onAction(action);
    setTimeout(() => setActiveButton(null), 150);
  };

  const buttonClasses = (action: GameAction) => {
    const base = `flex items-center justify-center rounded-lg transition-all duration-150 transform ${
      activeButton === action ? 'scale-95' : 'hover:scale-105'
    }`;

    const size = compact ? 'w-10 h-10' : 'w-12 h-12';

    const colors = theme === 'dark'
      ? activeButton === action
        ? 'bg-blue-600 text-white'
        : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
      : activeButton === action
        ? 'bg-blue-500 text-white'
        : 'bg-gray-200 text-gray-700 hover:bg-gray-300';

    return `${base} ${size} ${colors}`;
  };

  const getActionIcon = (action: GameAction) => {
    switch (action) {
      case 'UP':
        return <ChevronUpIcon className="w-6 h-6" />;
      case 'DOWN':
        return <ChevronDownIcon className="w-6 h-6" />;
      case 'LEFT':
        return <ChevronLeftIcon className="w-6 h-6" />;
      case 'RIGHT':
        return <ChevronRightIcon className="w-6 h-6" />;
      case 'A':
        return <PlusIcon className="w-5 h-5" />;
      case 'B':
        return <MinusIcon className="w-5 h-5" />;
      case 'START':
        return <PlayIcon className="w-5 h-5" />;
      case 'SELECT':
        return <StopIcon className="w-5 h-5" />;
      default:
        return null;
    }
  };

  const getActionLabel = (action: GameAction) => {
    switch (action) {
      case 'UP':
        return 'Up';
      case 'DOWN':
        return 'Down';
      case 'LEFT':
        return 'Left';
      case 'RIGHT':
        return 'Right';
      case 'A':
        return 'A';
      case 'B':
        return 'B';
      case 'START':
        return 'Start';
      case 'SELECT':
        return 'Select';
      default:
        return '';
    }
  };

  if (compact) {
    return (
      <div className={`p-3 rounded-lg ${theme === 'dark' ? 'bg-black/70' : 'bg-white/90'}`}>
        {/* D-Pad */}
        <div className="grid grid-cols-3 gap-1 mb-2">
          <div></div>
          <button
            onClick={() => handleAction('UP')}
            className={buttonClasses('UP')}
            title="Up"
          >
            {getActionIcon('UP')}
          </button>
          <div></div>

          <button
            onClick={() => handleAction('LEFT')}
            className={buttonClasses('LEFT')}
            title="Left"
          >
            {getActionIcon('LEFT')}
          </button>
          <div className={`w-10 h-10 rounded-lg ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-300'}`}></div>
          <button
            onClick={() => handleAction('RIGHT')}
            className={buttonClasses('RIGHT')}
            title="Right"
          >
            {getActionIcon('RIGHT')}
          </button>

          <div></div>
          <button
            onClick={() => handleAction('DOWN')}
            className={buttonClasses('DOWN')}
            title="Down"
          >
            {getActionIcon('DOWN')}
          </button>
          <div></div>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-1">
          <button
            onClick={() => handleAction('A')}
            className={`${buttonClasses('A')} bg-red-600 hover:bg-red-700 text-white`}
            title="A Button"
          >
            {getActionIcon('A')}
          </button>
          <button
            onClick={() => handleAction('B')}
            className={`${buttonClasses('B')} bg-yellow-600 hover:bg-yellow-700 text-white`}
            title="B Button"
          >
            {getActionIcon('B')}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`p-6 rounded-lg ${theme === 'dark' ? 'bg-gray-800' : 'bg-gray-100'}`}>
      <div className="grid grid-cols-2 gap-8">
        {/* D-Pad Section */}
        <div className="space-y-2">
          <h3 className={`text-sm font-medium ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
            D-Pad
          </h3>
          <div className="grid grid-cols-3 gap-2">
            <div></div>
            <button
              onClick={() => handleAction('UP')}
              className={buttonClasses('UP')}
              title="Up"
            >
              {getActionIcon('UP')}
            </button>
            <div></div>

            <button
              onClick={() => handleAction('LEFT')}
              className={buttonClasses('LEFT')}
              title="Left"
            >
              {getActionIcon('LEFT')}
            </button>
            <div className={`w-12 h-12 rounded-lg ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-300'}`}></div>
            <button
              onClick={() => handleAction('RIGHT')}
              className={buttonClasses('RIGHT')}
              title="Right"
            >
              {getActionIcon('RIGHT')}
            </button>

            <div></div>
            <button
              onClick={() => handleAction('DOWN')}
              className={buttonClasses('DOWN')}
              title="Down"
            >
              {getActionIcon('DOWN')}
            </button>
            <div></div>
          </div>
        </div>

        {/* Action Buttons Section */}
        <div className="space-y-2">
          <h3 className={`text-sm font-medium ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
            Action Buttons
          </h3>
          <div className="space-y-2">
            <div className="flex gap-2">
              <button
                onClick={() => handleAction('A')}
                className={`${buttonClasses('A')} bg-red-600 hover:bg-red-700 text-white`}
                title="A Button"
              >
                {getActionIcon('A')}
              </button>
              <button
                onClick={() => handleAction('B')}
                className={`${buttonClasses('B')} bg-yellow-600 hover:bg-yellow-700 text-white`}
                title="B Button"
              >
                {getActionIcon('B')}
              </button>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => handleAction('START')}
                className={buttonClasses('START')}
                title="Start"
              >
                {getActionIcon('START')}
              </button>
              <button
                onClick={() => handleAction('SELECT')}
                className={buttonClasses('SELECT')}
                title="Select"
              >
                {getActionIcon('SELECT')}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Keyboard Shortcuts Info */}
      <div className={`mt-4 pt-4 border-t ${theme === 'dark' ? 'border-gray-700' : 'border-gray-300'}`}>
        <p className={`text-xs ${theme === 'dark' ? 'text-gray-500' : 'text-gray-600'}`}>
          You can also use keyboard controls: Arrow keys for D-Pad, Z/X for A/B buttons
        </p>
      </div>
    </div>
  );
};

export default GameControls;