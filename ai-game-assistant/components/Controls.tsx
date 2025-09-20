
import React from 'react';
import type { GameAction } from '../types';

interface ControlsProps {
  lastAction: GameAction | null;
}

const DPadButton: React.FC<{ label: string; gridArea: string; isActive: boolean; onClick: () => void }> = ({ label, gridArea, isActive, onClick }) => (
  <button
    style={{ gridArea }}
    onClick={onClick}
    className={`w-10 h-10 md:w-12 md:h-12 flex items-center justify-center bg-neutral-800 transition-all duration-150 rounded-lg ${
      isActive ? 'bg-green-glow scale-110 shadow-lg shadow-green-glow/30' : 'hover:bg-neutral-700 active:scale-95'
    }`}
    aria-label={label}
  >
    {/* {label} */}
  </button>
);

const ActionButton: React.FC<{ label: string; color: string; isActive: boolean; onClick: () => void }> = ({ label, color, isActive, onClick }) => (
  <div className="relative">
    <button
      onClick={onClick}
      className={`w-14 h-14 md:w-16 md:h-16 rounded-full font-bold text-xl transition-all duration-150 ${color} ${
        isActive ? 'scale-110 shadow-lg' : 'hover:scale-105 active:scale-95'
      }`}
    >
      {label}
    </button>
    {isActive && <div className={`absolute -inset-1 rounded-full ${color} opacity-30 animate-ping`}></div>}
  </div>
);

const SystemButton: React.FC<{ label: string; isActive: boolean; onClick: () => void }> = ({ label, isActive, onClick }) => (
  <button
    onClick={onClick}
    className={`h-6 w-16 bg-neutral-800 rounded-full text-xs font-mono font-bold transition-all duration-150 ${
      isActive ? 'bg-green-glow text-black scale-110 shadow-lg shadow-green-glow/30' : 'hover:bg-neutral-700 text-neutral-400 active:scale-95'
    }`}
  >
    {label}
  </button>
);


const Controls: React.FC<ControlsProps> = ({ lastAction }) => {
  const onAction = (action: string) => {
    // Dispatch custom event for App.tsx to handle
    window.dispatchEvent(new CustomEvent('game-control-press', { detail: action }));
  };

  return (
    <div className="flex items-center justify-around w-full p-4 bg-neutral-900/50 border-t border-neutral-800">
      <div className="grid grid-cols-3 grid-rows-3 gap-1 w-32 h-32 md:w-36 md:h-36">
        <DPadButton
          label="U"
          gridArea="1 / 2 / 2 / 3"
          isActive={lastAction === 'UP'}
          onClick={() => onAction('UP')}
        />
        <DPadButton
          label="L"
          gridArea="2 / 1 / 3 / 2"
          isActive={lastAction === 'LEFT'}
          onClick={() => onAction('LEFT')}
        />
        <div style={{ gridArea: '2 / 2 / 3 / 3' }} className="w-10 h-10 md:w-12 md:h-12 bg-neutral-800 rounded-lg"></div>
        <DPadButton
          label="R"
          gridArea="2 / 3 / 3 / 4"
          isActive={lastAction === 'RIGHT'}
          onClick={() => onAction('RIGHT')}
        />
        <DPadButton
          label="D"
          gridArea="3 / 2 / 4 / 3"
          isActive={lastAction === 'DOWN'}
          onClick={() => onAction('DOWN')}
        />
      </div>

      <div className="flex flex-col items-center justify-center space-y-3">
        <SystemButton
          label="SELECT"
          isActive={lastAction === 'SELECT'}
          onClick={() => onAction('SELECT')}
        />
        <SystemButton
          label="START"
          isActive={lastAction === 'START'}
          onClick={() => onAction('START')}
        />
      </div>

      <div className="flex items-center space-x-4">
        <ActionButton
          label="B"
          color="bg-red-600 text-white shadow-red-600/30"
          isActive={lastAction === 'B'}
          onClick={() => onAction('B')}
        />
        <ActionButton
          label="A"
          color="bg-green-500 text-white shadow-green-500/30"
          isActive={lastAction === 'A'}
          onClick={() => onAction('A')}
        />
      </div>
    </div>
  );
};

export default Controls;
