import React from 'react';

export type PokemonType = 'normal' | 'fire' | 'water' | 'electric' | 'grass' | 'ice' | 'fighting' | 'poison' | 'ground' | 'flying' | 'psychic' | 'bug' | 'rock' | 'ghost' | 'dragon' | 'dark' | 'steel' | 'fairy';

export type PokemonStatus = 'psn' | 'brn' | 'frz' | 'par' | 'slp' | null;

interface TypeBadgeProps {
  type: PokemonType;
  className?: string;
}

export const TypeBadge: React.FC<TypeBadgeProps> = ({ type, className = '' }) => {
  const typeColors = {
    normal: 'bg-gray-400',
    fire: 'bg-red-500',
    water: 'bg-blue-500',
    electric: 'bg-yellow-400',
    grass: 'bg-green-500',
    ice: 'bg-cyan-300',
    fighting: 'bg-red-700',
    poison: 'bg-purple-500',
    ground: 'bg-yellow-600',
    flying: 'bg-indigo-300',
    psychic: 'bg-pink-500',
    bug: 'bg-green-400',
    rock: 'bg-yellow-700',
    ghost: 'bg-purple-600',
    dragon: 'bg-indigo-600',
    dark: 'bg-gray-800',
    steel: 'bg-gray-500',
    fairy: 'bg-pink-300'
  };

  return (
    <span className={`px-2 py-1 rounded text-xs font-bold text-white uppercase ${typeColors[type]} ${className}`}>
      {type}
    </span>
  );
};

interface HPBarProps {
  current: number;
  max: number;
  className?: string;
}

export const HPBar: React.FC<HPBarProps> = ({ current, max, className = '' }) => {
  const percentage = (current / max) * 100;
  const hpColor = percentage > 50 ? 'bg-green-500' : percentage > 20 ? 'bg-yellow-500' : 'bg-red-500';

  return (
    <div className={`w-full ${className}`}>
      <div className="flex justify-between text-xs mb-1">
        <span className="font-semibold">HP</span>
        <span>{current}/{max}</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className={`h-2 rounded-full transition-all duration-300 ${hpColor}`}
          style={{ width: `${Math.max(0, percentage)}%` }}
        ></div>
      </div>
    </div>
  );
};

interface StatusBadgeProps {
  status: PokemonStatus;
  className?: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, className = '' }) => {
  if (!status) return null;

  const statusConfig = {
    psn: { label: 'PSN', color: 'bg-purple-500' },
    brn: { label: 'BRN', color: 'bg-red-500' },
    frz: { label: 'FRZ', color: 'bg-cyan-300' },
    par: { label: 'PAR', color: 'bg-yellow-400' },
    slp: { label: 'SLP', color: 'bg-gray-500' }
  };

  const config = statusConfig[status];

  return (
    <span className={`px-2 py-1 rounded text-xs font-bold text-white ${config.color} ${className}`}>
      {config.label}
    </span>
  );
};

interface BattleIndicatorProps {
  isActive?: boolean;
  className?: string;
}

export const BattleIndicator: React.FC<BattleIndicatorProps> = ({ isActive = false, className = '' }) => {
  return (
    <div className={`relative inline-block ${className}`}>
      <div className={`w-3 h-3 rounded-full ${isActive ? 'bg-green-500' : 'bg-gray-500'}`}></div>
      {isActive && (
        <div className="absolute inset-0 rounded-full bg-green-500 animate-ping"></div>
      )}
    </div>
  );
};

interface PokemonCardProps {
  name: string;
  level: number;
  types: PokemonType[];
  currentHP: number;
  maxHP: number;
  status?: PokemonStatus;
  sprite?: string;
  className?: string;
}

export const PokemonCard: React.FC<PokemonCardProps> = ({
  name,
  level,
  types,
  currentHP,
  maxHP,
  status,
  sprite,
  className = ''
}) => {
  return (
    <div className={`bg-gray-50 dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700 ${className}`}>
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="font-bold text-gray-900 dark:text-white capitalize">{name}</h3>
          <p className="text-sm text-gray-600 dark:text-gray-400">Lv. {level}</p>
        </div>
        <div className="flex gap-2">
          {types.map((type, index) => (
            <TypeBadge key={index} type={type} />
          ))}
        </div>
      </div>

      {sprite && (
        <div className="w-20 h-20 mx-auto mb-3 flex items-center justify-center bg-gray-100 dark:bg-gray-700 rounded-lg">
          <div className="text-2xl">{sprite}</div>
        </div>
      )}

      <HPBar current={currentHP} max={maxHP} />

      {status && (
        <div className="mt-2">
          <StatusBadge status={status} />
        </div>
      )}
    </div>
  );
};

interface GameStateDisplayProps {
  step: number;
  location?: string;
  isPlaying?: boolean;
  className?: string;
}

export const GameStateDisplay: React.FC<GameStateDisplayProps> = ({
  step,
  location = 'Unknown',
  isPlaying = false,
  className = ''
}) => {
  return (
    <div className={`bg-gray-900 text-white p-3 rounded-lg ${className}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <BattleIndicator isActive={isPlaying} />
          <div>
            <div className="text-xs text-gray-400">Game State</div>
            <div className="font-semibold">{isPlaying ? 'Playing' : 'Paused'}</div>
          </div>
        </div>

        <div className="text-right">
          <div className="text-xs text-gray-400">Step</div>
          <div className="font-mono font-bold">{step.toLocaleString()}</div>
        </div>

        <div className="text-right">
          <div className="text-xs text-gray-400">Location</div>
          <div className="font-semibold text-sm">{location}</div>
        </div>
      </div>
    </div>
  );
};