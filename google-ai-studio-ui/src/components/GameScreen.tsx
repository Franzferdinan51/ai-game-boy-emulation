import React, { useRef, useState, useCallback } from 'react';
import { EmulatorState, GameAction } from '../types';
import StreamingIndicator from './StreamingIndicator';
import GameControls from './GameControls';
import {
  ArrowsPointingInIcon,
  ArrowsPointingOutIcon,
  UploadIcon,
  Cog6ToothIcon,
  PlayIcon,
  PauseIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline';

interface GameScreenProps {
  emulatorState: EmulatorState;
  onLoadROM: (file: File) => void;
  onGameAction: (action: GameAction) => void;
  isFullscreen: boolean;
  onToggleFullscreen: () => void;
  theme: 'light' | 'dark';
}

const GameScreen: React.FC<GameScreenProps> = ({
  emulatorState,
  onLoadROM,
  onGameAction,
  isFullscreen,
  onToggleFullscreen,
  theme
}) => {
  const [showROMDrop, setShowROMDrop] = useState(false);
  const [showControls, setShowControls] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const bgColor = theme === 'dark' ? 'bg-gray-900' : 'bg-gray-100';
  const borderColor = theme === 'dark' ? 'border-gray-800' : 'border-gray-300';
  const textColor = theme === 'dark' ? 'text-gray-300' : 'text-gray-700';

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setShowROMDrop(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setShowROMDrop(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setShowROMDrop(false);

    const files = Array.from(e.dataTransfer.files);
    const romFiles = files.filter(file => {
      const validExtensions = ['.gb', '.gbc', '.gba', '.zip'];
      return validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
    });

    if (romFiles.length > 0) {
      onLoadROM(romFiles[0]);
    }
  }, [onLoadROM]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onLoadROM(file);
    }
  }, [onLoadROM]);

  const triggerFileInput = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const renderEmulatorScreen = () => {
    if (!emulatorState.romName) {
      return (
        <div className="w-full h-full flex items-center justify-center">
          <div className="text-center max-w-md">
            <div className="mb-6">
              <div className="w-24 h-24 mx-auto bg-gradient-to-br from-green-500 to-blue-600 rounded-2xl flex items-center justify-center mb-4">
                <PlayIcon className="w-12 h-12 text-white" />
              </div>
              <h3 className="text-2xl font-bold mb-2">Load a Game</h3>
              <p className={`${textColor} mb-6`}>
                Drag and drop a ROM file here or click to browse. Supported formats: .gb, .gbc, .gba
              </p>
            </div>

            <div className="space-y-3">
              <button
                onClick={triggerFileInput}
                className="w-full px-6 py-3 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
              >
                <UploadIcon className="w-5 h-5" />
                Browse ROM Files
              </button>

              <div className={`text-sm ${textColor}`}>
                <p>Quick tip: Your ROM files will be processed locally</p>
                <p>and streamed to the AI for analysis.</p>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="w-full h-full flex items-center justify-center relative">
        {/* Game Screen */}
        <div className="relative">
          {/* Emulator Display */}
          {emulatorState.screenData ? (
            <div className="relative">
              <img
                src={emulatorState.screenData}
                alt="Game Screen"
                className="max-w-full max-h-full object-contain rounded-lg shadow-2xl"
                style={{
                  imageRendering: 'pixelated',
                  aspectRatio: emulatorState.type === 'gb' ? '10/9' : '3/2'
                }}
              />

              {/* Screen Overlay Info */}
              <div className="absolute top-4 left-4 flex items-center gap-3">
                <StreamingIndicator
                  status={emulatorState.streamingStatus}
                  fps={emulatorState.fps}
                  theme={theme}
                />
                <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                  emulatorState.isRunning
                    ? 'bg-green-500/20 text-green-400'
                    : 'bg-gray-500/20 text-gray-400'
                }`}>
                  {emulatorState.isRunning ? 'AI Playing' : 'Manual Control'}
                </div>
              </div>

              {/* Game Info */}
              <div className="absolute bottom-4 left-4">
                <div className={`px-3 py-2 rounded-lg ${theme === 'dark' ? 'bg-black/70' : 'bg-white/90'}`}>
                  <p className="text-sm font-medium">{emulatorState.romName}</p>
                  <p className="text-xs opacity-70">{emulatorState.type.toUpperCase()} Mode</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="w-[320px] h-[288px] bg-black rounded-lg flex items-center justify-center">
              <div className="text-center">
                <div className="w-16 h-16 mx-auto mb-4 bg-gray-800 rounded-full flex items-center justify-center">
                  <PlayIcon className="w-8 h-8 text-gray-600" />
                </div>
                <p className={`${textColor} text-sm`}>Waiting for game stream...</p>
              </div>
            </div>
          )}
        </div>

        {/* Floating Controls */}
        {showControls && (
          <div className="absolute bottom-4 right-4">
            <GameControls
              onAction={onGameAction}
              theme={theme}
              compact={true}
            />
          </div>
        )}

        {/* Screen Controls Toggle */}
        <button
          onClick={() => setShowControls(!showControls)}
          className={`absolute top-4 right-4 p-2 rounded-lg ${
            theme === 'dark' ? 'bg-black/70 hover:bg-black/80' : 'bg-white/90 hover:bg-white'
          } transition-colors`}
          title={showControls ? 'Hide controls' : 'Show controls'}
        >
          <Cog6ToothIcon className="w-5 h-5" />
        </button>
      </div>
    );
  };

  return (
    <div className={`h-full ${bgColor} ${isFullscreen ? 'p-0' : 'p-4'}`}>
      {/* Drop Zone Overlay */}
      {showROMDrop && (
        <div
          className="absolute inset-0 bg-blue-500/20 backdrop-blur-sm border-2 border-dashed border-blue-400 rounded-lg flex items-center justify-center z-50"
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="text-center">
            <UploadIcon className="w-12 h-12 text-blue-400 mx-auto mb-2" />
            <p className="text-blue-400 font-medium">Drop ROM here to load</p>
          </div>
        </div>
      )}

      {/* Game Container */}
      <div
        className={`h-full rounded-lg border ${borderColor} relative overflow-hidden ${
          showROMDrop ? 'pointer-events-none' : ''
        }`}
        onDragOver={!emulatorState.romName ? handleDragOver : undefined}
        onDragLeave={!emulatorState.romName ? handleDragLeave : undefined}
        onDrop={!emulatorState.romName ? handleDrop : undefined}
      >
        {/* Header */}
        {!isFullscreen && (
          <div className={`border-b ${borderColor} p-4 flex items-center justify-between`}>
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-10 h-10 bg-gradient-to-br from-green-500 to-blue-600 rounded-lg">
                <PlayIcon className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="font-semibold text-lg">Game Screen</h2>
                <p className="text-sm opacity-70">
                  {emulatorState.romName || 'No ROM loaded'}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {emulatorState.romName && (
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="p-2 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
                  title="Load different ROM"
                >
                  <UploadIcon className="w-5 h-5" />
                </button>
              )}

              <button
                onClick={onToggleFullscreen}
                className="p-2 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
                title={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
              >
                {isFullscreen ? (
                  <ArrowsPointingInIcon className="w-5 h-5" />
                ) : (
                  <ArrowsPointingOutIcon className="w-5 h-5" />
                )}
              </button>
            </div>
          </div>
        )}

        {/* Game Display Area */}
        <div className="flex-1 relative">
          {renderEmulatorScreen()}
        </div>

        {/* Bottom Bar */}
        {!isFullscreen && emulatorState.romName && (
          <div className={`border-t ${borderColor} p-4`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${
                    emulatorState.streamingStatus === 'connected'
                      ? 'bg-green-500'
                      : emulatorState.streamingStatus === 'connecting'
                        ? 'bg-yellow-500'
                        : 'bg-red-500'
                  }`}></div>
                  <span>{emulatorState.streamingStatus}</span>
                </div>
                {emulatorState.fps > 0 && (
                  <span>{emulatorState.fps} FPS</span>
                )}
                <span>{emulatorState.type.toUpperCase()}</span>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => setShowControls(!showControls)}
                  className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
                    theme === 'dark'
                      ? 'bg-gray-800 hover:bg-gray-700'
                      : 'bg-gray-200 hover:bg-gray-300'
                  }`}
                >
                  {showControls ? 'Hide Controls' : 'Show Controls'}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".gb,.gbc,.gba,.zip"
        onChange={handleFileSelect}
        className="hidden"
      />
    </div>
  );
};

export default GameScreen;