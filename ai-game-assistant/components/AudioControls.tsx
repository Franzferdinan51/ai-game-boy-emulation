import React, { useState, useEffect, useRef } from 'react';

interface AudioControlsProps {
  className?: string;
  onAudioToggle?: (enabled: boolean) => void;
  onVolumeChange?: (volume: number) => void;
}

const AudioControls: React.FC<AudioControlsProps> = ({
  className = '',
  onAudioToggle,
  onVolumeChange
}) => {
  const [audioEnabled, setAudioEnabled] = useState<boolean>(false);
  const [volume, setVolume] = useState<number>(50);
  const [audioContext, setAudioContext] = useState<AudioContext | null>(null);
  const [audioBuffer, setAudioBuffer] = useState<AudioBuffer | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [audioStatus, setAudioStatus] = useState<string>('disconnected');

  const audioContextRef = useRef<AudioContext | null>(null);
  const audioSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);

  // Initialize Web Audio API
  useEffect(() => {
    const initAudio = async () => {
      try {
        const context = new (window.AudioContext || (window as any).webkitAudioContext)();
        audioContextRef.current = context;
        setAudioContext(context);

        // Create gain node for volume control
        const gainNode = context.createGain();
        gainNode.connect(context.destination);
        gainNodeRef.current = gainNode;

        setAudioStatus('ready');
      } catch (error) {
        console.error('Failed to initialize audio:', error);
        setAudioStatus('error');
      }
    };

    if (audioEnabled) {
      initAudio();
    } else {
      // Cleanup audio resources
      if (audioSourceRef.current) {
        audioSourceRef.current.stop();
        audioSourceRef.current = null;
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }
      setAudioContext(null);
      setAudioStatus('disconnected');
      setIsPlaying(false);
    }

    return () => {
      if (audioSourceRef.current) {
        audioSourceRef.current.stop();
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, [audioEnabled]);

  // Update volume when changed
  useEffect(() => {
    if (gainNodeRef.current) {
      gainNodeRef.current.gain.value = volume / 100;
    }
  }, [volume]);

  // Handle audio toggle
  const handleAudioToggle = () => {
    const newEnabled = !audioEnabled;
    setAudioEnabled(newEnabled);
    setIsPlaying(false);
    if (onAudioToggle) {
      onAudioToggle(newEnabled);
    }
  };

  // Handle volume change
  const handleVolumeChange = (newVolume: number) => {
    setVolume(newVolume);
    if (onVolumeChange) {
      onVolumeChange(newVolume);
    }
  };

  // Fetch audio data from backend
  const fetchAudioData = async () => {
    if (!audioContext) return;

    try {
      const response = await fetch('/api/audio/stream');
      if (response.ok) {
        const arrayBuffer = await response.arrayBuffer();
        const audioData = await audioContext.decodeAudioData(arrayBuffer);
        setAudioBuffer(audioData);
        setAudioStatus('buffered');
      }
    } catch (error) {
      console.error('Failed to fetch audio data:', error);
      setAudioStatus('error');
    }
  };

  // Play audio buffer
  const playAudio = () => {
    if (!audioContext || !audioBuffer) return;

    try {
      // Stop any currently playing audio
      if (audioSourceRef.current) {
        audioSourceRef.current.stop();
        audioSourceRef.current = null;
      }

      // Create new audio source
      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(gainNodeRef.current!);
      source.start();

      audioSourceRef.current = source;
      setIsPlaying(true);

      // Handle audio end
      source.onended = () => {
        setIsPlaying(false);
        audioSourceRef.current = null;
      };
    } catch (error) {
      console.error('Failed to play audio:', error);
      setAudioStatus('error');
    }
  };

  // Stop audio
  const stopAudio = () => {
    if (audioSourceRef.current) {
      audioSourceRef.current.stop();
      audioSourceRef.current = null;
    }
    setIsPlaying(false);
  };

  // Test audio (play a simple tone)
  const testAudio = () => {
    if (!audioContext) return;

    try {
      // Create a simple test tone
      const sampleRate = audioContext.sampleRate;
      const duration = 0.5; // 0.5 seconds
      const frameCount = sampleRate * duration;

      const arrayBuffer = audioContext.createBuffer(1, frameCount, sampleRate);
      const channelData = arrayBuffer.getChannelData(0);

      // Generate a 440Hz sine wave
      for (let i = 0; i < frameCount; i++) {
        channelData[i] = Math.sin(2 * Math.PI * 440 * i / sampleRate) * 0.3;
      }

      // Stop any currently playing audio
      if (audioSourceRef.current) {
        audioSourceRef.current.stop();
        audioSourceRef.current = null;
      }

      // Create and play the test tone
      const source = audioContext.createBufferSource();
      source.buffer = arrayBuffer;
      source.connect(gainNodeRef.current!);
      source.start();

      audioSourceRef.current = source;
      setIsPlaying(true);

      source.onended = () => {
        setIsPlaying(false);
        audioSourceRef.current = null;
      };
    } catch (error) {
      console.error('Failed to play test tone:', error);
    }
  };

  const getStatusColor = () => {
    switch (audioStatus) {
      case 'ready':
      case 'buffered':
        return 'text-green-500';
      case 'error':
        return 'text-red-500';
      default:
        return 'text-gray-500';
    }
  };

  const getStatusText = () => {
    switch (audioStatus) {
      case 'ready':
        return 'Ready';
      case 'buffered':
        return 'Buffered';
      case 'error':
        return 'Error';
      default:
        return 'Disabled';
    }
  };

  return (
    <div className={`bg-gray-800 rounded-lg p-4 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          🎵 Audio Controls
          <span className={`text-xs ${getStatusColor()}`}>
            ({getStatusText()})
          </span>
        </h3>

        <button
          onClick={handleAudioToggle}
          className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
            audioEnabled
              ? 'bg-red-600 hover:bg-red-700 text-white'
              : 'bg-green-600 hover:bg-green-700 text-white'
          }`}
        >
          {audioEnabled ? '🔇 Disable' : '🔊 Enable'}
        </button>
      </div>

      {audioEnabled && (
        <div className="space-y-4">
          {/* Volume Control */}
          <div className="flex items-center gap-3">
            <span className="text-sm text-gray-300 w-16">Volume:</span>
            <input
              type="range"
              min="0"
              max="100"
              value={volume}
              onChange={(e) => handleVolumeChange(parseInt(e.target.value))}
              className="flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
              disabled={!audioEnabled}
            />
            <span className="text-sm text-gray-300 w-12 text-right">
              {volume}%
            </span>
          </div>

          {/* Audio Controls */}
          <div className="flex gap-2">
            <button
              onClick={fetchAudioData}
              disabled={!audioContext || audioStatus === 'buffered'}
              className="px-3 py-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white text-sm rounded-md transition-colors"
            >
              📥 Buffer Audio
            </button>

            <button
              onClick={playAudio}
              disabled={!audioBuffer || isPlaying}
              className="px-3 py-1 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white text-sm rounded-md transition-colors"
            >
              {isPlaying ? '⏸️ Playing' : '▶️ Play'}
            </button>

            <button
              onClick={stopAudio}
              disabled={!isPlaying}
              className="px-3 py-1 bg-yellow-600 hover:bg-yellow-700 disabled:bg-gray-600 text-white text-sm rounded-md transition-colors"
            >
              ⏹️ Stop
            </button>

            <button
              onClick={testAudio}
              disabled={!audioContext}
              className="px-3 py-1 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 text-white text-sm rounded-md transition-colors"
            >
              🔊 Test
            </button>
          </div>

          {/* Audio Status */}
          <div className="text-xs text-gray-400 space-y-1">
            <div>Context: {audioContext ? '✅ Active' : '❌ Inactive'}</div>
            <div>Buffer: {audioBuffer ? '✅ Ready' : '❌ Empty'}</div>
            <div>Source: {audioSourceRef.current ? '✅ Playing' : '❌ Stopped'}</div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AudioControls;