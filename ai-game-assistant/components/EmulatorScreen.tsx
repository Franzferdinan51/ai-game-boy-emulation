import React, { useState, useEffect } from 'react';
import { AIState } from '../types';
import type { EmulatorMode } from '../types';

interface EmulatorScreenProps {
  emulatorMode: EmulatorMode;
  romName: string | null;
  onRomLoad: (file: File) => void;
  aiState: AIState;
  screenImage?: string;
  streamingStatus?: 'disconnected' | 'connecting' | 'connected' | 'error' | 'failed';
  streamingInfo?: { fps: number; frameCount: number };
}


const StreamingStatusIndicator: React.FC<{
  status: 'disconnected' | 'connecting' | 'connected' | 'error' | 'failed';
  fps?: number;
}> = ({ status, fps }) => {
  if (status === 'disconnected') return null;

  let text = '';
  let color = '';
  let showPulse = false;

  switch (status) {
    case 'connecting':
      text = 'Connecting...';
      color = 'text-yellow-500';
      showPulse = true;
      break;
    case 'connected':
      text = fps ? `LIVE ${fps} FPS` : 'LIVE';
      color = 'text-green-500';
      break;
    case 'error':
      text = 'Stream Error';
      color = 'text-orange-500';
      break;
    case 'failed':
      text = 'Stream Failed';
      color = 'text-red-500';
      break;
  }

  return (
    <div className="absolute top-2 right-2 bg-black/80 px-2 py-1 rounded pointer-events-none">
      <p className={`font-mono text-xs font-bold ${color} ${showPulse ? 'animate-pulse' : ''}`}>
        {text}
      </p>
    </div>
  );
};


const EmulatorScreen: React.FC<EmulatorScreenProps> = ({
  emulatorMode,
  romName,
  onRomLoad,
  aiState,
  screenImage: externalScreenImage,
  streamingStatus = 'disconnected',
  streamingInfo = { fps: 0, frameCount: 0 }
}) => {
  const [internalScreenImage, setInternalScreenImage] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  // Use external screenImage if provided, otherwise use internal state
  const displayScreenImage = externalScreenImage || internalScreenImage;

  // Fallback fetching - ONLY used if SSE streaming is not available
  useEffect(() => {
    // If external screenImage is provided (SSE is working), don't fetch internally
    if (externalScreenImage) {
      return;
    }

    // If streaming is connected, don't use fallback fetching
    if (streamingStatus === 'connected') {
      return;
    }

    // Clear screen and error when no ROM is loaded
    if (!romName) {
      setInternalScreenImage(null);
      setError(null);
      return;
    }

    const fetchScreen = async () => {
      if (loading) return;

      setLoading(true);
      try {
        // First check if the server is running and ROM is loaded
        const statusResponse = await fetch('http://localhost:5000/api/status');
        if (!statusResponse.ok) {
          throw new Error("Server is not responding");
        }

        const statusData = await statusResponse.json();
        if (!statusData.rom_loaded) {
          setError("No ROM loaded in emulator");
          setInternalScreenImage(null);
          setLoading(false);
          return;
        }

        const response = await fetch('http://localhost:5000/api/screen');
        if (response.ok) {
          const data = await response.json();
          setInternalScreenImage(`data:image/jpeg;base64,${data.image}`);
          setError(null);
        } else if (response.status === 400) {
          // Handle "No ROM loaded" error specifically
          setError("No ROM loaded in emulator");
          setInternalScreenImage(null);
        } else {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
      } catch (error) {
        console.error("Error fetching screen:", error);
        setError("Failed to fetch screen");
        // Fallback to placeholder if there's an error
        const fallbackUrl = emulatorMode === 'gb'
          ? 'https://picsum.photos/seed/gb/320/288'
          : 'https://picsum.photos/seed/gba/480/320';
        setInternalScreenImage(fallbackUrl);
      } finally {
        setLoading(false);
      }
    };

    // Initial fetch
    fetchScreen();

    // Set up periodic refresh only if ROM is loaded and streaming is not working
    let interval: NodeJS.Timeout | null = null;
    if (romName && streamingStatus !== 'connected') {
      interval = setInterval(fetchScreen, 1000); // Slower refresh for fallback
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [romName, emulatorMode, loading, streamingStatus, externalScreenImage]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      onRomLoad(event.target.files[0]);
    }
  };
  
  const handleLoadClick = () => {
    fileInputRef.current?.click();
  };

  // GB is 10:9, GBA is 3:2. Let's approximate.
  const aspectRatio = emulatorMode === 'gb' ? 'aspect-[10/9]' : 'aspect-[3/2]';
  
  return (
    <div className="flex flex-col items-center justify-center p-4 md:p-8 flex-grow">
      <div className={`relative w-full max-w-2xl ${aspectRatio} bg-neutral-900 rounded-lg shadow-2xl shadow-black/50 overflow-hidden border-2 border-neutral-800`}>
        {romName ? (
          <>
            {displayScreenImage ? (
              <div className="relative w-full h-full">
                <img
                  src={displayScreenImage}
                  alt="Game Screen"
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    // Fallback to placeholder if image fails to load
                    const fallbackUrl = emulatorMode === 'gb'
                      ? 'https://picsum.photos/seed/gb/320/288'
                      : 'https://picsum.photos/seed/gba/480/320';
                    (e.target as HTMLImageElement).src = fallbackUrl;
                  }}
                />
                <StreamingStatusIndicator
                  status={streamingStatus}
                  fps={streamingInfo?.fps}
                />
              </div>
            ) : error ? (
              <div className="w-full h-full flex flex-col items-center justify-center bg-black p-4">
                <h2 className="text-2xl font-mono text-red-500">ERROR</h2>
                <p className="text-neutral-400 mt-2">{error}</p>
                <p className="text-neutral-600 mt-2 text-sm">Please check if the backend server is running and a ROM is loaded.</p>
                <StreamingStatusIndicator
                  status={streamingStatus}
                  fps={streamingInfo?.fps}
                />
              </div>
            ) : (
              <div className="w-full h-full flex items-center justify-center bg-black">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-cyan-glow"></div>
                <StreamingStatusIndicator
                  status={streamingStatus}
                  fps={streamingInfo?.fps}
                />
              </div>
            )}
          </>
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center bg-black p-4">
            <h2 className="text-2xl font-mono text-neutral-500">NO ROM LOADED</h2>
            <p className="text-neutral-600 mt-2">Please load a game ROM to begin.</p>
          </div>
        )}
      </div>
      <div className="mt-4 flex items-center space-x-4">
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          className="hidden"
          accept=".gb, .gbc, .gba"
        />
        <button 
          onClick={handleLoadClick}
          className="px-6 py-2 bg-cyan-glow text-neutral-950 font-bold rounded-md hover:bg-opacity-80 transition-all duration-200"
        >
          Load ROM
        </button>
        <div className="h-8 w-[1px] bg-neutral-700"></div>
        <div className="text-sm font-mono text-neutral-400">
          <span className="font-bold text-neutral-300">ROM:</span> {romName || 'N/A'}
        </div>
      </div>
    </div>
  );
};

export default EmulatorScreen;