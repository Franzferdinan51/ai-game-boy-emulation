import React, { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Gamepad2, Circle, Square, Triangle, Plus, Minus, ArrowUp, ArrowDown, ArrowLeft, ArrowRight } from 'lucide-react';

interface TouchControlsProps {
  onAction: (action: string) => void;
  disabled?: boolean;
  className?: string;
}

interface ControlButtonProps {
  icon: React.ReactNode;
  label: string;
  action: string;
  onAction: (action: string) => void;
  disabled?: boolean;
  position?: { top?: string; bottom?: string; left?: string; right?: string };
  size?: 'sm' | 'md' | 'lg';
}

const ControlButton: React.FC<ControlButtonProps> = ({
  icon,
  label,
  action,
  onAction,
  disabled = false,
  position,
  size = 'md'
}) => {
  const [isPressed, setIsPressed] = useState(false);

  const sizeClasses = {
    sm: 'w-12 h-12',
    md: 'w-16 h-16',
    lg: 'w-20 h-20'
  };

  const handlePressStart = useCallback(() => {
    if (disabled) return;
    setIsPressed(true);
    onAction(action);
  }, [action, onAction, disabled]);

  const handlePressEnd = useCallback(() => {
    setIsPressed(false);
  }, []);

  useEffect(() => {
    if (isPressed) {
      const interval = setInterval(() => {
        onAction(action);
      }, 100);
      return () => clearInterval(interval);
    }
  }, [isPressed, action, onAction]);

  return (
    <motion.button
      className={`absolute touch-target ${sizeClasses[size]} bg-gray-800 border-2 border-gray-600 rounded-full flex items-center justify-center text-white active:scale-95 transition-all duration-150 ${
        disabled ? 'opacity-50 cursor-not-allowed' : ''
      } ${isPressed ? 'bg-cyan-500 border-cyan-400 scale-95' : 'hover:bg-gray-700'}`}
      style={position}
      whileTap={{ scale: 0.95 }}
      disabled={disabled}
      onTouchStart={handlePressStart}
      onTouchEnd={handlePressEnd}
      onMouseDown={handlePressStart}
      onMouseUp={handlePressEnd}
      onMouseLeave={handlePressEnd}
      aria-label={label}
    >
      <motion.div
        animate={{ scale: isPressed ? 1.1 : 1 }}
        transition={{ duration: 0.1 }}
      >
        {icon}
      </motion.div>
    </motion.button>
  );
};

interface DPadProps {
  onAction: (action: string) => void;
  disabled?: boolean;
  position?: { top?: string; bottom?: string; left?: string; right?: string };
}

const DPad: React.FC<DPadProps> = ({ onAction, disabled, position }) => {
  return (
    <div className="absolute" style={position}>
      {/* Up */}
      <ControlButton
        icon={<ArrowUp className="w-6 h-6" />}
        label="Up"
        action="UP"
        onAction={onAction}
        disabled={disabled}
        size="sm"
        position={{ top: '0', left: '50%', transform: 'translateX(-50%)' }}
      />

      {/* Down */}
      <ControlButton
        icon={<ArrowDown className="w-6 h-6" />}
        label="Down"
        action="DOWN"
        onAction={onAction}
        disabled={disabled}
        size="sm"
        position={{ bottom: '0', left: '50%', transform: 'translateX(-50%)' }}
      />

      {/* Left */}
      <ControlButton
        icon={<ArrowLeft className="w-6 h-6" />}
        label="Left"
        action="LEFT"
        onAction={onAction}
        disabled={disabled}
        size="sm"
        position={{ left: '0', top: '50%', transform: 'translateY(-50%)' }}
      />

      {/* Right */}
      <ControlButton
        icon={<ArrowRight className="w-6 h-6" />}
        label="Right"
        action="RIGHT"
        onAction={onAction}
        disabled={disabled}
        size="sm"
        position={{ right: '0', top: '50%', transform: 'translateY(-50%)' }}
      />

      {/* Center */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="w-12 h-12 bg-gray-900 rounded-full border-2 border-gray-600"></div>
      </div>
    </div>
  );
};

export const TouchControls: React.FC<TouchControlsProps> = ({
  onAction,
  disabled = false,
  className = ''
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [isLandscape, setIsLandscape] = useState(false);

  // Check orientation and visibility
  useEffect(() => {
    const checkOrientation = () => {
      setIsLandscape(window.innerWidth > window.innerHeight);
    };

    const checkVisibility = () => {
      setIsVisible('ontouchstart' in window || navigator.maxTouchPoints > 0);
    };

    checkOrientation();
    checkVisibility();

    window.addEventListener('resize', checkOrientation);
    window.addEventListener('orientationchange', checkOrientation);

    return () => {
      window.removeEventListener('resize', checkOrientation);
      window.removeEventListener('orientationchange', checkOrientation);
    };
  }, []);

  if (!isVisible) return null;

  return (
    <AnimatePresence>
      <motion.div
        className={`fixed inset-0 pointer-events-none md:hidden ${className}`}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      >
        {/* D-Pad */}
        <DPad
          onAction={onAction}
          disabled={disabled}
          position={{
            bottom: isLandscape ? '20px' : '120px',
            left: '20px'
          }}
        />

        {/* Action Buttons */}
        <div className="absolute" style={{
          bottom: isLandscape ? '20px' : '120px',
          right: '20px'
        }}>
          {/* A Button */}
          <ControlButton
            icon={<Circle className="w-6 h-6" />}
            label="A Button"
            action="A"
            onAction={onAction}
            disabled={disabled}
            size="md"
            position={{ right: '0', top: '0' }}
          />

          {/* B Button */}
          <ControlButton
            icon={<Circle className="w-6 h-6" />}
            label="B Button"
            action="B"
            onAction={onAction}
            disabled={disabled}
            size="md"
            position={{ right: '70px', top: '35px' }}
          />

          {/* Start Button */}
          <ControlButton
            icon={<Plus className="w-4 h-4" />}
            label="Start"
            action="START"
            onAction={onAction}
            disabled={disabled}
            size="sm"
            position={{ right: '35px', top: '-50px' }}
          />

          {/* Select Button */}
          <ControlButton
            icon={<Minus className="w-4 h-4" />}
            label="Select"
            action="SELECT"
            onAction={onAction}
            disabled={disabled}
            size="sm"
            position={{ left: '35px', top: '-50px' }}
          />
        </div>

        {/* Quick Action Menu */}
        <motion.div
          className="absolute top-4 right-4 flex flex-col space-y-2"
          initial={{ x: 100 }}
          animate={{ x: 0 }}
          transition={{ delay: 0.3 }}
        >
          <ControlButton
            icon={<Gamepad2 className="w-4 h-4" />}
            label="Toggle Controls"
            action="TOGGLE_CONTROLS"
            onAction={() => {}}
            disabled={false}
            size="sm"
          />
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

// Gesture controls component
interface GestureControlsProps {
  onSwipe: (direction: 'up' | 'down' | 'left' | 'right') => void;
  onDoubleTap: () => void;
  disabled?: boolean;
}

export const GestureControls: React.FC<GestureControlsProps> = ({
  onSwipe,
  onDoubleTap,
  disabled = false
}) => {
  const [touchStart, setTouchStart] = useState<{ x: number; y: number } | null>(null);
  const [lastTap, setLastTap] = useState<number>(0);

  const handleTouchStart = useCallback((e: React.TouchEvent) => {
    if (disabled) return;
    const touch = e.touches[0];
    setTouchStart({ x: touch.clientX, y: touch.clientY });

    // Double tap detection
    const now = Date.now();
    if (now - lastTap < 300) {
      onDoubleTap();
    }
    setLastTap(now);
  }, [disabled, onDoubleTap, lastTap]);

  const handleTouchEnd = useCallback((e: React.TouchEvent) => {
    if (disabled || !touchStart) return;

    const touch = e.changedTouches[0];
    const deltaX = touch.clientX - touchStart.x;
    const deltaY = touch.clientY - touchStart.y;
    const minSwipeDistance = 50;

    if (Math.abs(deltaX) > Math.abs(deltaY)) {
      // Horizontal swipe
      if (Math.abs(deltaX) > minSwipeDistance) {
        onSwipe(deltaX > 0 ? 'right' : 'left');
      }
    } else {
      // Vertical swipe
      if (Math.abs(deltaY) > minSwipeDistance) {
        onSwipe(deltaY > 0 ? 'down' : 'up');
      }
    }

    setTouchStart(null);
  }, [disabled, touchStart, onSwipe]);

  return (
    <div
      className="fixed inset-0 pointer-events-none md:hidden"
      onTouchStart={handleTouchStart}
      onTouchEnd={handleTouchEnd}
      aria-label="Gesture controls"
    />
  );
};