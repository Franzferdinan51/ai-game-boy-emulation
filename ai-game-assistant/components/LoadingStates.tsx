import React from 'react';
import { motion } from 'framer-motion';
import { Loader2, Wifi, WifiOff, AlertCircle } from 'lucide-react';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  color?: string;
  className?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'md',
  color = 'text-cyan-500',
  className = ''
}) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12'
  };

  return (
    <motion.div
      className={`${sizeClasses[size]} ${color} ${className}`}
      animate={{ rotate: 360 }}
      transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
    >
      <Loader2 className="w-full h-full" />
    </motion.div>
  );
};

interface SkeletonProps {
  width?: string | number;
  height?: string | number;
  className?: string;
  rounded?: boolean;
}

export const Skeleton: React.FC<SkeletonProps> = ({
  width = '100%',
  height = '1rem',
  className = '',
  rounded = false
}) => {
  return (
    <div
      className={`skeleton ${rounded ? 'rounded-lg' : ''} ${className}`}
      style={{ width, height }}
    />
  );
};

interface LoadingCardProps {
  title?: string;
  subtitle?: string;
  className?: string;
}

export const LoadingCard: React.FC<LoadingCardProps> = ({
  title = 'Loading...',
  subtitle,
  className = ''
}) => {
  return (
    <motion.div
      className={`bg-gray-800 rounded-lg p-6 ${className}`}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="flex items-center space-x-3">
        <LoadingSpinner />
        <div>
          <h3 className="text-lg font-semibold text-gray-100">{title}</h3>
          {subtitle && (
            <p className="text-sm text-gray-400">{subtitle}</p>
          )}
        </div>
      </div>
    </motion.div>
  );
};

interface ConnectionStatusProps {
  status: 'connected' | 'connecting' | 'disconnected' | 'error';
  fps?: number;
  className?: string;
}

export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({
  status,
  fps,
  className = ''
}) => {
  const statusConfig = {
    connected: {
      icon: Wifi,
      color: 'text-green-500',
      bgColor: 'bg-green-500/20',
      text: fps ? `LIVE ${fps} FPS` : 'CONNECTED'
    },
    connecting: {
      icon: Loader2,
      color: 'text-yellow-500',
      bgColor: 'bg-yellow-500/20',
      text: 'CONNECTING...'
    },
    disconnected: {
      icon: WifiOff,
      color: 'text-gray-500',
      bgColor: 'bg-gray-500/20',
      text: 'DISCONNECTED'
    },
    error: {
      icon: AlertCircle,
      color: 'text-red-500',
      bgColor: 'bg-red-500/20',
      text: 'ERROR'
    }
  };

  const config = statusConfig[status];
  const Icon = config.icon;

  return (
    <motion.div
      className={`flex items-center space-x-2 px-3 py-2 rounded-lg ${config.bgColor} ${className}`}
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ duration: 0.2 }}
    >
      <Icon className={`w-4 h-4 ${config.color}`} />
      <span className={`text-xs font-mono font-semibold ${config.color}`}>
        {config.text}
      </span>
    </motion.div>
  );
};

interface ProgressIndicatorProps {
  progress: number;
  text?: string;
  className?: string;
}

export const ProgressIndicator: React.FC<ProgressIndicatorProps> = ({
  progress,
  text,
  className = ''
}) => {
  return (
    <div className={`w-full ${className}`}>
      <div className="flex items-center justify-between mb-2">
        {text && (
          <span className="text-sm text-gray-400">{text}</span>
        )}
        <span className="text-sm text-gray-400">{Math.round(progress)}%</span>
      </div>
      <div className="w-full bg-gray-700 rounded-full h-2">
        <motion.div
          className="bg-gradient-to-r from-cyan-500 to-blue-500 h-2 rounded-full"
          initial={{ width: 0 }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.3, ease: 'easeOut' }}
        />
      </div>
    </div>
  );
};

interface LoadingOverlayProps {
  message?: string;
  showProgress?: boolean;
  progress?: number;
  className?: string;
}

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  message = 'Loading...',
  showProgress = false,
  progress = 0,
  className = ''
}) => {
  return (
    <motion.div
      className={`fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 ${className}`}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.div
        className="bg-gray-800 rounded-lg p-8 max-w-sm w-full mx-4"
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.3, delay: 0.1 }}
      >
        <div className="flex flex-col items-center space-y-4">
          <LoadingSpinner size="lg" />
          <h3 className="text-lg font-semibold text-gray-100 text-center">
            {message}
          </h3>
          {showProgress && (
            <ProgressIndicator progress={progress} className="w-full" />
          )}
        </div>
      </motion.div>
    </motion.div>
  );
};