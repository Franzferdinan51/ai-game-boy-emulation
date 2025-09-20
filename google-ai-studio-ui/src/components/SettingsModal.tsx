import React, { useState } from 'react';
import { Settings } from '../types';
import {
  XMarkIcon,
  SunIcon,
  MoonIcon,
  ComputerDesktopIcon,
  SparklesIcon,
  BellIcon,
  FilmIcon,
  Cog6ToothIcon,
  ArrowPathIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  settings: Settings;
  onSave: (settings: Settings) => void;
  theme: 'light' | 'dark';
}

const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  settings,
  onSave,
  theme
}) => {
  const [localSettings, setLocalSettings] = useState<Settings>(settings);
  const [activeTab, setActiveTab] = useState<'general' | 'ai' | 'advanced'>('general');

  const handleSave = () => {
    onSave(localSettings);
    onClose();
  };

  const resetToDefaults = () => {
    setLocalSettings({
      theme: 'dark',
      aiProvider: 'gemini',
      aiModel: 'gemini-pro',
      autoSave: true,
      notifications: true,
      streamingQuality: 'high'
    });
  };

  if (!isOpen) return null;

  const modalBg = theme === 'dark' ? 'bg-gray-900' : 'bg-white';
  const borderColor = theme === 'dark' ? 'border-gray-800' : 'border-gray-200';
  const inputBg = theme === 'dark' ? 'bg-gray-800' : 'bg-gray-50';
  const textColor = theme === 'dark' ? 'text-gray-100' : 'text-gray-900';
  const secondaryTextColor = theme === 'dark' ? 'text-gray-400' : 'text-gray-600';

  const aiProviders = [
    { id: 'gemini', name: 'Google Gemini', models: ['gemini-pro', 'gemini-pro-vision'] },
    { id: 'openrouter', name: 'OpenRouter', models: ['anthropic/claude-3-sonnet', 'openai/gpt-4'] },
    { id: 'nvidia', name: 'NVIDIA NIM', models: ['llama2-70b', 'mixtral-8x7b'] },
    { id: 'openai', name: 'OpenAI Compatible', models: ['gpt-3.5-turbo', 'gpt-4'] },
    { id: 'mock', name: 'Mock Provider', models: ['mock-model'] }
  ];

  const tabs = [
    { id: 'general', label: 'General', icon: Cog6ToothIcon },
    { id: 'ai', label: 'AI Settings', icon: SparklesIcon },
    { id: 'advanced', label: 'Advanced', icon: InformationCircleIcon }
  ];

  const currentProvider = aiProviders.find(p => p.id === localSettings.aiProvider);

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className={`w-full max-w-2xl rounded-xl shadow-2xl ${modalBg} ${borderColor} border`}>
        {/* Header */}
        <div className={`flex items-center justify-between p-6 border-b ${borderColor}`}>
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
              <Cog6ToothIcon className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold">Settings</h2>
              <p className={`text-sm ${secondaryTextColor}`}>
                Configure your AI Game Studio experience
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className={`p-2 rounded-lg ${theme === 'dark' ? 'hover:bg-gray-800' : 'hover:bg-gray-100'} transition-colors`}
          >
            <XMarkIcon className="w-5 h-5" />
          </button>
        </div>

        {/* Tabs */}
        <div className={`flex border-b ${borderColor}`}>
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors relative ${
                activeTab === tab.id
                  ? theme === 'dark' ? 'text-blue-400' : 'text-blue-600'
                  : secondaryTextColor
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
              {activeTab === tab.id && (
                <div className={`absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-blue-500 to-purple-600`}></div>
              )}
            </button>
          ))}
        </div>

        {/* Settings Content */}
        <div className="p-6 max-h-96 overflow-y-auto">
          {activeTab === 'general' && (
            <div className="space-y-6">
              {/* Theme Settings */}
              <div>
                <label className={`block text-sm font-medium ${textColor} mb-3`}>
                  Appearance
                </label>
                <div className="grid grid-cols-3 gap-3">
                  {[
                    { value: 'light', label: 'Light', icon: SunIcon },
                    { value: 'dark', label: 'Dark', icon: MoonIcon },
                    { value: 'auto', label: 'Auto', icon: ComputerDesktopIcon }
                  ].map((option) => (
                    <button
                      key={option.value}
                      onClick={() => setLocalSettings(prev => ({ ...prev, theme: option.value as any }))}
                      className={`flex flex-col items-center gap-2 p-4 rounded-lg border-2 transition-colors ${
                        localSettings.theme === option.value
                          ? 'border-blue-500 bg-blue-500/10'
                          : borderColor
                      } ${theme === 'dark' ? 'hover:bg-gray-800' : 'hover:bg-gray-50'}`}
                    >
                      <option.icon className={`w-6 h-6 ${
                        localSettings.theme === option.value ? 'text-blue-500' : secondaryTextColor
                      }`} />
                      <span className={`text-sm ${
                        localSettings.theme === option.value ? textColor : secondaryTextColor
                      }`}>
                        {option.label}
                      </span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Notification Settings */}
              <div>
                <label className={`block text-sm font-medium ${textColor} mb-3`}>
                  Notifications
                </label>
                <div className="space-y-3">
                  <label className="flex items-center justify-between p-3 rounded-lg border">
                    <div className="flex items-center gap-3">
                      <BellIcon className={`w-5 h-5 ${secondaryTextColor}`} />
                      <span className={textColor}>Enable notifications</span>
                    </div>
                    <input
                      type="checkbox"
                      checked={localSettings.notifications}
                      onChange={(e) => setLocalSettings(prev => ({ ...prev, notifications: e.target.checked }))}
                      className="w-4 h-4 text-blue-500 rounded focus:ring-blue-500"
                    />
                  </label>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'ai' && (
            <div className="space-y-6">
              {/* AI Provider Selection */}
              <div>
                <label className={`block text-sm font-medium ${textColor} mb-3`}>
                  AI Provider
                </label>
                <select
                  value={localSettings.aiProvider}
                  onChange={(e) => {
                    const provider = aiProviders.find(p => p.id === e.target.value);
                    if (provider) {
                      setLocalSettings(prev => ({
                        ...prev,
                        aiProvider: e.target.value,
                        aiModel: provider.models[0]
                      }));
                    }
                  }}
                  className={`w-full px-3 py-2 rounded-lg border ${borderColor} ${inputBg} ${textColor} focus:ring-2 focus:ring-blue-500 focus:border-blue-500`}
                >
                  {aiProviders.map((provider) => (
                    <option key={provider.id} value={provider.id}>
                      {provider.name}
                    </option>
                  ))}
                </select>
              </div>

              {/* AI Model Selection */}
              <div>
                <label className={`block text-sm font-medium ${textColor} mb-3`}>
                  AI Model
                </label>
                <select
                  value={localSettings.aiModel}
                  onChange={(e) => setLocalSettings(prev => ({ ...prev, aiModel: e.target.value }))}
                  className={`w-full px-3 py-2 rounded-lg border ${borderColor} ${inputBg} ${textColor} focus:ring-2 focus:ring-blue-500 focus:border-blue-500`}
                >
                  {currentProvider?.models.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </select>
              </div>

              {/* Streaming Quality */}
              <div>
                <label className={`block text-sm font-medium ${textColor} mb-3`}>
                  Streaming Quality
                </label>
                <div className="grid grid-cols-3 gap-3">
                  {[
                    { value: 'low', label: 'Low (Fast)' },
                    { value: 'medium', label: 'Medium' },
                    { value: 'high', label: 'High (Best)' }
                  ].map((option) => (
                    <button
                      key={option.value}
                      onClick={() => setLocalSettings(prev => ({ ...prev, streamingQuality: option.value as any }))}
                      className={`p-3 rounded-lg border-2 text-center transition-colors ${
                        localSettings.streamingQuality === option.value
                          ? 'border-blue-500 bg-blue-500/10'
                          : borderColor
                      } ${theme === 'dark' ? 'hover:bg-gray-800' : 'hover:bg-gray-50'}`}
                    >
                      <span className={`text-sm font-medium ${
                        localSettings.streamingQuality === option.value ? textColor : secondaryTextColor
                      }`}>
                        {option.label}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'advanced' && (
            <div className="space-y-6">
              {/* Auto Save */}
              <div>
                <label className={`block text-sm font-medium ${textColor} mb-3`}>
                  Data & Storage
                </label>
                <div className="space-y-3">
                  <label className="flex items-center justify-between p-3 rounded-lg border">
                    <div className="flex items-center gap-3">
                      <ArrowPathIcon className={`w-5 h-5 ${secondaryTextColor}`} />
                      <div className="text-left">
                        <span className={textColor}>Auto-save conversations</span>
                        <p className={`text-xs ${secondaryTextColor}`}>
                          Automatically save your chat history
                        </p>
                      </div>
                    </div>
                    <input
                      type="checkbox"
                      checked={localSettings.autoSave}
                      onChange={(e) => setLocalSettings(prev => ({ ...prev, autoSave: e.target.checked }))}
                      className="w-4 h-4 text-blue-500 rounded focus:ring-blue-500"
                    />
                  </label>
                </div>
              </div>

              {/* Info */}
              <div className={`p-4 rounded-lg ${theme === 'dark' ? 'bg-blue-500/10' : 'bg-blue-50'}`}>
                <div className="flex items-start gap-3">
                  <InformationCircleIcon className="w-5 h-5 text-blue-500 mt-0.5" />
                  <div>
                    <h4 className={`font-medium ${textColor} mb-1`}>About AI Game Studio</h4>
                    <p className={`text-sm ${secondaryTextColor}`}>
                      This interface connects to your local AI Game Boy emulator server. All processing happens locally on your machine.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer Actions */}
        <div className={`flex items-center justify-between p-6 border-t ${borderColor}`}>
          <button
            onClick={resetToDefaults}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg ${theme === 'dark' ? 'hover:bg-gray-800' : 'hover:bg-gray-100'} transition-colors`}
          >
            <ArrowPathIcon className="w-4 h-4" />
            Reset to Defaults
          </button>

          <div className="flex items-center gap-3">
            <button
              onClick={onClose}
              className={`px-4 py-2 rounded-lg ${theme === 'dark' ? 'hover:bg-gray-800' : 'hover:bg-gray-100'} transition-colors`}
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white rounded-lg transition-colors"
            >
              Save Changes
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;