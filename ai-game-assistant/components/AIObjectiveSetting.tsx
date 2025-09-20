import React, { useState, useEffect } from 'react';

interface AIObjectiveSettingProps {
  aiGoal: string;
  onGoalChange: (goal: string) => void;
  isRunning: boolean;
  className?: string;
}

const AIObjectiveSetting: React.FC<AIObjectiveSettingProps> = ({
  aiGoal,
  onGoalChange,
  isRunning,
  className = ''
}) => {
  const [localGoal, setLocalGoal] = useState(aiGoal);
  const [isExpanded, setIsExpanded] = useState(false);
  const [charCount, setCharCount] = useState(0);

  // Pre-defined objective templates
  const objectiveTemplates = [
    "Defeat the first gym leader",
    "Catch a Pokemon",
    "Explore the current town",
    "Level up your starter Pokemon",
    "Complete the current story objective",
    "Find and use an HM move",
    "Battle a trainer",
    "Visit the Pokemon Center",
    "Buy items from the Pokemart"
  ];

  useEffect(() => {
    setLocalGoal(aiGoal);
    setCharCount(aiGoal.length);
  }, [aiGoal]);

  const handleGoalChange = (newGoal: string) => {
    setLocalGoal(newGoal);
    setCharCount(newGoal.length);
    onGoalChange(newGoal);
  };

  const handleTemplateSelect = (template: string) => {
    handleGoalChange(template);
  };

  const handleClearGoal = () => {
    handleGoalChange('');
  };

  const handleSaveGoal = () => {
    // Goal is already being updated via onGoalChange
    setIsExpanded(false);
  };

  const getStatusColor = () => {
    if (!localGoal.trim()) return 'text-gray-400';
    if (isRunning) return 'text-green-400';
    return 'text-blue-400';
  };

  const getStatusText = () => {
    if (!localGoal.trim()) return 'No objective set';
    if (isRunning) return 'AI is executing objective';
    return 'Objective ready';
  };

  return (
    <div className={`bg-gray-800 rounded-lg p-4 ${className}`}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          🎯 AI Objective
          <span className={`text-xs ${getStatusColor()}`}>
            ({getStatusText()})
          </span>
        </h3>

        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded-md transition-colors"
        >
          {isExpanded ? '▲ Collapse' : '▼ Expand'}
        </button>
      </div>

      {/* Compact view (when collapsed) */}
      {!isExpanded && (
        <div className="space-y-2">
          <div className="text-sm text-gray-300 p-2 bg-gray-700 rounded min-h-[2.5rem] max-h-20 overflow-y-auto">
            {localGoal || (
              <span className="text-gray-500 italic">No objective set...</span>
            )}
          </div>

          <div className="flex items-center justify-between text-xs text-gray-400">
            <span>{charCount}/500 characters</span>
            <span className={getStatusColor()}>
              {getStatusText()}
            </span>
          </div>
        </div>
      )}

      {/* Expanded view */}
      {isExpanded && (
        <div className="space-y-4">
          {/* Objective Input */}
          <div className="space-y-2">
            <label htmlFor="ai-objective" className="block text-sm font-medium text-gray-300">
              Current AI Objective:
            </label>
            <textarea
              id="ai-objective"
              value={localGoal}
              onChange={(e) => handleGoalChange(e.target.value)}
              placeholder="e.g., 'Defeat the first gym leader', 'Catch a wild Pokemon', 'Explore Viridian City'"
              className="w-full p-3 bg-gray-700 border border-gray-600 rounded-md text-sm text-white placeholder-gray-500 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none resize-none"
              rows={3}
              maxLength={500}
              disabled={isRunning}
            />

            <div className="flex items-center justify-between text-xs text-gray-400">
              <span>{charCount}/500 characters</span>
              <span className={getStatusColor()}>
                {getStatusText()}
              </span>
            </div>
          </div>

          {/* Quick Templates */}
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-300">
              Quick Templates:
            </label>
            <div className="grid grid-cols-2 gap-2">
              {objectiveTemplates.slice(0, 6).map((template, index) => (
                <button
                  key={index}
                  onClick={() => handleTemplateSelect(template)}
                  disabled={isRunning}
                  className="px-3 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:cursor-not-allowed text-gray-300 hover:text-white disabled:text-gray-600 text-xs rounded-md transition-colors text-left"
                >
                  {template}
                </button>
              ))}
            </div>

            {/* More templates */}
            <div className="grid grid-cols-3 gap-2">
              {objectiveTemplates.slice(6).map((template, index) => (
                <button
                  key={index + 6}
                  onClick={() => handleTemplateSelect(template)}
                  disabled={isRunning}
                  className="px-3 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:cursor-not-allowed text-gray-300 hover:text-white disabled:text-gray-600 text-xs rounded-md transition-colors text-left"
                >
                  {template}
                </button>
              ))}
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-2">
            <button
              onClick={handleSaveGoal}
              disabled={isRunning || !localGoal.trim()}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white text-sm font-medium rounded-md transition-colors"
            >
              💾 Save Objective
            </button>

            <button
              onClick={handleClearGoal}
              disabled={isRunning}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white text-sm font-medium rounded-md transition-colors"
            >
              🗑️ Clear
            </button>
          </div>

          {/* Status Information */}
          <div className="text-xs text-gray-400 space-y-1">
            <div>💡 <strong>Tip:</strong> Be specific with your objectives for better AI performance</div>
            <div>🎮 <strong>Note:</strong> AI will use this objective to make decisions during gameplay</div>
            <div>⚡ <strong>Tip:</strong> Use templates for common objectives or create your own custom ones</div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AIObjectiveSetting;