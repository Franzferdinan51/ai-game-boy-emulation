import React, { useState } from 'react';

interface SidebarItem {
  id: string;
  icon: string;
  label: string;
  badge?: string;
  badgeType?: 'live' | 'completed';
  onClick?: () => void;
}

interface SidebarSection {
  title: string;
  items: SidebarItem[];
}

interface CollapsibleSidebarProps {
  sections: SidebarSection[];
  className?: string;
}

const CollapsibleSidebar: React.FC<CollapsibleSidebarProps> = ({
  sections,
  className = ''
}) => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [activeItem, setActiveItem] = useState<string>('');

  const handleItemClick = (item: SidebarItem) => {
    setActiveItem(item.id);
    item.onClick?.();
  };

  const toggleCollapse = () => {
    setIsCollapsed(!isCollapsed);
  };

  const getBadgeClass = (type?: string) => {
    switch (type) {
      case 'live':
        return 'bg-gradient-to-r from-red-500 to-pink-500 animate-pulse';
      case 'completed':
        return 'bg-gradient-to-r from-green-500 to-emerald-500';
      default:
        return 'bg-gray-500';
    }
  };

  return (
    <div className={`bg-gray-50 dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800 transition-all duration-300 ${isCollapsed ? 'w-16' : 'w-64'} ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-800">
        {!isCollapsed && (
          <h2 className="text-lg font-bold text-gray-900 dark:text-white">
            AI Game Control
          </h2>
        )}
        <button
          onClick={toggleCollapse}
          className="p-2 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-800 transition-colors"
        >
          <svg
            className={`w-5 h-5 text-gray-600 dark:text-gray-400 transition-transform ${isCollapsed ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto">
        {sections.map((section, sectionIndex) => (
          <div key={sectionIndex} className="mb-6">
            {/* Section Header */}
            {!isCollapsed && section.title && (
              <div className="px-4 py-2">
                <h3 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  {section.title}
                </h3>
              </div>
            )}

            {/* Section Items */}
            <div className="space-y-1">
              {section.items.map((item) => (
                <button
                  key={item.id}
                  onClick={() => handleItemClick(item)}
                  className={`w-full flex items-center px-4 py-3 text-left transition-colors relative group ${
                    activeItem === item.id
                      ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 border-r-2 border-blue-500'
                      : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
                  }`}
                >
                  {/* Icon */}
                  <span className="text-lg">{item.icon}</span>

                  {/* Label and Badge */}
                  {!isCollapsed && (
                    <div className="flex-1 flex items-center justify-between ml-3">
                      <span className="font-medium truncate">{item.label}</span>

                      {/* Badge */}
                      {item.badge && (
                        <span
                          className={`px-2 py-1 text-xs font-bold text-white rounded-full ${getBadgeClass(item.badgeType)}`}
                        >
                          {item.badge}
                        </span>
                      )}
                    </div>
                  )}

                  {/* Tooltip for collapsed state */}
                  {isCollapsed && (
                    <div className="absolute left-full ml-2 px-2 py-1 bg-gray-800 text-white text-sm rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-50">
                      {item.label}
                      {item.badge && (
                        <span className={`ml-2 px-1 py-0.5 text-xs rounded ${getBadgeClass(item.badgeType)}`}>
                          {item.badge}
                        </span>
                      )}
                    </div>
                  )}
                </button>
              ))}
            </div>
          </div>
        ))}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-800">
        {!isCollapsed ? (
          <div className="flex gap-2">
            <button className="flex-1 bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-700 px-3 py-2 rounded-lg text-sm font-medium transition-colors">
              📊 Stats
            </button>
            <button className="flex-1 bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-700 px-3 py-2 rounded-lg text-sm font-medium transition-colors">
              ⚙️ Settings
            </button>
          </div>
        ) : (
          <div className="space-y-2">
            <button className="w-full p-2 bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-700 rounded-lg transition-colors">
              📊
            </button>
            <button className="w-full p-2 bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-700 rounded-lg transition-colors">
              ⚙️
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default CollapsibleSidebar;