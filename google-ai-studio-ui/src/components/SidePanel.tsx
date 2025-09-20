import React from 'react';

interface SidePanelProps {
  children: React.ReactNode;
  theme: 'light' | 'dark';
}

const SidePanel: React.FC<SidePanelProps> = ({ children, theme }) => {
  const bgColor = theme === 'dark' ? 'bg-gray-800' : 'bg-gray-50';
  const borderColor = theme === 'dark' ? 'border-gray-700' : 'border-gray-200';

  return (
    <div className={`h-full ${bgColor} border-l ${borderColor} flex flex-col`}>
      {children}
    </div>
  );
};

export default SidePanel;