import React, { createContext, useContext, useEffect, useState } from 'react';

type Theme = 'light' | 'dark' | 'system';

interface ThemeContextType {
  theme: Theme;
  currentTheme: 'light' | 'dark';
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

interface ThemeProviderProps {
  children: React.ReactNode;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [theme, setThemeState] = useState<Theme>(() => {
    const saved = localStorage.getItem('theme');
    return (saved as Theme) || 'system';
  });

  const [currentTheme, setCurrentTheme] = useState<'light' | 'dark'>('dark');

  useEffect(() => {
    const applyTheme = (newTheme: 'light' | 'dark') => {
      const root = document.documentElement;

      if (newTheme === 'dark') {
        root.classList.add('dark');
        root.style.setProperty('--bg-primary', '#101014');
        root.style.setProperty('--bg-secondary', '#18181c');
        root.style.setProperty('--bg-tertiary', '#25252b');
        root.style.setProperty('--text-primary', '#f3f4f6');
        root.style.setProperty('--text-secondary', '#9ca3af');
        root.style.setProperty('--border-color', '#36363f');
        root.style.setProperty('--accent-color', '#00f6ff');
      } else {
        root.classList.remove('dark');
        root.style.setProperty('--bg-primary', '#ffffff');
        root.style.setProperty('--bg-secondary', '#f3f4f6');
        root.style.setProperty('--bg-tertiary', '#e5e7eb');
        root.style.setProperty('--text-primary', '#111827');
        root.style.setProperty('--text-secondary', '#6b7280');
        root.style.setProperty('--border-color', '#d1d5db');
        root.style.setProperty('--accent-color', '#0891b2');
      }

      setCurrentTheme(newTheme);
    };

    const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    const themeToApply = theme === 'system' ? systemTheme : theme;
    applyTheme(themeToApply);
  }, [theme]);

  const setTheme = (newTheme: Theme) => {
    setThemeState(newTheme);
    localStorage.setItem('theme', newTheme);
  };

  const toggleTheme = () => {
    if (theme === 'dark') {
      setTheme('light');
    } else if (theme === 'light') {
      setTheme('system');
    } else {
      setTheme('dark');
    }
  };

  const value = {
    theme,
    currentTheme,
    setTheme,
    toggleTheme,
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
};