import React from 'react';
import { ViewMode } from '../types';
import { 
  BookOpen, 
  Sparkles, 
  Flame, 
  Layers, 
  Code2, 
  Search, 
  TableProperties
} from 'lucide-react';

interface HeaderProps {
  currentView: ViewMode;
  onSelectView: (mode: ViewMode) => void;
  onOpenSearch: () => void;
  stats: {
    notesCount: number;
    questionsCount: number;
    codeCount: number;
  };
}

export const Header: React.FC<HeaderProps> = ({
  currentView,
  onSelectView,
  onOpenSearch,
  stats
}) => {
  const navItems: { mode: ViewMode; label: string; icon: React.FC<{ className?: string }>; badge?: string }[] = [
    { mode: 'browse', label: 'Notes & Docs', icon: BookOpen },
    { mode: 'quiz', label: 'Pattern Quiz', icon: Sparkles, badge: 'Algorithms' },
    { mode: 'cram', label: 'Cram Mode', icon: Flame, badge: 'High Signal' },
    { mode: 'flashcards', label: 'Flashcards', icon: Layers, badge: `${stats.questionsCount}` },
    { mode: 'code', label: 'Code Vault', icon: Code2, badge: `${stats.codeCount}` },
    { mode: 'lookup', label: 'Pattern Table', icon: TableProperties },
  ];

  return (
    <header className="bg-neutral-900/90 backdrop-blur-md border-b border-neutral-800 sticky top-0 z-30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 gap-4">
          {/* Brand */}
          <div className="flex items-center gap-3 shrink-0 cursor-pointer" onClick={() => onSelectView('browse')}>
            <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-amber-600 via-orange-500 to-yellow-500 flex items-center justify-center text-white font-bold text-lg shadow-md shadow-orange-950/40">
              IP
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-neutral-100 tracking-tight text-base sm:text-lg">Interview Prep</span>
                <span className="text-[10px] uppercase font-semibold tracking-wider px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
                  Knowledge Base
                </span>
              </div>
              <p className="text-xs text-neutral-400 hidden sm:block">
                Curated 2-minute revision notes, patterns & code
              </p>
            </div>
          </div>

          {/* Search trigger button */}
          <div className="flex-1 max-w-md hidden md:block">
            <button
              id="search-trigger-btn"
              onClick={onOpenSearch}
              className="w-full flex items-center justify-between px-3.5 py-1.5 rounded-lg bg-neutral-800/80 hover:bg-neutral-800 border border-neutral-700/60 text-sm text-neutral-400 hover:text-neutral-200 transition-colors shadow-inner"
            >
              <div className="flex items-center gap-2">
                <Search className="w-4 h-4 text-neutral-400" />
                <span>Search all notes, code, clues...</span>
              </div>
              <kbd className="text-[10px] font-mono bg-neutral-900 px-1.5 py-0.5 rounded border border-neutral-700 text-neutral-400">
                ⌘K
              </kbd>
            </button>
          </div>

          {/* Right Mobile Search Icon */}
          <div className="flex items-center gap-2 md:hidden">
            <button
              onClick={onOpenSearch}
              className="p-2 rounded-lg bg-neutral-800 border border-neutral-700 text-neutral-300 hover:text-white"
              aria-label="Search"
            >
              <Search className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* View mode tabs */}
        <div className="flex items-center gap-1 overflow-x-auto py-1 border-t border-neutral-800/50 scrollbar-none text-xs sm:text-sm">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentView === item.mode;
            return (
              <button
                key={item.mode}
                id={`nav-tab-${item.mode}`}
                onClick={() => onSelectView(item.mode)}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-md font-medium whitespace-nowrap transition-all ${
                  isActive
                    ? 'bg-neutral-800 text-amber-400 shadow-sm border border-neutral-700'
                    : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/50'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-amber-400' : 'text-neutral-400'}`} />
                <span>{item.label}</span>
                {item.badge && (
                  <span className={`text-[10px] px-1.5 py-0.2 rounded-full font-mono ${
                    isActive 
                      ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30' 
                      : 'bg-neutral-800 text-neutral-400 border border-neutral-700'
                  }`}>
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
};
