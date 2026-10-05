import { BookOpen, Code2, Flame, Layers, Search, Sparkles, TableProperties, type LucideIcon } from 'lucide-react';
import type { ViewMode } from '../types';
import { CODE_FILES, QUESTIONS } from '../data';

const TABS: { mode: ViewMode; label: string; icon: LucideIcon; badge?: number }[] = [
  { mode: 'browse', label: 'Notes', icon: BookOpen },
  { mode: 'quiz', label: 'Pattern Quiz', icon: Sparkles },
  { mode: 'lookup', label: 'Pattern Table', icon: TableProperties },
  { mode: 'cram', label: 'Cram Mode', icon: Flame },
  { mode: 'flashcards', label: 'Flashcards', icon: Layers, badge: QUESTIONS.length },
  { mode: 'code', label: 'Code', icon: Code2, badge: CODE_FILES.length },
];

interface Props {
  view: ViewMode;
  onSelectView: (mode: ViewMode) => void;
  onOpenSearch: () => void;
}

export function Header({ view, onSelectView, onOpenSearch }: Props) {
  return (
    <header className="bg-neutral-900 border-b border-neutral-800 px-4 sm:px-6">
      <div className="flex items-center justify-between h-14 gap-4">
        <button className="flex items-center gap-3 shrink-0" onClick={() => onSelectView('browse')}>
          <span className="w-8 h-8 rounded-lg bg-gradient-to-tr from-amber-600 to-yellow-500 flex items-center justify-center text-white font-bold">
            IP
          </span>
          <span className="font-bold tracking-tight text-lg">Interview Prep</span>
        </button>

        <button
          onClick={onOpenSearch}
          aria-label="Search"
          className="flex items-center gap-2 md:flex-1 md:max-w-md px-3 py-1.5 rounded-lg bg-neutral-800 border border-neutral-700 text-sm text-neutral-400 hover:text-neutral-200 transition-colors"
        >
          <Search className="w-4 h-4" />
          <span className="hidden md:inline flex-1 text-left">Search notes and code…</span>
          <kbd className="hidden md:inline text-[10px] font-mono px-1.5 rounded border border-neutral-700">Ctrl K</kbd>
        </button>
      </div>

      <nav className="flex items-center gap-1 overflow-x-auto py-1 text-xs sm:text-sm">
        {TABS.map(({ mode, label, icon: Icon, badge }) => {
          const active = view === mode;
          return (
            <button
              key={mode}
              onClick={() => onSelectView(mode)}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-md font-medium whitespace-nowrap transition-colors ${
                active ? 'bg-neutral-800 text-amber-400' : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/50'
              }`}
            >
              <Icon className="w-4 h-4" />
              {label}
              {badge !== undefined && <span className="text-[10px] px-1.5 rounded-full font-mono bg-neutral-800 border border-neutral-700">{badge}</span>}
            </button>
          );
        })}
      </nav>
    </header>
  );
}
