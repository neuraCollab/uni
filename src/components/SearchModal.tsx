import React, { useState, useEffect, useMemo, useRef } from 'react';
import { NOTES, CODE_FILES, NoteItem, CodeItem } from '../data/repoData';
import { Search, X, BookOpen, Code2, ArrowRight } from 'lucide-react';

interface SearchModalProps {
  isOpen: boolean;
  onClose: () => void;
  onNavigateNote: (notePath: string) => void;
  onNavigateCode: (codePath: string) => void;
}

interface SearchResult {
  type: 'note' | 'code';
  id: string;
  path: string;
  title: string;
  section: string;
  snippet: string;
  score: number;
}

export const SearchModal: React.FC<SearchModalProps> = ({
  isOpen,
  onClose,
  onNavigateNote,
  onNavigateCode,
}) => {
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 50);
    } else {
      setQuery('');
      setSelectedIndex(0);
    }
  }, [isOpen]);

  // Global hotkey escape
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        if (isOpen) onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  // Search execution
  const results = useMemo<SearchResult[]>(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [];

    const list: SearchResult[] = [];

    // Search notes
    for (const note of NOTES) {
      let score = 0;
      const titleLower = note.title.toLowerCase();
      const contentLower = note.content.toLowerCase();

      if (titleLower.includes(q)) score += 50;
      if (note.path.toLowerCase().includes(q)) score += 20;

      const idx = contentLower.indexOf(q);
      if (idx !== -1) {
        score += 10;
        // Count occurrences
        const occurrences = (contentLower.match(new RegExp(q, 'g')) || []).length;
        score += Math.min(occurrences, 10);

        // Generate snippet around match
        const start = Math.max(0, idx - 40);
        const end = Math.min(note.content.length, idx + q.length + 60);
        let snippet = note.content.substring(start, end).replace(/[#*`_]/g, '');
        if (start > 0) snippet = '...' + snippet;
        if (end < note.content.length) snippet = snippet + '...';

        list.push({
          type: 'note',
          id: note.id,
          path: note.path,
          title: note.title,
          section: note.section,
          snippet,
          score,
        });
      } else if (score > 0) {
        list.push({
          type: 'note',
          id: note.id,
          path: note.path,
          title: note.title,
          section: note.section,
          snippet: note.excerpt || 'Matched in title',
          score,
        });
      }
    }

    // Search code files
    for (const code of CODE_FILES) {
      let score = 0;
      const fileLower = code.filename.toLowerCase();
      const contentLower = code.content.toLowerCase();

      if (fileLower.includes(q)) score += 40;
      const idx = contentLower.indexOf(q);
      if (idx !== -1) {
        score += 8;
        const start = Math.max(0, idx - 30);
        const end = Math.min(code.content.length, idx + q.length + 50);
        const snippet = code.content.substring(start, end);
        list.push({
          type: 'code',
          id: code.id,
          path: code.path,
          title: code.filename,
          section: code.section,
          snippet: `...${snippet.trim()}...`,
          score,
        });
      }
    }

    return list.sort((a, b) => b.score - a.score).slice(0, 20);
  }, [query]);

  const handleSelectResult = (item: SearchResult) => {
    if (item.type === 'note') {
      onNavigateNote(item.path);
    } else {
      onNavigateCode(item.path);
    }
    onClose();
  };

  const handleKeyDownList = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex(prev => (prev < results.length - 1 ? prev + 1 : 0));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex(prev => (prev > 0 ? prev - 1 : results.length - 1));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (results[selectedIndex]) {
        handleSelectResult(results[selectedIndex]);
      }
    }
  };

  if (!isOpen) return null;

  return (
    <div 
      className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-start justify-center p-4 sm:p-6 md:p-12 animate-in fade-in duration-150"
      onClick={onClose}
    >
      <div 
        className="w-full max-w-2xl bg-neutral-900 border border-neutral-700 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh] animate-in zoom-in-95 duration-150"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={handleKeyDownList}
      >
        {/* Search input bar */}
        <div className="flex items-center px-4 py-3.5 border-b border-neutral-800 gap-3 bg-neutral-950/60">
          <Search className="w-5 h-5 text-amber-400 shrink-0" />
          <input
            ref={inputRef}
            type="text"
            placeholder="Search all 95 notes, 44 code files, topics (e.g. data leakage, sliding window, p-value)..."
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSelectedIndex(0);
            }}
            className="flex-1 bg-transparent text-sm sm:text-base text-neutral-100 placeholder-neutral-400 focus:outline-none"
          />
          {query && (
            <button
              onClick={() => setQuery('')}
              className="p-1 text-neutral-400 hover:text-white"
            >
              <X className="w-4 h-4" />
            </button>
          )}
          <kbd className="text-[10px] font-mono bg-neutral-800 text-neutral-400 px-2 py-0.5 rounded border border-neutral-700">
            ESC
          </kbd>
        </div>

        {/* Results Container */}
        <div className="flex-1 overflow-y-auto p-3 space-y-1">
          {query.trim() === '' ? (
            <div className="py-12 text-center text-xs text-neutral-400 space-y-2">
              <p>Type to search across the entire interview knowledge base.</p>
              <div className="flex items-center justify-center gap-2 text-[11px] text-neutral-400">
                <span>Popular:</span>
                <button onClick={() => setQuery('data leakage')} className="underline hover:text-amber-400">data leakage</button>
                <span>•</span>
                <button onClick={() => setQuery('sliding window')} className="underline hover:text-amber-400">sliding window</button>
                <span>•</span>
                <button onClick={() => setQuery('p-value')} className="underline hover:text-amber-400">p-value</button>
                <span>•</span>
                <button onClick={() => setQuery('regularization')} className="underline hover:text-amber-400">regularization</button>
              </div>
            </div>
          ) : results.length === 0 ? (
            <div className="py-12 text-center text-xs text-neutral-400">
              No matching notes or code files found for "{query}".
            </div>
          ) : (
            results.map((item, idx) => {
              const isSelected = idx === selectedIndex;
              return (
                <button
                  key={`${item.type}-${item.path}`}
                  onClick={() => handleSelectResult(item)}
                  onMouseEnter={() => setSelectedIndex(idx)}
                  className={`w-full text-left p-3 rounded-xl transition-colors flex items-start gap-3 border ${
                    isSelected
                      ? 'bg-amber-500/15 border-amber-500/40 text-neutral-100'
                      : 'bg-neutral-900/50 border-transparent text-neutral-300 hover:bg-neutral-800/60'
                  }`}
                >
                  <div className="mt-0.5 shrink-0">
                    {item.type === 'note' ? (
                      <BookOpen className={`w-4 h-4 ${isSelected ? 'text-amber-400' : 'text-neutral-400'}`} />
                    ) : (
                      <Code2 className={`w-4 h-4 ${isSelected ? 'text-emerald-400' : 'text-neutral-400'}`} />
                    )}
                  </div>

                  <div className="flex-1 min-w-0 space-y-0.5">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-sm truncate">{item.title}</span>
                      <span className="text-[10px] px-1.5 py-0.2 rounded bg-neutral-800 text-neutral-400 uppercase font-mono tracking-wider">
                        {item.section.replace(/-/g, ' ')}
                      </span>
                    </div>
                    <p className="text-xs text-neutral-400 line-clamp-2 leading-relaxed font-sans">
                      {item.snippet}
                    </p>
                  </div>

                  <ArrowRight className={`w-4 h-4 shrink-0 mt-1 transition-opacity ${isSelected ? 'opacity-100 text-amber-400' : 'opacity-0'}`} />
                </button>
              );
            })
          )}
        </div>

        {/* Footer info */}
        <div className="px-4 py-2 bg-neutral-950/70 border-t border-neutral-800 text-[11px] text-neutral-400 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span>Use <kbd className="font-mono bg-neutral-800 px-1 rounded">↑</kbd> <kbd className="font-mono bg-neutral-800 px-1 rounded">↓</kbd> to navigate</span>
            <span><kbd className="font-mono bg-neutral-800 px-1 rounded">↵</kbd> to select</span>
          </div>
          <span>{results.length} results</span>
        </div>
      </div>
    </div>
  );
};
