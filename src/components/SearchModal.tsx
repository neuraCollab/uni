import { useEffect, useMemo, useState, type KeyboardEvent } from 'react';
import { ArrowRight, BookOpen, Code2, Search } from 'lucide-react';
import type { Nav } from '../App';
import { CODE_FILES, NOTES, humanize } from '../data';

interface Result {
  type: 'note' | 'code';
  path: string;
  title: string;
  section: string;
  snippet: string;
  score: number;
}

const SUGGESTIONS = ['data leakage', 'sliding window', 'p-value', 'regularization'];

function snippetAround(text: string, idx: number, len: number) {
  const start = Math.max(0, idx - 40);
  const end = Math.min(text.length, idx + len + 60);
  return `${start > 0 ? '…' : ''}${text.slice(start, end).replace(/[#*`_]/g, '')}${end < text.length ? '…' : ''}`;
}

function countOccurrences(haystack: string, needle: string) {
  let n = 0;
  for (let i = haystack.indexOf(needle); i !== -1 && n < 10; i = haystack.indexOf(needle, i + needle.length)) n++;
  return n;
}

function search(query: string): Result[] {
  const q = query.trim().toLowerCase();
  if (!q) return [];
  const results: Result[] = [];

  for (const n of NOTES) {
    const body = n.content.toLowerCase();
    const idx = body.indexOf(q);
    const score = (n.title.toLowerCase().includes(q) ? 50 : 0) + (n.path.includes(q) ? 20 : 0) + (idx !== -1 ? 10 + countOccurrences(body, q) : 0);
    if (score > 0) {
      results.push({ type: 'note', path: n.path, title: n.title, section: n.section, score, snippet: idx !== -1 ? snippetAround(n.content, idx, q.length) : n.excerpt });
    }
  }

  for (const c of CODE_FILES) {
    const idx = c.content.toLowerCase().indexOf(q);
    const score = (c.filename.toLowerCase().includes(q) ? 40 : 0) + (idx !== -1 ? 8 : 0);
    if (score > 0) {
      results.push({ type: 'code', path: c.path, title: c.filename, section: c.section, score, snippet: idx !== -1 ? snippetAround(c.content, idx, q.length) : c.path });
    }
  }

  return results.sort((a, b) => b.score - a.score).slice(0, 20);
}

export function SearchModal({ onClose, nav }: { onClose: () => void; nav: Nav }) {
  const [query, setQuery] = useState('');
  const [selected, setSelected] = useState(0);
  const results = useMemo(() => search(query), [query]);

  useEffect(() => setSelected(0), [query]);

  const open = (r: Result) => {
    if (r.type === 'note') nav.openNote(r.path);
    else nav.openCode(r.path);
    onClose();
  };

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Escape') onClose();
    if (!results.length) return;
    if (e.key === 'ArrowDown') setSelected((i) => (i + 1) % results.length);
    else if (e.key === 'ArrowUp') setSelected((i) => (i - 1 + results.length) % results.length);
    else if (e.key === 'Enter') open(results[selected]);
    else return;
    e.preventDefault();
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-start justify-center p-4 sm:p-12" onClick={onClose}>
      <div
        className="w-full max-w-2xl bg-neutral-900 border border-neutral-700 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh]"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={onKeyDown}
      >
        <div className="flex items-center px-4 py-3.5 border-b border-neutral-800 gap-3">
          <Search className="w-5 h-5 text-amber-400 shrink-0" />
          <input
            autoFocus
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={`Search ${NOTES.length} notes and ${CODE_FILES.length} code files…`}
            className="flex-1 bg-transparent text-sm sm:text-base placeholder-neutral-500 focus:outline-none"
          />
          <kbd className="text-[10px] font-mono bg-neutral-800 text-neutral-400 px-2 py-0.5 rounded border border-neutral-700">ESC</kbd>
        </div>

        <div className="flex-1 overflow-y-auto p-3 space-y-1">
          {!query.trim() ? (
            <div className="py-10 text-center text-xs text-neutral-400 space-x-2">
              <span>Try:</span>
              {SUGGESTIONS.map((s) => (
                <button key={s} onClick={() => setQuery(s)} className="underline hover:text-amber-400">
                  {s}
                </button>
              ))}
            </div>
          ) : results.length === 0 ? (
            <div className="py-10 text-center text-xs text-neutral-400">Nothing found for “{query}”.</div>
          ) : (
            results.map((r, i) => (
              <button
                key={`${r.type}-${r.path}`}
                onClick={() => open(r)}
                onMouseEnter={() => setSelected(i)}
                className={`w-full text-left p-3 rounded-xl flex items-start gap-3 border ${
                  i === selected ? 'bg-amber-500/15 border-amber-500/40' : 'border-transparent text-neutral-300'
                }`}
              >
                {r.type === 'note' ? <BookOpen className="w-4 h-4 mt-0.5 shrink-0 text-amber-400" /> : <Code2 className="w-4 h-4 mt-0.5 shrink-0 text-emerald-400" />}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-sm truncate">{r.title}</span>
                    <span className="text-[10px] px-1.5 rounded bg-neutral-800 text-neutral-400 uppercase font-mono">{humanize(r.section)}</span>
                  </div>
                  <p className="text-xs text-neutral-400 line-clamp-2">{r.snippet}</p>
                </div>
                {i === selected && <ArrowRight className="w-4 h-4 shrink-0 mt-1 text-amber-400" />}
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
