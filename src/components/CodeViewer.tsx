import { useMemo, useState } from 'react';
import { BookOpen, Check, Copy, FileCode, Search } from 'lucide-react';
import type { Nav } from '../App';
import { CODE_FILES, NOTES, codeByPath, humanize } from '../data';
import { useCopy } from '../hooks';
import { Pill } from './ui';

const SECTIONS = [...new Set(CODE_FILES.map((c) => c.section))];

export function CodeViewer({ path, nav }: { path: string; nav: Nav }) {
  const [query, setQuery] = useState('');
  const [section, setSection] = useState('all');
  const [copied, copy] = useCopy();

  const files = useMemo(() => {
    const q = query.toLowerCase();
    return CODE_FILES.filter(
      (f) => (section === 'all' || f.section === section) && (!q || f.path.toLowerCase().includes(q) || f.content.toLowerCase().includes(q)),
    );
  }, [query, section]);

  const file = codeByPath.get(path) ?? CODE_FILES[0];
  const lines = file.content.split('\n');
  const referencedBy = NOTES.filter((n) => n.codeRefs.includes(file.path));

  return (
    <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
      <div className="w-full md:w-80 max-h-[40vh] md:max-h-none bg-neutral-900 border-b md:border-b-0 md:border-r border-neutral-800 flex flex-col shrink-0">
        <div className="p-3 border-b border-neutral-800">
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-neutral-400" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={`Filter ${CODE_FILES.length} files…`}
              className="w-full pl-8 pr-3 py-1.5 rounded-lg bg-neutral-950 border border-neutral-800 text-xs placeholder-neutral-500 focus:outline-none focus:border-amber-500"
            />
          </div>
          <div className="flex items-center gap-1 overflow-x-auto pt-2">
            <Pill active={section === 'all'} onClick={() => setSection('all')}>All</Pill>
            {SECTIONS.map((s) => (
              <Pill key={s} active={section === s} onClick={() => setSection(s)}>
                {humanize(s)}
              </Pill>
            ))}
          </div>
        </div>
        <div className="flex-1 overflow-y-auto p-2 space-y-0.5">
          {files.map((f) => (
            <button
              key={f.path}
              onClick={() => nav.openCode(f.path)}
              title={f.path}
              className={`w-full text-left px-2.5 py-2 rounded-lg text-xs flex items-center gap-2 font-mono ${
                f.path === file.path ? 'bg-emerald-950/40 text-emerald-300' : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/60'
              }`}
            >
              <FileCode className="w-3.5 h-3.5 shrink-0" />
              <span className="truncate">{f.filename}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 flex flex-col overflow-hidden bg-neutral-950">
        <div className="px-4 py-3 bg-neutral-900 border-b border-neutral-800 flex items-center justify-between gap-3">
          <div className="min-w-0">
            <div className="font-mono text-sm font-bold">{file.filename}</div>
            <div className="text-xs font-mono text-neutral-400 truncate">{file.path}</div>
          </div>
          <button onClick={() => copy(file.content)} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-xs border border-neutral-700">
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            {copied ? 'Copied' : 'Copy'}
          </button>
        </div>

        {referencedBy.length > 0 && (
          <div className="px-4 py-2 bg-amber-500/5 border-b border-amber-500/20 flex items-center gap-2 text-xs text-amber-300 overflow-x-auto">
            <BookOpen className="w-3.5 h-3.5 shrink-0" />
            <span className="font-semibold shrink-0">Referenced in:</span>
            {referencedBy.map((n) => (
              <button key={n.path} onClick={() => nav.openNote(n.path)} className="underline hover:text-white shrink-0">
                {n.title}
              </button>
            ))}
          </div>
        )}

        <div className="flex-1 overflow-auto p-4 font-mono text-xs leading-relaxed flex">
          <div className="select-none text-right pr-4 text-neutral-600 border-r border-neutral-800">
            {lines.map((_, i) => (
              <div key={i}>{i + 1}</div>
            ))}
          </div>
          <pre className="pl-4 text-neutral-200 flex-1">{file.content}</pre>
        </div>
      </div>
    </div>
  );
}
