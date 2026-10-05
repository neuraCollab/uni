import { useState } from 'react';
import { ChevronDown, ChevronRight, Clock, Code2, HelpCircle } from 'lucide-react';
import type { Note } from '../types';
import type { Nav } from '../App';
import { NOTES, SECTIONS, humanize } from '../data';

function groupBySubsection(notes: Note[]) {
  const groups = new Map<string, Note[]>();
  for (const n of notes) {
    const key = n.subsection || 'overview';
    groups.set(key, [...(groups.get(key) ?? []), n]);
  }
  return groups;
}

export function Sidebar({ current, nav }: { current: Note; nav: Nav }) {
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const section = SECTIONS.find((s) => s.id === current.section)!;
  const sectionNotes = NOTES.filter((n) => n.section === section.id);

  const toggle = (key: string) =>
    setCollapsed((prev) => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });

  const openSection = (id: string) => {
    const first = NOTES.find((n) => n.section === id && n.isReadme) ?? NOTES.find((n) => n.section === id);
    if (first) nav.openNote(first.path);
  };

  return (
    <aside className="w-full md:w-80 bg-neutral-900 border-r border-neutral-800 flex flex-col h-full overflow-y-auto">
      <div className="p-3 border-b border-neutral-800 grid grid-cols-2 gap-1.5">
        {SECTIONS.map((s) => (
          <button
            key={s.id}
            onClick={() => openSection(s.id)}
            className={`flex items-center justify-between px-2.5 py-2 rounded-lg text-xs font-medium border transition-colors ${
              s.id === section.id ? 'bg-amber-500/15 text-amber-400 border-amber-500/30' : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/60 border-transparent'
            }`}
          >
            <span className="truncate">{s.title}</span>
            <span className="text-[10px] font-mono text-neutral-500">{NOTES.filter((n) => n.section === s.id).length}</span>
          </button>
        ))}
      </div>

      <div className="p-4 border-b border-neutral-800 flex items-center justify-between gap-2">
        <div className="min-w-0">
          <h2 className="text-sm font-bold">{section.title}</h2>
          <p className="text-xs text-neutral-400 truncate">{section.desc}</p>
        </div>
        <button
          onClick={() => nav.openCram(section.id)}
          className="flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 border border-amber-500/30"
        >
          <Clock className="w-3.5 h-3.5" />
          Cram
        </button>
      </div>

      <div className="flex-1 p-3 space-y-4">
        {[...groupBySubsection(sectionNotes)].map(([key, notes]) => (
          <div key={key} className="space-y-1">
            <button
              onClick={() => toggle(key)}
              className="w-full flex items-center justify-between px-2 py-1 text-xs font-semibold uppercase tracking-wider text-neutral-400 hover:text-neutral-300"
            >
              <span className="flex items-center gap-1.5">
                {collapsed.has(key) ? <ChevronRight className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                {key === 'overview' ? 'Overview' : humanize(key)}
              </span>
              <span className="text-[10px] font-mono">{notes.length}</span>
            </button>

            {!collapsed.has(key) && (
              <div className="space-y-0.5 pl-2 border-l border-neutral-800">
                {notes.map((n) => (
                  <button
                    key={n.path}
                    onClick={() => nav.openNote(n.path)}
                    className={`w-full text-left px-2.5 py-2 rounded-md text-xs transition-colors ${
                      n.path === current.path ? 'bg-amber-500/15 text-amber-300 font-medium' : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/60'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-1">
                      <span className="truncate">{n.title}</span>
                      {n.isScaffolding && <span className="text-[9px] px-1 rounded bg-yellow-950/40 text-yellow-500 shrink-0">WIP</span>}
                    </div>
                    {(n.questionCount > 0 || n.codeRefs.length > 0) && (
                      <div className="flex items-center gap-2 text-[10px] mt-0.5">
                        {n.questionCount > 0 && (
                          <span className="flex items-center gap-0.5 text-amber-400/80">
                            <HelpCircle className="w-3 h-3" />
                            {n.questionCount} Qs
                          </span>
                        )}
                        {n.codeRefs.length > 0 && (
                          <span className="flex items-center gap-0.5 text-emerald-400/80">
                            <Code2 className="w-3 h-3" />
                            {n.codeRefs.length} code
                          </span>
                        )}
                      </div>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </aside>
  );
}
