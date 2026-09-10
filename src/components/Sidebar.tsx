import React, { useState } from 'react';
import { NoteItem, SectionMeta } from '../types';
import { 
  ChevronRight, 
  ChevronDown, 
  BookOpen, 
  Code2, 
  HelpCircle,
  Clock,
  Sparkles
} from 'lucide-react';

interface SidebarProps {
  sections: SectionMeta[];
  notes: NoteItem[];
  selectedSectionId: string;
  selectedNotePath: string;
  onSelectSection: (sectionId: string) => void;
  onSelectNote: (notePath: string) => void;
  onStartCram: (sectionId: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  sections,
  notes,
  selectedSectionId,
  selectedNotePath,
  onSelectSection,
  onSelectNote,
  onStartCram,
}) => {
  const [collapsedSubsections, setCollapsedSubsections] = useState<Record<string, boolean>>({});

  const toggleSubsection = (sub: string) => {
    setCollapsedSubsections(prev => ({
      ...prev,
      [sub]: !prev[sub]
    }));
  };

  // Group notes for the active section by subsection
  const sectionNotes = notes.filter(n => n.section === selectedSectionId);

  const subsectionGroups: { [key: string]: NoteItem[] } = {};
  sectionNotes.forEach(note => {
    const groupKey = note.subsection || 'overview';
    if (!subsectionGroups[groupKey]) {
      subsectionGroups[groupKey] = [];
    }
    subsectionGroups[groupKey].push(note);
  });

  const activeSection = sections.find(s => s.id === selectedSectionId);

  return (
    <aside className="w-full md:w-80 lg:w-88 bg-neutral-900 border-r border-neutral-800 flex flex-col h-full shrink-0 overflow-y-auto">
      {/* Section selector pills */}
      <div className="p-3 border-b border-neutral-800/80 bg-neutral-900/50">
        <label className="text-[11px] font-semibold tracking-wider text-neutral-400 uppercase px-1 mb-2 block">
          Domain Knowledge Base
        </label>
        <div className="grid grid-cols-2 gap-1.5">
          {sections.map(sec => {
            const count = notes.filter(n => n.section === sec.id).length;
            const isSelected = sec.id === selectedSectionId;
            return (
              <button
                key={sec.id}
                id={`sidebar-sec-${sec.id}`}
                onClick={() => {
                  onSelectSection(sec.id);
                  // also select the README or first note of this section
                  const defaultNote = notes.find(n => n.section === sec.id && n.isReadme) || notes.find(n => n.section === sec.id);
                  if (defaultNote) onSelectNote(defaultNote.path);
                }}
                className={`flex items-center justify-between px-2.5 py-2 rounded-lg text-xs font-medium text-left transition-all ${
                  isSelected
                    ? 'bg-amber-500/15 text-amber-400 border border-amber-500/30'
                    : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/60 border border-transparent'
                }`}
              >
                <span className="truncate">{sec.title}</span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-neutral-800 text-neutral-400 font-mono">
                  {count}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Active section header & quick action */}
      <div className="p-4 border-b border-neutral-800 flex items-center justify-between bg-neutral-950/40">
        <div>
          <h2 className="text-sm font-bold text-neutral-100 flex items-center gap-2">
            <span>{activeSection?.title}</span>
            <span className="text-xs font-normal text-neutral-400">({sectionNotes.length} notes)</span>
          </h2>
          <p className="text-xs text-neutral-400 line-clamp-1 mt-0.5">{activeSection?.desc}</p>
        </div>
        <button
          onClick={() => onStartCram(selectedSectionId)}
          title="Start sequential Cram Mode for this topic"
          className="flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 border border-amber-500/30 transition-colors"
        >
          <Clock className="w-3.5 h-3.5" />
          <span>Cram</span>
        </button>
      </div>

      {/* Notes list by subsection */}
      <div className="flex-1 p-3 space-y-4">
        {Object.entries(subsectionGroups).map(([subsectionKey, groupNotes]) => {
          const isCollapsed = !!collapsedSubsections[subsectionKey];
          const displaySubTitle = subsectionKey === 'overview' 
            ? 'Index & Guides' 
            : subsectionKey.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

          return (
            <div key={subsectionKey} className="space-y-1">
              <button
                onClick={() => toggleSubsection(subsectionKey)}
                className="w-full flex items-center justify-between px-2 py-1 text-xs font-semibold uppercase tracking-wider text-neutral-400 hover:text-neutral-300 transition-colors"
              >
                <div className="flex items-center gap-1.5">
                  {isCollapsed ? <ChevronRight className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                  <span>{displaySubTitle}</span>
                </div>
                <span className="text-[10px] font-mono text-neutral-400">
                  {groupNotes.length}
                </span>
              </button>

              {!isCollapsed && (
                <div className="space-y-0.5 pl-2 border-l border-neutral-800">
                  {groupNotes.map(note => {
                    const isSelected = note.path === selectedNotePath;
                    return (
                      <button
                        key={note.path}
                        id={`sidebar-note-${note.id.replace(/[^a-zA-Z0-9_-]/g, '_')}`}
                        onClick={() => onSelectNote(note.path)}
                        className={`w-full text-left px-2.5 py-2 rounded-md text-xs transition-all flex flex-col gap-0.5 ${
                          isSelected
                            ? 'bg-amber-500/15 text-amber-300 font-medium border border-amber-500/30 shadow-sm'
                            : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/60'
                        }`}
                      >
                        <div className="flex items-center justify-between gap-1 w-full">
                          <span className="truncate">{note.title}</span>
                          {note.isScaffolding && (
                            <span className="text-[9px] px-1 py-0.2 rounded bg-yellow-950/40 text-yellow-500 border border-yellow-800/40 shrink-0">
                              WIP
                            </span>
                          )}
                        </div>

                        {/* Metadata tags */}
                        <div className="flex items-center gap-2 text-[10px] text-neutral-400">
                          {note.interviewQuestionsCount > 0 && (
                            <span className="flex items-center gap-0.5 text-amber-400/80">
                              <HelpCircle className="w-3 h-3" />
                              <span>{note.interviewQuestionsCount} Qs</span>
                            </span>
                          )}
                          {note.codeRefs && note.codeRefs.length > 0 && (
                            <span className="flex items-center gap-0.5 text-emerald-400/80">
                              <Code2 className="w-3 h-3" />
                              <span>{note.codeRefs.length} code</span>
                            </span>
                          )}
                          {note.isReadme && (
                            <span className="flex items-center gap-0.5 text-neutral-400">
                              <BookOpen className="w-3 h-3" />
                              <span>Index</span>
                            </span>
                          )}
                        </div>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </aside>
  );
};
