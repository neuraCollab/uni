import React, { useState, useMemo } from 'react';
import { CODE_FILES, NOTES, CodeItem } from '../data/repoData';
import { 
  Code2, 
  Copy, 
  Check, 
  Search, 
  BookOpen, 
  FileCode, 
  ChevronRight,
  ExternalLink
} from 'lucide-react';

interface CodeViewerProps {
  initialCodePath?: string | null;
  onNavigateNote: (notePath: string) => void;
}

export const CodeViewer: React.FC<CodeViewerProps> = ({
  initialCodePath,
  onNavigateNote,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSection, setSelectedSection] = useState<string>('all');
  const [copied, setCopied] = useState(false);

  // Group code files by section
  const sectionsList = useMemo(() => {
    const set = new Set<string>();
    CODE_FILES.forEach(c => set.add(c.section));
    return Array.from(set);
  }, []);

  // Filter code files
  const filteredFiles = useMemo(() => {
    return CODE_FILES.filter(file => {
      const matchSection = selectedSection === 'all' || file.section === selectedSection;
      const matchSearch = searchQuery === '' || 
        file.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
        file.path.toLowerCase().includes(searchQuery.toLowerCase()) ||
        file.content.toLowerCase().includes(searchQuery.toLowerCase());
      return matchSection && matchSearch;
    });
  }, [selectedSection, searchQuery]);

  // Selected file
  const [selectedFilePath, setSelectedFilePath] = useState<string>(() => {
    if (initialCodePath && CODE_FILES.some(c => c.path === initialCodePath)) {
      return initialCodePath;
    }
    return CODE_FILES[0]?.path || '';
  });

  const activeFile = CODE_FILES.find(c => c.path === selectedFilePath) || filteredFiles[0] || CODE_FILES[0];

  const handleCopy = () => {
    if (!activeFile) return;
    navigator.clipboard.writeText(activeFile.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Find notes referencing this code file
  const referencingNotes = useMemo(() => {
    if (!activeFile) return [];
    return NOTES.filter(note => 
      (note.codeRefs || []).some(ref => ref === activeFile.path || ref.endsWith(activeFile.filename) || activeFile.path.endsWith(ref))
    );
  }, [activeFile]);

  return (
    <div className="flex-1 flex flex-col md:flex-row h-full overflow-hidden">
      {/* Left Code File List */}
      <div className="w-full md:w-80 bg-neutral-900 border-r border-neutral-800 flex flex-col h-full shrink-0">
        {/* Top filter and search */}
        <div className="p-3 border-b border-neutral-800 space-y-2">
          <div className="flex items-center gap-2 text-xs font-bold text-neutral-200">
            <Code2 className="w-4 h-4 text-emerald-400" />
            <span>Python Code Vault ({CODE_FILES.length})</span>
          </div>

          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-neutral-400" />
            <input
              type="text"
              placeholder="Filter scripts..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-8 pr-3 py-1.5 rounded-lg bg-neutral-950 border border-neutral-800 text-xs text-neutral-200 placeholder-neutral-400 focus:outline-none focus:border-amber-500"
            />
          </div>

          {/* Section Pills */}
          <div className="flex items-center gap-1 overflow-x-auto pb-1 scrollbar-none">
            <button
              onClick={() => setSelectedSection('all')}
              className={`px-2 py-0.5 rounded text-[11px] font-medium whitespace-nowrap transition-colors ${
                selectedSection === 'all'
                  ? 'bg-neutral-700 text-white'
                  : 'text-neutral-400 hover:text-neutral-200'
              }`}
            >
              All
            </button>
            {sectionsList.map(sec => (
              <button
                key={sec}
                onClick={() => setSelectedSection(sec)}
                className={`px-2 py-0.5 rounded text-[11px] font-medium whitespace-nowrap transition-colors ${
                  selectedSection === sec
                    ? 'bg-emerald-950 text-emerald-300 border border-emerald-500/30'
                    : 'text-neutral-400 hover:text-neutral-200'
                }`}
              >
                {sec.replace(/-/g, ' ')}
              </button>
            ))}
          </div>
        </div>

        {/* Files List */}
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {filteredFiles.map(file => {
            const isSelected = file.path === activeFile?.path;
            return (
              <button
                key={file.path}
                onClick={() => setSelectedFilePath(file.path)}
                className={`w-full text-left px-2.5 py-2 rounded-lg text-xs transition-all flex items-center justify-between ${
                  isSelected
                    ? 'bg-emerald-950/40 text-emerald-300 font-medium border border-emerald-500/40'
                    : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/60'
                }`}
              >
                <div className="flex items-center gap-2 truncate">
                  <FileCode className={`w-3.5 h-3.5 shrink-0 ${isSelected ? 'text-emerald-400' : 'text-neutral-400'}`} />
                  <span className="truncate font-mono">{file.filename}</span>
                </div>
                <span className="text-[10px] text-neutral-400 font-mono shrink-0 ml-1">
                  {file.lines}L
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Right Main Editor View */}
      {activeFile ? (
        <div className="flex-1 flex flex-col h-full overflow-hidden bg-neutral-950">
          {/* File Header Bar */}
          <div className="px-4 py-3 bg-neutral-900 border-b border-neutral-800 flex flex-wrap items-center justify-between gap-3 shrink-0">
            <div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-sm font-bold text-neutral-100">{activeFile.filename}</span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  Python
                </span>
              </div>
              <p className="text-xs font-mono text-neutral-400 mt-0.5">{activeFile.path}</p>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={handleCopy}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-neutral-800 hover:bg-neutral-750 text-neutral-200 text-xs font-medium border border-neutral-700 transition-colors"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copied ? 'Copied' : 'Copy Code'}</span>
              </button>
            </div>
          </div>

          {/* Referenced By Notes Banner if any */}
          {referencingNotes.length > 0 && (
            <div className="px-4 py-2 bg-amber-500/5 border-b border-amber-500/20 flex items-center gap-2 text-xs text-amber-300 overflow-x-auto shrink-0">
              <BookOpen className="w-3.5 h-3.5 text-amber-400 shrink-0" />
              <span className="font-semibold shrink-0">Referenced in theory notes:</span>
              <div className="flex items-center gap-2">
                {referencingNotes.map(n => (
                  <button
                    key={n.path}
                    onClick={() => onNavigateNote(n.path)}
                    className="underline hover:text-white shrink-0 font-medium"
                  >
                    {n.title}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Code Body with Line Numbers */}
          <div className="flex-1 overflow-auto p-4 font-mono text-xs leading-relaxed flex">
            {/* Line numbers gutter */}
            <div className="select-none text-right pr-4 text-neutral-400 font-mono text-xs border-r border-neutral-850">
              {activeFile.content.split('\n').map((_, i) => (
                <div key={i}>{i + 1}</div>
              ))}
            </div>
            {/* Code text */}
            <pre className="pl-4 text-neutral-200 whitespace-pre overflow-x-auto flex-1">
              <code>{activeFile.content}</code>
            </pre>
          </div>
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center p-8 text-neutral-400">
          Select a code file from the left sidebar to inspect.
        </div>
      )}
    </div>
  );
};
