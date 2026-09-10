import React, { useState, useEffect } from 'react';
import { ViewMode, NoteItem } from './types';
import { SECTIONS, NOTES, CODE_FILES, ALL_QUESTIONS } from './data/repoData';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { NoteViewer } from './components/NoteViewer';
import { PatternQuiz } from './components/PatternQuiz';
import { CramMode } from './components/CramMode';
import { FlashcardsMode } from './components/FlashcardsMode';
import { CodeViewer } from './components/CodeViewer';
import { PatternLookupTable } from './components/PatternLookupTable';
import { SearchModal } from './components/SearchModal';
import { Menu, X } from 'lucide-react';

export const App: React.FC = () => {
  // Navigation states
  const [currentView, setCurrentView] = useState<ViewMode>('browse');
  const [selectedSectionId, setSelectedSectionId] = useState<string>('algorithms');
  const [selectedNotePath, setSelectedNotePath] = useState<string>(() => {
    // Default to algorithms/README.md or first note
    const defaultNote = NOTES.find(n => n.path === 'algorithms/README.md') || NOTES[0];
    return defaultNote?.path || '';
  });
  const [selectedCodePath, setSelectedCodePath] = useState<string | null>(null);
  const [flashcardFilterNote, setFlashcardFilterNote] = useState<string | null>(null);
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);

  // Sync with window.location.hash for shareable links
  useEffect(() => {
    const handleHash = () => {
      const hash = window.location.hash.replace(/^#/, '');
      if (!hash) return;
      const params = new URLSearchParams(hash);
      const view = params.get('view') as ViewMode | null;
      const note = params.get('note');
      const code = params.get('code');
      const sec = params.get('section');

      if (view && ['browse', 'quiz', 'cram', 'flashcards', 'code', 'lookup'].includes(view)) {
        setCurrentView(view);
      }
      if (note && NOTES.some(n => n.path === note)) {
        setSelectedNotePath(note);
        const match = NOTES.find(n => n.path === note);
        if (match) setSelectedSectionId(match.section);
      }
      if (code) {
        setSelectedCodePath(code);
      }
      if (sec && SECTIONS.some(s => s.id === sec)) {
        setSelectedSectionId(sec);
      }
    };

    handleHash();
    window.addEventListener('hashchange', handleHash);
    return () => window.removeEventListener('hashchange', handleHash);
  }, []);

  // Update hash when note or view changes
  const updateHash = (view: ViewMode, notePath?: string, codePath?: string) => {
    const params = new URLSearchParams();
    params.set('view', view);
    if (notePath) params.set('note', notePath);
    if (codePath) params.set('code', codePath);
    window.location.hash = params.toString();
  };

  const handleSelectView = (mode: ViewMode) => {
    setCurrentView(mode);
    updateHash(mode, mode === 'browse' ? selectedNotePath : undefined);
  };

  const handleNavigateNote = (path: string) => {
    setSelectedNotePath(path);
    const found = NOTES.find(n => n.path === path);
    if (found) {
      setSelectedSectionId(found.section);
    }
    setCurrentView('browse');
    setIsMobileSidebarOpen(false);
    updateHash('browse', path);
  };

  const handleNavigateCode = (path: string) => {
    setSelectedCodePath(path);
    setCurrentView('code');
    setIsMobileSidebarOpen(false);
    updateHash('code', undefined, path);
  };

  const handleOpenFlashcardsForNote = (notePath: string) => {
    setFlashcardFilterNote(notePath);
    setCurrentView('flashcards');
    updateHash('flashcards');
  };

  const handleStartCram = (sectionId: string) => {
    setSelectedSectionId(sectionId);
    setCurrentView('cram');
    updateHash('cram');
  };

  // Active note item
  const activeNote = NOTES.find(n => n.path === selectedNotePath) || NOTES[0];

  return (
    <div className="flex flex-col h-screen bg-neutral-950 text-neutral-100 antialiased overflow-hidden select-text">
      {/* Top Navigation Bar */}
      <Header
        currentView={currentView}
        onSelectView={handleSelectView}
        onOpenSearch={() => setIsSearchOpen(true)}
        stats={{
          notesCount: NOTES.length,
          questionsCount: ALL_QUESTIONS.length,
          codeCount: CODE_FILES.length,
        }}
      />

      {/* Main Workspace Layout */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Mobile menu toggle bar if in browse mode */}
        {currentView === 'browse' && (
          <div className="md:hidden fixed bottom-4 right-4 z-40">
            <button
              onClick={() => setIsMobileSidebarOpen(!isMobileSidebarOpen)}
              className="p-3.5 rounded-full bg-amber-500 text-neutral-950 font-bold shadow-2xl flex items-center justify-center border border-amber-400"
              aria-label="Toggle notes navigation"
            >
              {isMobileSidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        )}

        {/* View Routing */}
        {currentView === 'browse' && (
          <>
            {/* Desktop and Mobile Drawer Sidebar */}
            <div className={`fixed inset-0 z-30 md:relative md:flex md:inset-auto ${
              isMobileSidebarOpen ? 'flex' : 'hidden md:flex'
            }`}>
              {/* Mobile overlay backdrop */}
              <div 
                className="fixed inset-0 bg-black/60 md:hidden"
                onClick={() => setIsMobileSidebarOpen(false)}
              />
              <div className="relative z-10 h-full w-4/5 max-w-xs md:w-auto md:max-w-none">
                <Sidebar
                  sections={SECTIONS}
                  notes={NOTES}
                  selectedSectionId={selectedSectionId}
                  selectedNotePath={selectedNotePath}
                  onSelectSection={setSelectedSectionId}
                  onSelectNote={handleNavigateNote}
                  onStartCram={handleStartCram}
                />
              </div>
            </div>

            {/* Note Content Viewer */}
            {activeNote ? (
              <NoteViewer
                note={activeNote}
                allNotes={NOTES}
                allCodeFiles={CODE_FILES}
                onNavigateNote={handleNavigateNote}
                onNavigateCode={handleNavigateCode}
                onOpenFlashcardsForNote={handleOpenFlashcardsForNote}
              />
            ) : (
              <div className="flex-1 flex items-center justify-center p-8 text-neutral-400">
                Select a note from the sidebar.
              </div>
            )}
          </>
        )}

        {currentView === 'quiz' && (
          <PatternQuiz
            notes={NOTES}
            onNavigateNote={handleNavigateNote}
          />
        )}

        {currentView === 'cram' && (
          <CramMode
            sections={SECTIONS}
            notes={NOTES}
            initialSectionId={selectedSectionId}
            onNavigateNote={handleNavigateNote}
            onOpenFlashcardsForNote={handleOpenFlashcardsForNote}
          />
        )}

        {currentView === 'flashcards' && (
          <FlashcardsMode
            filterNotePath={flashcardFilterNote}
            onClearFilterNote={() => setFlashcardFilterNote(null)}
            onNavigateNote={handleNavigateNote}
          />
        )}

        {currentView === 'code' && (
          <CodeViewer
            initialCodePath={selectedCodePath}
            onNavigateNote={handleNavigateNote}
          />
        )}

        {currentView === 'lookup' && (
          <PatternLookupTable
            notes={NOTES}
            onNavigateNote={handleNavigateNote}
          />
        )}
      </div>

      {/* Global Search Modal */}
      <SearchModal
        isOpen={isSearchOpen}
        onClose={() => setIsSearchOpen(false)}
        onNavigateNote={handleNavigateNote}
        onNavigateCode={handleNavigateCode}
      />
    </div>
  );
};
export default App;
