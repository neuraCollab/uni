import { useEffect, useState } from 'react';
import { Menu, X } from 'lucide-react';
import { noteByPath } from './data';
import { useRoute } from './hooks';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { NoteViewer } from './components/NoteViewer';
import { PatternQuiz } from './components/PatternQuiz';
import { CramMode } from './components/CramMode';
import { FlashcardsMode } from './components/FlashcardsMode';
import { CodeViewer } from './components/CodeViewer';
import { PatternLookupTable } from './components/PatternLookupTable';
import { SearchModal } from './components/SearchModal';

export interface Nav {
  openNote: (path: string, anchor?: string) => void;
  openCode: (path: string) => void;
  openFlashcards: (notePath: string) => void;
  openCram: (section: string) => void;
}

export default function App() {
  const [route, navigate] = useRoute();
  const [searchOpen, setSearchOpen] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  // Wrapped in an object so re-clicking the same anchor scrolls again.
  const [anchor, setAnchor] = useState<{ id?: string }>({});

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setSearchOpen((open) => !open);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const nav: Nav = {
    openNote: (note, id) => {
      setDrawerOpen(false);
      setAnchor({ id });
      navigate({ view: 'browse', note });
    },
    openCode: (code) => navigate({ view: 'code', code }),
    openFlashcards: (filter) => navigate({ view: 'flashcards', filter }),
    openCram: (section) => navigate({ view: 'cram', section }),
  };

  const note = noteByPath.get(route.note)!;

  return (
    <div className="flex flex-col h-screen bg-neutral-950 text-neutral-100 overflow-hidden">
      <Header view={route.view} onSelectView={(view) => navigate({ view, filter: '' })} onOpenSearch={() => setSearchOpen(true)} />

      <main className="flex-1 flex overflow-hidden relative">
        {route.view === 'browse' && (
          <>
            <button
              onClick={() => setDrawerOpen(!drawerOpen)}
              className="md:hidden fixed bottom-4 right-4 z-40 p-3.5 rounded-full bg-amber-500 text-neutral-950 shadow-2xl"
              aria-label="Toggle notes navigation"
            >
              {drawerOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
            <div className={`fixed inset-0 z-30 md:relative md:inset-auto md:flex ${drawerOpen ? 'flex' : 'hidden'}`}>
              <div className="fixed inset-0 bg-black/60 md:hidden" onClick={() => setDrawerOpen(false)} />
              <div className="relative z-10 h-full w-4/5 max-w-xs md:w-auto md:max-w-none">
                <Sidebar current={note} nav={nav} />
              </div>
            </div>
            <NoteViewer key={note.path} note={note} anchor={anchor} nav={nav} />
          </>
        )}
        {route.view === 'quiz' && <PatternQuiz nav={nav} />}
        {route.view === 'cram' && <CramMode key={route.section} initialSection={route.section || note.section} nav={nav} />}
        {route.view === 'flashcards' && <FlashcardsMode key={route.filter} filterNote={route.filter} nav={nav} />}
        {route.view === 'code' && <CodeViewer path={route.code} nav={nav} />}
        {route.view === 'lookup' && <PatternLookupTable nav={nav} />}
      </main>

      {searchOpen && <SearchModal onClose={() => setSearchOpen(false)} nav={nav} />}
    </div>
  );
}
