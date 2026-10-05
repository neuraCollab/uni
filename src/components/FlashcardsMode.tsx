import { useState } from 'react';
import { ArrowLeft, ArrowRight, BookOpen, CheckCircle, Eye, Layers, Shuffle } from 'lucide-react';
import type { Nav } from '../App';
import { QUESTIONS, SECTIONS, humanize, noteByPath } from '../data';
import { Page, PageHeader, Pill, PillRow, btnPrimary, btnSecondary } from './ui';

export function FlashcardsMode({ filterNote, nav }: { filterNote: string; nav: Nav }) {
  const [section, setSection] = useState('all');
  const [index, setIndex] = useState(0);
  const [revealed, setRevealed] = useState(false);
  const [mastered, setMastered] = useState<Set<string>>(new Set());

  const cards = QUESTIONS.filter((q) => (filterNote ? q.notePath === filterNote : section === 'all' || q.section === section));
  const card = cards[index];

  const go = (i: number) => {
    setRevealed(false);
    setIndex((i + cards.length) % cards.length);
  };

  const selectSection = (id: string) => {
    setSection(id);
    setIndex(0);
    setRevealed(false);
  };

  const toggleMastered = () =>
    setMastered((prev) => {
      const next = new Set(prev);
      next.has(card.id) ? next.delete(card.id) : next.add(card.id);
      return next;
    });

  return (
    <Page>
      <PageHeader icon={Layers} kicker="Interview questions" title="Flashcards">
        <p className="text-sm text-neutral-400 mt-1">Pulled from the “Common interview questions” section of every note.</p>
        {filterNote ? (
          <div className="mt-4 p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/30 flex items-center justify-between text-xs text-amber-300">
            <span>
              Only: <strong>{noteByPath.get(filterNote)?.title ?? filterNote}</strong>
            </span>
            <button onClick={() => nav.openFlashcards('')} className="underline hover:text-white">
              Show all
            </button>
          </div>
        ) : (
          <PillRow>
            <Pill active={section === 'all'} onClick={() => selectSection('all')}>
              All ({QUESTIONS.length})
            </Pill>
            {SECTIONS.map((s) => {
              const count = QUESTIONS.filter((q) => q.section === s.id).length;
              return count > 0 ? (
                <Pill key={s.id} active={section === s.id} onClick={() => selectSection(s.id)}>
                  {s.title} ({count})
                </Pill>
              ) : null;
            })}
          </PillRow>
        )}
      </PageHeader>

      {!card ? (
        <div className="p-8 text-center text-neutral-400">No questions here.</div>
      ) : (
        <>
          <div className="flex items-center justify-between text-xs text-neutral-400 mb-4">
            <span className="font-mono">
              {index + 1} / {cards.length} · <span className="text-amber-400 uppercase">{humanize(card.section)}</span>
            </span>
            <span className="flex items-center gap-2 font-mono">
              mastered {cards.filter((q) => mastered.has(q.id)).length}
              <button
                onClick={toggleMastered}
                title={mastered.has(card.id) ? 'Unmark' : 'Mark as mastered'}
                className={`p-1.5 rounded-md border ${mastered.has(card.id) ? 'bg-emerald-950 text-emerald-400 border-emerald-500/50' : 'bg-neutral-800 border-neutral-700 hover:text-white'}`}
              >
                <CheckCircle className="w-4 h-4" />
              </button>
            </span>
          </div>

          <div className="p-6 sm:p-8 rounded-2xl bg-neutral-900 border border-neutral-800 space-y-4">
            <div className="text-xs text-amber-400/90 pb-2 border-b border-neutral-800">{card.noteTitle}</div>
            <div className="text-lg sm:text-xl font-semibold leading-snug">{card.question}</div>

            {revealed ? (
              <div className="p-4 rounded-xl bg-amber-500/5 border border-amber-500/20 text-sm leading-relaxed">
                {card.hint || 'No short answer in the note — open it for the full explanation.'}
              </div>
            ) : (
              <button
                onClick={() => setRevealed(true)}
                className="w-full py-4 rounded-xl border border-dashed border-neutral-700 hover:border-amber-500/50 text-neutral-400 hover:text-amber-400 text-xs flex items-center justify-center gap-2"
              >
                <Eye className="w-4 h-4" /> Reveal hint
              </button>
            )}

            <div className="pt-4 border-t border-neutral-800 flex items-center justify-between gap-3">
              <button onClick={() => nav.openNote(card.notePath)} className="flex items-center gap-1.5 text-xs text-neutral-400 hover:text-amber-400">
                <BookOpen className="w-3.5 h-3.5" /> Open note
              </button>
              <div className="flex items-center gap-2">
                <button onClick={() => go(Math.floor(Math.random() * cards.length))} className={btnSecondary} title="Random card">
                  <Shuffle className="w-4 h-4" />
                </button>
                <button onClick={() => go(index - 1)} className={btnSecondary} title="Previous">
                  <ArrowLeft className="w-4 h-4" />
                </button>
                <button onClick={() => go(index + 1)} className={btnPrimary}>
                  Next <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </>
      )}
    </Page>
  );
}
