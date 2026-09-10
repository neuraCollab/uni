import React, { useState, useMemo } from 'react';
import { ALL_QUESTIONS, SECTIONS, InterviewQuestion } from '../data/repoData';
import { 
  Layers, 
  Eye, 
  EyeOff, 
  ArrowLeft, 
  ArrowRight, 
  Shuffle, 
  BookOpen, 
  CheckCircle, 
  HelpCircle,
  RotateCcw
} from 'lucide-react';

interface FlashcardsModeProps {
  filterNotePath?: string | null;
  onClearFilterNote?: () => void;
  onNavigateNote: (notePath: string) => void;
}

export const FlashcardsMode: React.FC<FlashcardsModeProps> = ({
  filterNotePath,
  onClearFilterNote,
  onNavigateNote,
}) => {
  const [selectedSection, setSelectedSection] = useState<string>('all');
  const [isRevealed, setIsRevealed] = useState<boolean>(false);
  const [currentIndex, setCurrentIndex] = useState<number>(0);
  const [masteredIds, setMasteredIds] = useState<Record<string, boolean>>({});

  // Filter questions
  const filteredQuestions = useMemo<InterviewQuestion[]>(() => {
    let list = ALL_QUESTIONS;
    if (filterNotePath) {
      list = list.filter(q => q.notePath === filterNotePath);
    } else if (selectedSection !== 'all') {
      list = list.filter(q => q.section === selectedSection);
    }
    return list;
  }, [selectedSection, filterNotePath]);

  const currentQ = filteredQuestions[currentIndex] || filteredQuestions[0];
  const isMastered = currentQ ? !!masteredIds[currentQ.id] : false;

  const handleNext = () => {
    setIsRevealed(false);
    if (currentIndex < filteredQuestions.length - 1) {
      setCurrentIndex(prev => prev + 1);
    } else {
      setCurrentIndex(0);
    }
  };

  const handlePrev = () => {
    setIsRevealed(false);
    if (currentIndex > 0) {
      setCurrentIndex(prev => prev - 1);
    } else {
      setCurrentIndex(filteredQuestions.length - 1);
    }
  };

  const handleShuffle = () => {
    setIsRevealed(false);
    setCurrentIndex(Math.floor(Math.random() * filteredQuestions.length));
  };

  const toggleMastered = () => {
    if (!currentQ) return;
    setMasteredIds(prev => ({
      ...prev,
      [currentQ.id]: !prev[currentQ.id]
    }));
  };

  const masteredCount = filteredQuestions.filter(q => !!masteredIds[q.id]).length;

  return (
    <div className="flex-1 overflow-y-auto p-4 sm:p-8 max-w-3xl mx-auto w-full">
      {/* Header Banner */}
      <div className="mb-6 pb-6 border-b border-neutral-800">
        <div className="flex items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-amber-500 mb-1">
              <Layers className="w-5 h-5" />
              <span className="text-xs uppercase font-bold tracking-wider">Interview Question Drill</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-neutral-100 tracking-tight">
              257 Technical Flashcards
            </h1>
            <p className="text-xs sm:text-sm text-neutral-400 mt-1">
              Extracted directly from the "Common interview questions" sections across all 95 curated notes.
            </p>
          </div>

          <button
            onClick={handleShuffle}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-neutral-200 text-xs font-medium border border-neutral-700 transition-colors shrink-0"
            title="Shuffle questions"
          >
            <Shuffle className="w-4 h-4 text-amber-400" />
            <span className="hidden sm:inline">Shuffle</span>
          </button>
        </div>

        {/* Note-specific filter badge if active */}
        {filterNotePath && (
          <div className="mt-4 p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/30 flex items-center justify-between text-xs text-amber-300">
            <span>Filtered for: <strong>{currentQ?.noteTitle || filterNotePath}</strong></span>
            {onClearFilterNote && (
              <button
                onClick={onClearFilterNote}
                className="text-xs text-amber-400 hover:text-white underline"
              >
                Clear filter & show all
              </button>
            )}
          </div>
        )}

        {/* Domain Filter Pills */}
        {!filterNotePath && (
          <div className="flex items-center gap-1.5 overflow-x-auto pt-4 pb-1 scrollbar-none">
            <button
              onClick={() => { setSelectedSection('all'); setCurrentIndex(0); setIsRevealed(false); }}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-colors border ${
                selectedSection === 'all'
                  ? 'bg-amber-500/20 text-amber-300 border-amber-500/50'
                  : 'bg-neutral-900 text-neutral-400 border-neutral-800 hover:text-white'
              }`}
            >
              All ({ALL_QUESTIONS.length})
            </button>
            {SECTIONS.map(sec => {
              const count = ALL_QUESTIONS.filter(q => q.section === sec.id).length;
              if (count === 0) return null;
              const isSelected = selectedSection === sec.id;
              return (
                <button
                  key={sec.id}
                  onClick={() => { setSelectedSection(sec.id); setCurrentIndex(0); setIsRevealed(false); }}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-colors border ${
                    isSelected
                      ? 'bg-amber-500/20 text-amber-300 border-amber-500/50'
                      : 'bg-neutral-900 text-neutral-400 border-neutral-800 hover:text-white'
                  }`}
                >
                  {sec.title} ({count})
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Card Arena */}
      {currentQ ? (
        <div className="space-y-6">
          {/* Card Meta & Counter */}
          <div className="flex items-center justify-between text-xs text-neutral-400">
            <div className="flex items-center gap-2">
              <span className="font-mono text-neutral-200 font-bold">
                {currentIndex + 1} of {filteredQuestions.length}
              </span>
              <span>•</span>
              <span className="text-amber-400 font-semibold uppercase tracking-wider text-[11px]">
                {currentQ.section.replace(/-/g, ' ')}
              </span>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-neutral-400 font-mono">Mastered: {masteredCount}</span>
              <button
                onClick={toggleMastered}
                className={`p-1.5 rounded-md border transition-colors ${
                  isMastered
                    ? 'bg-emerald-950 text-emerald-400 border-emerald-500/50'
                    : 'bg-neutral-800 text-neutral-400 border-neutral-700 hover:text-white'
                }`}
                title={isMastered ? 'Mark as learning' : 'Mark as mastered'}
              >
                <CheckCircle className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Flashcard Box */}
          <div className="p-6 sm:p-8 rounded-2xl bg-neutral-900 border border-neutral-800 shadow-2xl min-h-[260px] flex flex-col justify-between">
            <div className="space-y-4">
              <div className="flex items-center justify-between text-xs text-neutral-400 pb-2 border-b border-neutral-800/80">
                <span className="text-amber-400/90 font-medium">{currentQ.noteTitle}</span>
                <span className="text-[11px] font-mono text-neutral-400">Interview Question</span>
              </div>

              <div className="text-lg sm:text-xl font-semibold text-neutral-100 leading-snug py-2">
                {currentQ.question}
              </div>

              {/* Reveal Hint / Concept */}
              {isRevealed ? (
                <div className="p-4 rounded-xl bg-amber-500/5 border border-amber-500/20 text-neutral-200 text-sm leading-relaxed animate-in fade-in duration-200">
                  <div className="text-xs font-bold uppercase tracking-wider text-amber-400 mb-1 flex items-center gap-1.5">
                    <HelpCircle className="w-3.5 h-3.5" />
                    <span>Answer Key / Model Explanation</span>
                  </div>
                  <p className="mt-1">
                    {currentQ.hint ? currentQ.hint : "Review the full explanation, trade-offs, and examples in the underlying note."}
                  </p>
                </div>
              ) : (
                <button
                  onClick={() => setIsRevealed(true)}
                  className="w-full py-4 rounded-xl border border-dashed border-neutral-700 hover:border-amber-500/50 bg-neutral-950/40 hover:bg-neutral-950 text-neutral-400 hover:text-amber-400 text-xs font-medium transition-all flex items-center justify-center gap-2"
                >
                  <Eye className="w-4 h-4" />
                  <span>Click to reveal key concept / answer notes</span>
                </button>
              )}
            </div>

            {/* Actions at bottom of card */}
            <div className="pt-6 mt-4 border-t border-neutral-800 flex items-center justify-between gap-3">
              <button
                onClick={() => onNavigateNote(currentQ.notePath)}
                className="flex items-center gap-1.5 text-xs text-neutral-400 hover:text-amber-400 transition-colors"
              >
                <BookOpen className="w-3.5 h-3.5" />
                <span>Open in note context</span>
              </button>

              <div className="flex items-center gap-2">
                {isRevealed && (
                  <button
                    onClick={() => setIsRevealed(false)}
                    className="p-2 rounded-lg bg-neutral-800 text-neutral-400 hover:text-white text-xs transition-colors"
                    title="Hide answer"
                  >
                    <EyeOff className="w-4 h-4" />
                  </button>
                )}
                <button
                  onClick={handlePrev}
                  className="p-2 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-neutral-200 transition-colors"
                  title="Previous question"
                >
                  <ArrowLeft className="w-4 h-4" />
                </button>
                <button
                  onClick={handleNext}
                  className="flex items-center gap-1 px-4 py-2 rounded-lg bg-amber-500 hover:bg-amber-400 text-neutral-950 font-bold text-xs transition-colors shadow-sm"
                >
                  <span>Next</span>
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="p-8 text-center text-neutral-400">
          No questions match this filter.
        </div>
      )}
    </div>
  );
};
