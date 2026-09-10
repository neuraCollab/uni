import React, { useState, useMemo } from 'react';
import { PATTERN_CLUES, ALGORITHMS_LOOKUP_TABLE, NoteItem } from '../data/repoData';
import { 
  Sparkles, 
  CheckCircle2, 
  XCircle, 
  ArrowRight, 
  RotateCcw, 
  BookOpen, 
  HelpCircle,
  Shuffle
} from 'lucide-react';

interface PatternQuizProps {
  notes: NoteItem[];
  onNavigateNote: (notePath: string) => void;
}

interface QuizItem {
  id: string;
  clue: string;
  correctPattern: string;
  options: string[];
  notePath: string;
  explanation?: string;
}

export const PatternQuiz: React.FC<PatternQuizProps> = ({ notes, onNavigateNote }) => {
  // Extract all distinct pattern names
  const allPatternNames = useMemo(() => {
    const set = new Set<string>();
    PATTERN_CLUES.forEach(p => set.add(p.pattern));
    ALGORITHMS_LOOKUP_TABLE.forEach(l => {
      if (l.pattern) set.add(l.pattern);
    });
    return Array.from(set);
  }, []);

  // Build quiz questions from lookup table and pattern clues
  const quizItems = useMemo<QuizItem[]>(() => {
    const items: QuizItem[] = [];

    // From lookup table
    ALGORITHMS_LOOKUP_TABLE.forEach((row, idx) => {
      if (!row.clue || !row.pattern) return;

      // Pick 3 random distractors
      const distractors = allPatternNames
        .filter(name => name.toLowerCase() !== row.pattern.toLowerCase())
        .sort(() => 0.5 - Math.random())
        .slice(0, 3);

      const options = [row.pattern, ...distractors].sort(() => 0.5 - Math.random());

      let notePath = row.targetPath;
      if (!notePath) {
        const found = notes.find(n => n.title.toLowerCase() === row.pattern.toLowerCase() && n.section === 'algorithms');
        notePath = found ? found.path : 'algorithms/README.md';
      }

      items.push({
        id: `lookup-${idx}`,
        clue: row.clue,
        correctPattern: row.pattern,
        options,
        notePath,
        explanation: `When a problem states: "${row.clue}", the primary algorithmic pattern to reach for is ${row.pattern}.`,
      });
    });

    // From pattern clues
    PATTERN_CLUES.forEach((pc, idx) => {
      pc.clues.forEach((clueText, clueIdx) => {
        const distractors = allPatternNames
          .filter(name => name.toLowerCase() !== pc.pattern.toLowerCase())
          .sort(() => 0.5 - Math.random())
          .slice(0, 3);

        const options = [pc.pattern, ...distractors].sort(() => 0.5 - Math.random());

        items.push({
          id: `pattern-${idx}-${clueIdx}`,
          clue: clueText,
          correctPattern: pc.pattern,
          options,
          notePath: pc.notePath,
          explanation: `This key clue belongs to the ${pc.pattern} pattern: "${clueText}".`,
        });
      });
    });

    return items.sort(() => 0.5 - Math.random());
  }, [allPatternNames, notes]);

  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedOption, setSelectedOption] = useState<string | null>(null);
  const [score, setScore] = useState(0);
  const [answeredCount, setAnsweredCount] = useState(0);

  const currentItem = quizItems[currentIndex] || quizItems[0];
  const isAnswered = selectedOption !== null;
  const isCorrect = selectedOption?.toLowerCase() === currentItem?.correctPattern?.toLowerCase();

  const handleSelectOption = (opt: string) => {
    if (isAnswered) return;
    setSelectedOption(opt);
    setAnsweredCount(prev => prev + 1);
    if (opt.toLowerCase() === currentItem.correctPattern.toLowerCase()) {
      setScore(prev => prev + 1);
    }
  };

  const handleNext = () => {
    setSelectedOption(null);
    if (currentIndex < quizItems.length - 1) {
      setCurrentIndex(prev => prev + 1);
    } else {
      setCurrentIndex(0);
    }
  };

  const handleRestart = () => {
    setCurrentIndex(0);
    setSelectedOption(null);
    setScore(0);
    setAnsweredCount(0);
  };

  return (
    <div className="flex-1 overflow-y-auto p-4 sm:p-8 max-w-3xl mx-auto w-full">
      {/* Header Banner */}
      <div className="mb-8 pb-6 border-b border-neutral-800">
        <div className="flex items-center gap-2.5 text-amber-400 mb-2">
          <Sparkles className="w-5 h-5" />
          <span className="text-xs uppercase font-bold tracking-wider">Pattern Recognition Trainer</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-neutral-100 tracking-tight">
          Algorithmic Pattern Quiz
        </h1>
        <p className="text-sm text-neutral-400 mt-2 max-w-xl">
          In interviews, recognizing which pattern applies is 80% of the battle. Read the problem statement clue below, identify the correct technique, and inspect the code template.
        </p>

        {/* Scoreboard */}
        <div className="flex items-center justify-between mt-6 p-3 rounded-xl bg-neutral-900 border border-neutral-800 text-xs">
          <div className="flex items-center gap-4">
            <div>
              <span className="text-neutral-400">Question: </span>
              <span className="font-mono text-neutral-200 font-bold">{currentIndex + 1} / {quizItems.length}</span>
            </div>
            <div>
              <span className="text-neutral-400">Score: </span>
              <span className="font-mono text-amber-400 font-bold">{score}</span>
              {answeredCount > 0 && (
                <span className="text-neutral-400 ml-1">
                  ({Math.round((score / answeredCount) * 100)}%)
                </span>
              )}
            </div>
          </div>
          <button
            onClick={handleRestart}
            className="flex items-center gap-1 text-neutral-400 hover:text-neutral-200 transition-colors"
            title="Reset Quiz"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Restart</span>
          </button>
        </div>
      </div>

      {/* Question Card */}
      {currentItem && (
        <div className="space-y-6">
          <div className="p-6 rounded-2xl bg-neutral-900 border border-neutral-800 shadow-lg">
            <div className="flex items-center justify-between gap-2 mb-3">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-amber-500/90 flex items-center gap-1.5">
                <HelpCircle className="w-3.5 h-3.5" />
                <span>Problem Statement Telegraph</span>
              </span>
              <span className="text-[11px] font-mono text-neutral-400">
                #Q{currentIndex + 1}
              </span>
            </div>

            <div className="text-lg sm:text-xl font-medium text-neutral-100 leading-relaxed my-3 bg-neutral-950/60 p-4 rounded-xl border border-neutral-800/80 font-sans">
              "{currentItem.clue.replace(/^-\s*/, '')}"
            </div>

            <p className="text-xs text-neutral-400 mt-3">
              Which algorithmic pattern is indicated by this clue?
            </p>

            {/* Multiple Choice Options */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 mt-5">
              {currentItem.options.map((option, idx) => {
                const isSelected = selectedOption === option;
                const isThisCorrect = option.toLowerCase() === currentItem.correctPattern.toLowerCase();

                let btnStyle = 'bg-neutral-800/70 hover:bg-neutral-800 text-neutral-200 border-neutral-700/80';
                if (isAnswered) {
                  if (isThisCorrect) {
                    btnStyle = 'bg-emerald-950/50 text-emerald-300 border-emerald-500 font-semibold';
                  } else if (isSelected) {
                    btnStyle = 'bg-rose-950/50 text-rose-300 border-rose-500';
                  } else {
                    btnStyle = 'bg-neutral-900/40 text-neutral-400 border-neutral-800/60 opacity-60';
                  }
                }

                return (
                  <button
                    key={idx}
                    id={`quiz-option-${idx}`}
                    disabled={isAnswered}
                    onClick={() => handleSelectOption(option)}
                    className={`p-3.5 rounded-xl border text-sm font-medium text-left transition-all flex items-center justify-between ${btnStyle}`}
                  >
                    <span>{option}</span>
                    {isAnswered && isThisCorrect && (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                    )}
                    {isAnswered && isSelected && !isThisCorrect && (
                      <XCircle className="w-4 h-4 text-rose-400 shrink-0" />
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Feedback & Actions */}
          {isAnswered && (
            <div className="p-5 rounded-2xl bg-neutral-900 border border-neutral-800 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 animate-in fade-in duration-200">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  {isCorrect ? (
                    <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider flex items-center gap-1">
                      <CheckCircle2 className="w-4 h-4" /> Correct Pattern!
                    </span>
                  ) : (
                    <span className="text-xs font-bold text-rose-400 uppercase tracking-wider flex items-center gap-1">
                      <XCircle className="w-4 h-4" /> Not quite!
                    </span>
                  )}
                  <span className="text-xs text-neutral-300 font-medium">
                    Answer: <strong className="text-amber-400">{currentItem.correctPattern}</strong>
                  </span>
                </div>
                <p className="text-xs text-neutral-400">
                  {currentItem.explanation}
                </p>
              </div>

              <div className="flex items-center gap-2 shrink-0 w-full sm:w-auto">
                <button
                  onClick={() => onNavigateNote(currentItem.notePath)}
                  className="flex-1 sm:flex-initial flex items-center justify-center gap-1.5 px-3.5 py-2 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-neutral-200 text-xs font-medium border border-neutral-700 transition-colors"
                >
                  <BookOpen className="w-3.5 h-3.5 text-amber-400" />
                  <span>Open Pattern & Code</span>
                </button>
                <button
                  onClick={handleNext}
                  className="flex-1 sm:flex-initial flex items-center justify-center gap-1.5 px-4 py-2 rounded-lg bg-amber-500 hover:bg-amber-400 text-neutral-950 font-bold text-xs transition-colors shadow-sm"
                >
                  <span>Next Clue</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
