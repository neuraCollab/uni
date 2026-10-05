import { useState } from 'react';
import { ArrowRight, BookOpen, CheckCircle2, RotateCcw, Sparkles, XCircle } from 'lucide-react';
import type { Nav } from '../App';
import { LOOKUP, PATTERN_CLUES, noteByPath, shuffle } from '../data';
import { Page, PageHeader, btnPrimary, btnSecondary } from './ui';

interface QuizItem {
  clue: string;
  answer: string;
  options: string[];
  notePath: string;
}

// Lookup-table names are short forms; use the linked note's title so each pattern has one name.
const RAW = [
  ...LOOKUP.map((l) => ({ clue: l.clue, answer: noteByPath.get(l.notePath)?.title ?? l.pattern, notePath: l.notePath || 'algorithms/README.md' })),
  ...PATTERN_CLUES.flatMap((p) => p.clues.map((clue) => ({ clue, answer: p.pattern, notePath: p.notePath }))),
];
const PATTERNS = [...new Set(RAW.map((r) => r.answer))];

function buildQuiz(): QuizItem[] {
  return shuffle(RAW).map((item) => ({
    ...item,
    options: shuffle([item.answer, ...shuffle(PATTERNS.filter((p) => p !== item.answer)).slice(0, 3)]),
  }));
}

export function PatternQuiz({ nav }: { nav: Nav }) {
  const [items, setItems] = useState(buildQuiz);
  const [index, setIndex] = useState(0);
  const [picked, setPicked] = useState<string | null>(null);
  const [score, setScore] = useState({ correct: 0, answered: 0 });

  const item = items[index];
  const answered = picked !== null;

  const pick = (option: string) => {
    if (answered) return;
    setPicked(option);
    setScore((s) => ({ correct: s.correct + Number(option === item.answer), answered: s.answered + 1 }));
  };

  const next = () => {
    setPicked(null);
    setIndex((i) => (i + 1) % items.length);
  };

  const restart = () => {
    setItems(buildQuiz());
    setIndex(0);
    setPicked(null);
    setScore({ correct: 0, answered: 0 });
  };

  const optionStyle = (option: string) => {
    if (!answered) return 'bg-neutral-800/70 hover:bg-neutral-800 border-neutral-700';
    if (option === item.answer) return 'bg-emerald-950/50 text-emerald-300 border-emerald-500';
    if (option === picked) return 'bg-rose-950/50 text-rose-300 border-rose-500';
    return 'border-neutral-800 text-neutral-500';
  };

  return (
    <Page>
      <PageHeader icon={Sparkles} kicker="Algorithms" title="Pattern Quiz">
        <p className="text-sm text-neutral-400 mt-2">Read the clue from a problem statement and name the pattern it points to.</p>
        <div className="flex items-center justify-between mt-5 p-3 rounded-xl bg-neutral-900 border border-neutral-800 text-xs font-mono">
          <span>
            {index + 1} / {items.length} · score <span className="text-amber-400 font-bold">{score.correct}</span>
            {score.answered > 0 && ` (${Math.round((score.correct / score.answered) * 100)}%)`}
          </span>
          <button onClick={restart} className="flex items-center gap-1 text-neutral-400 hover:text-neutral-200">
            <RotateCcw className="w-3.5 h-3.5" /> Restart
          </button>
        </div>
      </PageHeader>

      <div className="p-6 rounded-2xl bg-neutral-900 border border-neutral-800">
        <div className="text-lg sm:text-xl font-medium leading-relaxed bg-neutral-950/60 p-4 rounded-xl border border-neutral-800">“{item.clue}”</div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 mt-5">
          {item.options.map((option) => (
            <button
              key={option}
              disabled={answered}
              onClick={() => pick(option)}
              className={`p-3.5 rounded-xl border text-sm font-medium text-left flex items-center justify-between transition-colors ${optionStyle(option)}`}
            >
              {option}
              {answered && option === item.answer && <CheckCircle2 className="w-4 h-4 shrink-0" />}
              {answered && option === picked && option !== item.answer && <XCircle className="w-4 h-4 shrink-0" />}
            </button>
          ))}
        </div>
      </div>

      {answered && (
        <div className="mt-6 p-5 rounded-2xl bg-neutral-900 border border-neutral-800 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <span className={`text-sm font-bold ${picked === item.answer ? 'text-emerald-400' : 'text-rose-400'}`}>
            {picked === item.answer ? 'Correct!' : `Answer: ${item.answer}`}
          </span>
          <div className="flex items-center gap-2">
            <button onClick={() => nav.openNote(item.notePath)} className={btnSecondary}>
              <BookOpen className="w-3.5 h-3.5 text-amber-400" /> Open note
            </button>
            <button onClick={next} className={btnPrimary}>
              Next <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}
    </Page>
  );
}
