import { useState } from 'react';
import { ArrowLeft, ArrowRight, BookOpen, CheckCircle2, Circle, Flame, HelpCircle } from 'lucide-react';
import type { Nav } from '../App';
import { CRAM_PLANS, noteByPath, sectionTitle } from '../data';
import { Page, PageHeader, Pill, PillRow, btnPrimary, btnSecondary } from './ui';

export function CramMode({ initialSection, nav }: { initialSection: string; nav: Nav }) {
  const [planId, setPlanId] = useState(CRAM_PLANS.some((p) => p.section === initialSection) ? initialSection : CRAM_PLANS[0].section);
  const [step, setStep] = useState(0);
  const [done, setDone] = useState<Set<string>>(new Set());

  const plan = CRAM_PLANS.find((p) => p.section === planId)!;
  const steps = plan.steps.map((path) => noteByPath.get(path)!);
  const note = steps[step];
  const doneCount = plan.steps.filter((p) => done.has(p)).length;

  const toggleDone = (path: string, value = !done.has(path)) =>
    setDone((prev) => {
      const next = new Set(prev);
      value ? next.add(path) : next.delete(path);
      return next;
    });

  const selectPlan = (id: string) => {
    setPlanId(id);
    setStep(0);
  };

  const nextStep = () => {
    toggleDone(note.path, true);
    setStep((s) => Math.min(s + 1, steps.length - 1));
  };

  return (
    <Page width="max-w-4xl">
      <PageHeader icon={Flame} kicker="Cram mode" title="Quick revision path">
        <p className="text-sm text-neutral-400 mt-2">Follows the “Suggested review order” from each section’s README.</p>
        <PillRow>
          {CRAM_PLANS.map((p) => (
            <Pill key={p.section} active={p.section === planId} onClick={() => selectPlan(p.section)}>
              {sectionTitle(p.section)} · {p.steps.length}
            </Pill>
          ))}
        </PillRow>
      </PageHeader>

      <div className="mb-6">
        <div className="flex justify-between text-xs font-mono mb-1 text-neutral-400">
          <span>Progress</span>
          <span className="text-amber-400">
            {doneCount}/{steps.length}
          </span>
        </div>
        <div className="h-2 bg-neutral-800 rounded-full overflow-hidden">
          <div className="h-full bg-amber-500 transition-all" style={{ width: `${(doneCount / steps.length) * 100}%` }} />
        </div>
      </div>

      <div className="p-6 rounded-2xl bg-neutral-900 border border-neutral-800 space-y-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="text-xs text-amber-400 font-semibold uppercase tracking-wider">
              Step {step + 1} of {steps.length}
            </div>
            <h2 className="text-xl sm:text-2xl font-bold mt-1">{note.title}</h2>
          </div>
          <button onClick={() => toggleDone(note.path)} className={btnSecondary}>
            {done.has(note.path) ? <CheckCircle2 className="w-4 h-4 text-emerald-400" /> : <Circle className="w-4 h-4" />}
            {done.has(note.path) ? 'Done' : 'Mark as read'}
          </button>
        </div>

        {note.excerpt && <p className="text-sm text-neutral-300 leading-relaxed p-4 rounded-xl bg-neutral-950/70 border border-neutral-800">{note.excerpt}</p>}

        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <button onClick={() => nav.openNote(note.path)} className={btnPrimary}>
              <BookOpen className="w-4 h-4" /> Read note
            </button>
            {note.questionCount > 0 && (
              <button onClick={() => nav.openFlashcards(note.path)} className={btnSecondary}>
                <HelpCircle className="w-3.5 h-3.5 text-amber-400" /> {note.questionCount} Qs
              </button>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button disabled={step === 0} onClick={() => setStep(step - 1)} className={btnSecondary}>
              <ArrowLeft className="w-3.5 h-3.5" /> Prev
            </button>
            <button onClick={nextStep} className={btnSecondary}>
              {step === steps.length - 1 ? 'Finish' : 'Next'} <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      <ol className="mt-8 space-y-1.5">
        {steps.map((n, i) => (
          <li key={n.path}>
            <button
              onClick={() => setStep(i)}
              className={`w-full text-left p-3 rounded-xl border text-xs flex items-center gap-3 ${
                i === step ? 'bg-amber-500/15 border-amber-500/40' : 'bg-neutral-900/60 text-neutral-400 border-neutral-800 hover:text-neutral-200'
              }`}
            >
              <span
                className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-mono shrink-0 ${
                  done.has(n.path) ? 'bg-emerald-500/20 text-emerald-400' : i === step ? 'bg-amber-500 text-neutral-950' : 'bg-neutral-800'
                }`}
              >
                {done.has(n.path) ? '✓' : i + 1}
              </span>
              <span className="font-semibold truncate">{n.title}</span>
            </button>
          </li>
        ))}
      </ol>
    </Page>
  );
}
