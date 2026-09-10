import React, { useState, useMemo } from 'react';
import { NoteItem, SectionMeta } from '../types';
import { 
  Flame, 
  CheckCircle2, 
  Circle, 
  ArrowRight, 
  ArrowLeft, 
  BookOpen, 
  Clock, 
  HelpCircle,
  Sparkles
} from 'lucide-react';

interface CramModeProps {
  sections: SectionMeta[];
  notes: NoteItem[];
  initialSectionId?: string;
  onNavigateNote: (notePath: string) => void;
  onOpenFlashcardsForNote: (notePath: string) => void;
}

interface CramStep {
  title: string;
  notePath: string;
  note?: NoteItem;
  description: string;
}

interface CramPlan {
  sectionId: string;
  title: string;
  description: string;
  estimatedMinutes: number;
  steps: CramStep[];
}

export const CramMode: React.FC<CramModeProps> = ({
  sections,
  notes,
  initialSectionId,
  onNavigateNote,
  onOpenFlashcardsForNote,
}) => {
  // Define curated high-signal sequential cram paths matching the repository's explicit README quick revision orders
  const cramPlans = useMemo<CramPlan[]>(() => {
    return [
      {
        sectionId: 'machine-learning',
        title: 'Machine Learning Cram Path',
        description: 'The proven order most DS/ML technical rounds probe: from regularization foundation to bias-variance, trees, and clustering.',
        estimatedMinutes: 18,
        steps: [
          {
            title: 'Regularization (OLS, Ridge, Lasso, ElasticNet)',
            notePath: 'machine-learning/linear-models/regularization.md',
            description: 'Why regularize, L1 vs L2 geometry, normal equation breakdown, feature scaling requirements.'
          },
          {
            title: 'Logistic Regression',
            notePath: 'machine-learning/linear-models/logistic-regression.md',
            description: 'Log-odds, sigmoid, cross-entropy loss, decision boundaries, probability calibration.'
          },
          {
            title: 'Classification Metrics',
            notePath: 'machine-learning/model-evaluation/classification-metrics.md',
            description: 'Precision, recall, F1, ROC-AUC vs PR-AUC, confusion matrix, threshold tuning.'
          },
          {
            title: 'Cross-Validation & Stratification',
            notePath: 'machine-learning/model-evaluation/cross-validation.md',
            description: 'K-fold, Stratified K-fold, GroupKFold, time-series splitting, leakage prevention.'
          },
          {
            title: 'Data Leakage',
            notePath: 'machine-learning/model-evaluation/data-leakage.md',
            description: 'Anchored on real leakage bugs: fit_transform mistakes, target leakage, validation contamination.'
          },
          {
            title: 'Bias-Variance Tradeoff',
            notePath: 'machine-learning/model-evaluation/bias-variance-tradeoff.md',
            description: 'Decomposition of expected test error, underfitting vs overfitting diagnosis, mitigation matrix.'
          },
          {
            title: 'Decision Trees & Random Forests',
            notePath: 'machine-learning/trees-ensembles/decision-trees.md',
            description: 'Gini vs entropy impurity, bagging, feature subsampling, out-of-bag error.'
          },
          {
            title: 'Gradient Boosting (CatBoost / LightGBM)',
            notePath: 'machine-learning/trees-ensembles/gradient-boosting-catboost-lgbm.md',
            description: 'Boosting residuals, histogram-based splitting, symmetric trees, categorical features.'
          },
          {
            title: 'Clustering Overview (k-means, CURE, FOREL)',
            notePath: 'machine-learning/clustering/overview.md',
            description: 'K-means inertia, silhouette score, non-spherical clusters, outlier handling.'
          },
          {
            title: 'Hyperparameter Optimization',
            notePath: 'machine-learning/hyperparameter-optimization.md',
            description: 'Grid vs random vs Bayesian/TPE, Optuna pruning, search space design.'
          }
        ]
      },
      {
        sectionId: 'algorithms',
        title: 'Core Algorithm Patterns Cram',
        description: 'The highest-frequency coding patterns: recognition clues, invariants, and templates.',
        estimatedMinutes: 20,
        steps: [
          {
            title: 'Two Pointers',
            notePath: 'algorithms/patterns/two-pointers.md',
            description: 'Opposite-direction, same-direction, sorted arrays, in-place partitions.'
          },
          {
            title: 'Sliding Window',
            notePath: 'algorithms/patterns/sliding-window.md',
            description: 'Contiguous subarrays/substrings, monotonic invariants, fixed vs dynamic size.'
          },
          {
            title: 'Binary Search & On the Answer',
            notePath: 'algorithms/patterns/binary-search.md',
            description: 'Boundary conditions, monotonic predicates, bisect_left vs bisect_right.'
          },
          {
            title: 'BFS & DFS',
            notePath: 'algorithms/patterns/bfs-dfs.md',
            description: 'Shortest path on unweighted graphs, level-order traversal, connected components, cycle detection.'
          },
          {
            title: 'Shortest Paths (Dijkstra, Bellman-Ford)',
            notePath: 'algorithms/patterns/shortest-paths.md',
            description: 'Weighted graphs, min-heaps, negative edge weights, Floyd-Warshall.'
          },
          {
            title: 'Dynamic Programming',
            notePath: 'algorithms/patterns/dynamic-programming.md',
            description: 'Overlapping subproblems, state design, 1D/2D transitions, space optimization.'
          },
          {
            title: 'Intervals & Scheduling',
            notePath: 'algorithms/patterns/intervals.md',
            description: 'Sorting by start vs end time, merging overlapping ranges, meeting rooms / concurrent peaks.'
          },
          {
            title: 'Monotonic Stack',
            notePath: 'algorithms/patterns/monotonic-stack.md',
            description: 'Next greater/smaller element, largest rectangle in histogram, trapping rain water.'
          },
          {
            title: 'Heaps & Priority Queues',
            notePath: 'algorithms/data-structures/heaps.md',
            description: 'Top-K elements, running median with dual heaps, merging K sorted streams.'
          }
        ]
      },
      {
        sectionId: 'python',
        title: 'Python Engineering Fundamentals',
        description: 'Memory model, concurrency, generators, decorators, and high-frequency interview traps.',
        estimatedMinutes: 15,
        steps: [
          {
            title: 'Memory Model & Mutability',
            notePath: 'python/memory-model-mutability.md',
            description: 'Pass-by-assignment, object identity vs equality, mutable default arguments, interned objects.'
          },
          {
            title: 'Object-Oriented Programming & Dunder Methods',
            notePath: 'python/oop.md',
            description: 'MRO and super(), __slots__, descriptors, property decorators, class vs static methods.'
          },
          {
            title: 'Iterators & Generators',
            notePath: 'python/iterators-generators.md',
            description: '__iter__ and __next__, yield vs return, generator expressions, send() and close().'
          },
          {
            title: 'Decorators & Closures',
            notePath: 'python/decorators.md',
            description: 'Lexical scoping, functools.wraps, parameterized decorators, class decorators.'
          },
          {
            title: 'Context Managers',
            notePath: 'python/context-managers.md',
            description: '__enter__ and __exit__, exception suppression, contextlib.contextmanager generator syntax.'
          },
          {
            title: 'Concurrency & Asyncio',
            notePath: 'python/concurrency-async.md',
            description: 'GIL implications, multithreading vs multiprocessing vs event loop asyncio.'
          },
          {
            title: 'Common Interview Traps',
            notePath: 'python/common-interview-traps.md',
            description: 'Late-binding closures in loops, is vs ==, list comprehension scope, modifying while iterating.'
          }
        ]
      },
      {
        sectionId: 'sql',
        title: 'SQL & Analytics Query Cram',
        description: 'Query execution order, complex window functions, CTEs, self-joins, and performance indexing.',
        estimatedMinutes: 14,
        steps: [
          {
            title: 'Query Execution Order',
            notePath: 'sql/query-execution-order.md',
            description: 'FROM → WHERE → GROUP BY → HAVING → SELECT → WINDOW → ORDER BY → LIMIT.'
          },
          {
            title: 'Joins & Null Handling',
            notePath: 'sql/joins.md',
            description: 'INNER vs LEFT vs FULL vs CROSS vs ANTI/SEMI joins, three-valued boolean logic.'
          },
          {
            title: 'Window Functions',
            notePath: 'sql/window-functions.md',
            description: 'ROW_NUMBER vs RANK vs DENSE_RANK, OVER (PARTITION BY ... ORDER BY ...), LAG/LEAD, running sums.'
          },
          {
            title: 'CTEs & Subqueries',
            notePath: 'sql/ctes-subqueries.md',
            description: 'Recursive CTEs for hierarchy/graphs, correlated subqueries vs JOINs, materialization.'
          },
          {
            title: 'Indexes & Query Optimization',
            notePath: 'sql/indexes-optimization.md',
            description: 'B-Tree vs Hash indexes, composite index column order, sargable predicates, EXPLAIN plans.'
          },
          {
            title: 'Classic Interview Problems',
            notePath: 'sql/interview-problems.md',
            description: 'Consecutive active days, retention cohorts, second highest salary, manager hierarchy.'
          }
        ]
      },
      {
        sectionId: 'statistics',
        title: 'Statistics & A/B Testing Cram',
        description: 'Probability foundations, hypothesis testing, p-values, power analysis, and experiment design.',
        estimatedMinutes: 14,
        steps: [
          {
            title: 'Probability & Bayes Theorem',
            notePath: 'statistics/probability-bayes.md',
            description: 'Conditional probability, prior vs posterior, false positive paradox, independence.'
          },
          {
            title: 'Distributions & Central Limit Theorem',
            notePath: 'statistics/distributions-clt.md',
            description: 'Normal, Binomial, Poisson, Exponential, when CLT applies, skewness and fat tails.'
          },
          {
            title: 'Hypothesis Testing & P-Values',
            notePath: 'statistics/hypothesis-testing-pvalue.md',
            description: 'Null vs alternative, Type I (alpha) vs Type II (beta) errors, statistical power, p-value misconceptions.'
          },
          {
            title: 'Confidence Intervals',
            notePath: 'statistics/confidence-intervals.md',
            description: 'Interpretation (frequentist coverage), margin of error, bootstrap confidence intervals.'
          },
          {
            title: 'T-Test, ANOVA, Chi-Square',
            notePath: 'statistics/t-test-anova-chi-square.md',
            description: 'Continuous means vs categorical frequencies, one-sample vs two-sample vs paired, assumptions.'
          },
          {
            title: 'A/B Testing & Experimentation',
            notePath: 'statistics/ab-testing.md',
            description: 'Sample size calculation, minimum detectable effect (MDE), network effects, peeking problem.'
          }
        ]
      },
      {
        sectionId: 'deep-learning',
        title: 'Deep Learning & Neural Architectures',
        description: 'Backprop, PyTorch tensors/autograd, optimization, CNNs, Transformers, and VAEs.',
        estimatedMinutes: 16,
        steps: [
          {
            title: 'Neural Networks & Backpropagation',
            notePath: 'deep-learning/fundamentals/neural-networks-backprop.md',
            description: 'Forward pass, computational graphs, chain rule, vanishing/exploding gradients.'
          },
          {
            title: 'Optimization: SGD, Momentum, Adam',
            notePath: 'deep-learning/fundamentals/optimization-sgd-adam.md',
            description: 'First vs second moments, learning rate warmup/decay, weight decay vs L2 penalty.'
          },
          {
            title: 'PyTorch Tensors & Autograd',
            notePath: 'deep-learning/pytorch/tensors-autograd.md',
            description: 'Tensor operations, requires_grad, backward(), no_grad context, memory management.'
          },
          {
            title: 'Regularization in Neural Nets',
            notePath: 'deep-learning/regularization-overfitting.md',
            description: 'Dropout (train vs eval), Batch Normalization vs Layer Normalization, early stopping.'
          },
          {
            title: 'Attention & Transformers',
            notePath: 'deep-learning/attention-transformers.md',
            description: 'Scaled dot-product attention, queries/keys/values, multi-head attention, positional encoding.'
          },
          {
            title: 'Variational Autoencoders (VAEs)',
            notePath: 'deep-learning/pytorch/vae.md',
            description: 'Latent space, reconstruction loss + KL divergence, reparameterization trick.'
          }
        ]
      }
    ];
  }, []);

  const [activePlanId, setActivePlanId] = useState<string>(
    initialSectionId && cramPlans.some(p => p.sectionId === initialSectionId)
      ? initialSectionId
      : 'machine-learning'
  );

  const [currentStepIndex, setCurrentStepIndex] = useState<number>(0);
  const [completedSteps, setCompletedSteps] = useState<Record<string, boolean>>({});

  const activePlan = cramPlans.find(p => p.sectionId === activePlanId) || cramPlans[0];
  const currentStep = activePlan.steps[currentStepIndex] || activePlan.steps[0];

  // Resolve note object
  const resolvedNote = notes.find(n => n.path === currentStep.notePath || n.id === currentStep.notePath.replace('.md', ''));

  const isStepCompleted = !!completedSteps[`${activePlan.sectionId}-${currentStepIndex}`];

  const toggleComplete = (idx: number) => {
    const key = `${activePlan.sectionId}-${idx}`;
    setCompletedSteps(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const handleNextStep = () => {
    // Mark current as completed
    const key = `${activePlan.sectionId}-${currentStepIndex}`;
    setCompletedSteps(prev => ({ ...prev, [key]: true }));

    if (currentStepIndex < activePlan.steps.length - 1) {
      setCurrentStepIndex(prev => prev + 1);
    }
  };

  const handlePrevStep = () => {
    if (currentStepIndex > 0) {
      setCurrentStepIndex(prev => prev - 1);
    }
  };

  const completedCount = activePlan.steps.filter((_, idx) => !!completedSteps[`${activePlan.sectionId}-${idx}`]).length;
  const progressPercent = Math.round((completedCount / activePlan.steps.length) * 100);

  return (
    <div className="flex-1 overflow-y-auto p-4 sm:p-8 max-w-4xl mx-auto w-full">
      {/* Top Banner */}
      <div className="mb-8 pb-6 border-b border-neutral-800">
        <div className="flex items-center gap-2 text-amber-500 mb-2">
          <Flame className="w-5 h-5 fill-amber-500/20" />
          <span className="text-xs uppercase font-bold tracking-wider">Guided Sequential Cram Mode</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-neutral-100 tracking-tight">
          Morning-Of Interview Quick Revision
        </h1>
        <p className="text-sm text-neutral-400 mt-2 max-w-2xl">
          Based on the repository's explicit "Quick revision order" lists: a structured, high-signal path covering the core intuition, trade-offs, and questions without getting lost in tangents.
        </p>

        {/* Plan Selector Pills */}
        <div className="flex items-center gap-2 overflow-x-auto pt-5 pb-1 scrollbar-none">
          {cramPlans.map(plan => {
            const isCurrent = plan.sectionId === activePlanId;
            return (
              <button
                key={plan.sectionId}
                onClick={() => {
                  setActivePlanId(plan.sectionId);
                  setCurrentStepIndex(0);
                }}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-all border ${
                  isCurrent
                    ? 'bg-amber-500/20 text-amber-300 border-amber-500/50 shadow-sm'
                    : 'bg-neutral-900 text-neutral-400 border-neutral-800 hover:text-neutral-200 hover:bg-neutral-800'
                }`}
              >
                <span>{plan.title.replace(' Cram Path', '').replace(' Cram', '')}</span>
                <span className="text-[10px] px-1 py-0.2 rounded bg-neutral-800 font-mono text-neutral-400">
                  {plan.steps.length} steps
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Progress & Plan Overview */}
      <div className="p-4 rounded-xl bg-neutral-900 border border-neutral-800 mb-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-base font-bold text-neutral-100">{activePlan.title}</h2>
            <span className="text-xs text-amber-400 font-mono flex items-center gap-1">
              <Clock className="w-3.5 h-3.5" />
              <span>~{activePlan.estimatedMinutes} mins</span>
            </span>
          </div>
          <p className="text-xs text-neutral-400 mt-1">{activePlan.description}</p>
        </div>

        {/* Progress Bar */}
        <div className="w-full sm:w-48 shrink-0">
          <div className="flex items-center justify-between text-xs mb-1 font-mono">
            <span className="text-neutral-400">Progress</span>
            <span className="text-amber-400 font-bold">{completedCount}/{activePlan.steps.length} ({progressPercent}%)</span>
          </div>
          <div className="w-full h-2 bg-neutral-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-amber-500 to-orange-500 transition-all duration-300 rounded-full"
              style={{ width: `${progressPercent}%` }}
            ></div>
          </div>
        </div>
      </div>

      {/* Active Step Card */}
      <div className="p-6 rounded-2xl bg-neutral-900 border border-neutral-800 shadow-xl space-y-6">
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-xs text-amber-400 font-semibold uppercase tracking-wider">
              <span>Step {currentStepIndex + 1} of {activePlan.steps.length}</span>
              <span>•</span>
              <span className="text-neutral-400">~2 min read</span>
            </div>
            <h3 className="text-xl sm:text-2xl font-bold text-neutral-100">
              {currentStep.title}
            </h3>
          </div>

          <button
            onClick={() => toggleComplete(currentStepIndex)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
              isStepCompleted
                ? 'bg-emerald-950/40 text-emerald-300 border-emerald-500/40'
                : 'bg-neutral-800 text-neutral-400 border-neutral-700 hover:text-neutral-200'
            }`}
          >
            {isStepCompleted ? (
              <>
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span>Completed</span>
              </>
            ) : (
              <>
                <Circle className="w-4 h-4" />
                <span>Mark as read</span>
              </>
            )}
          </button>
        </div>

        {/* Step Core Takeaway & Excerpt */}
        <div className="p-4 rounded-xl bg-neutral-950/70 border border-neutral-800/80 space-y-3">
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-amber-400">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Key Focus & Concepts</span>
          </div>
          <p className="text-sm text-neutral-200 leading-relaxed font-medium">
            {currentStep.description}
          </p>
          {resolvedNote && resolvedNote.excerpt && (
            <p className="text-xs text-neutral-400 leading-relaxed pt-2 border-t border-neutral-850">
              "{resolvedNote.excerpt}..."
            </p>
          )}
        </div>

        {/* Actions for this Step */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
          <div className="flex items-center gap-2">
            <button
              onClick={() => onNavigateNote(currentStep.notePath)}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-amber-500 hover:bg-amber-400 text-neutral-950 font-bold text-xs transition-colors shadow-sm"
            >
              <BookOpen className="w-4 h-4" />
              <span>Read Full Note</span>
            </button>

            {resolvedNote && resolvedNote.interviewQuestionsCount > 0 && (
              <button
                onClick={() => onOpenFlashcardsForNote(currentStep.notePath)}
                className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-neutral-200 text-xs font-medium border border-neutral-700 transition-colors"
              >
                <HelpCircle className="w-3.5 h-3.5 text-amber-400" />
                <span>Test {resolvedNote.interviewQuestionsCount} Qs</span>
              </button>
            )}
          </div>

          <div className="flex items-center gap-2">
            <button
              disabled={currentStepIndex === 0}
              onClick={handlePrevStep}
              className="flex items-center gap-1 px-3 py-2 rounded-lg bg-neutral-800 hover:bg-neutral-700 disabled:opacity-40 disabled:cursor-not-allowed text-neutral-300 text-xs font-medium transition-colors"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Prev</span>
            </button>
            <button
              onClick={handleNextStep}
              className="flex items-center gap-1 px-4 py-2 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-amber-400 hover:text-amber-300 text-xs font-semibold border border-neutral-700 transition-colors"
            >
              <span>{currentStepIndex === activePlan.steps.length - 1 ? 'Finish Path' : 'Next Step'}</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Full Steps Roadmap Overview */}
      <div className="mt-8 space-y-2">
        <h4 className="text-xs font-bold uppercase tracking-wider text-neutral-400 px-1">
          Complete Revision Path Steps ({activePlan.steps.length})
        </h4>
        <div className="space-y-1.5">
          {activePlan.steps.map((step, idx) => {
            const isCurrent = idx === currentStepIndex;
            const isCompleted = !!completedSteps[`${activePlan.sectionId}-${idx}`];

            return (
              <button
                key={idx}
                onClick={() => setCurrentStepIndex(idx)}
                className={`w-full text-left p-3 rounded-xl border text-xs transition-all flex items-center justify-between gap-3 ${
                  isCurrent
                    ? 'bg-amber-500/15 text-neutral-100 border-amber-500/40 shadow-sm'
                    : 'bg-neutral-900/60 text-neutral-400 border-neutral-800/80 hover:bg-neutral-850 hover:text-neutral-200'
                }`}
              >
                <div className="flex items-center gap-3 truncate">
                  <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-mono shrink-0 ${
                    isCompleted
                      ? 'bg-emerald-500/20 text-emerald-400 font-bold'
                      : isCurrent
                      ? 'bg-amber-500 text-neutral-950 font-bold'
                      : 'bg-neutral-800 text-neutral-400'
                  }`}>
                    {isCompleted ? '✓' : idx + 1}
                  </span>
                  <div className="truncate">
                    <span className={`font-semibold ${isCurrent ? 'text-amber-300' : 'text-neutral-200'}`}>
                      {step.title}
                    </span>
                    <span className="text-neutral-400 ml-2 hidden sm:inline truncate text-[11px]">
                      {step.description}
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  {isCompleted && (
                    <span className="text-[10px] text-emerald-400 uppercase font-semibold">Done</span>
                  )}
                  {isCurrent && (
                    <span className="text-[10px] text-amber-400 font-semibold uppercase">Active</span>
                  )}
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};
