export type ViewMode = 'browse' | 'quiz' | 'cram' | 'flashcards' | 'code' | 'lookup';

export interface Route {
  view: ViewMode;
  note: string;
  code: string;
  /** Cram mode: selected section. */
  section: string;
  /** Flashcards: restrict to questions from one note. */
  filter: string;
}

export interface Section {
  id: string;
  title: string;
  desc: string;
}

export interface Note {
  path: string;
  title: string;
  section: string;
  subsection: string;
  filename: string;
  isReadme: boolean;
  isScaffolding: boolean;
  excerpt: string;
  content: string;
  codeRefs: string[];
  questionCount: number;
}

export interface CodeFile {
  path: string;
  filename: string;
  section: string;
  content: string;
}

export interface Question {
  id: string;
  question: string;
  hint: string;
  notePath: string;
  noteTitle: string;
  section: string;
}

export interface PatternClues {
  pattern: string;
  notePath: string;
  clues: string[];
}

export interface LookupRow {
  clue: string;
  pattern: string;
  notePath: string;
}

export interface CramPlan {
  section: string;
  steps: string[];
}

export interface RepoData {
  sections: Section[];
  notes: Note[];
  code: CodeFile[];
  questions: Question[];
  patternClues: PatternClues[];
  lookup: LookupRow[];
  cramPlans: CramPlan[];
}
