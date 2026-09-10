export type ViewMode = 'browse' | 'quiz' | 'cram' | 'flashcards' | 'code' | 'lookup';

export interface SectionMeta {
  id: string;
  title: string;
  icon: string;
  desc: string;
}

export interface NoteItem {
  id: string;
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
  quickRevisionOrder?: string;
  interviewQuestionsCount: number;
}

export interface CodeItem {
  id: string;
  path: string;
  filename: string;
  section: string;
  subsection: string;
  content: string;
  lines: number;
}

export interface InterviewQuestion {
  id: string;
  question: string;
  hint: string;
  notePath: string;
  noteTitle: string;
  section: string;
}

export interface PatternClueItem {
  pattern: string;
  notePath: string;
  clues: string[];
}

export interface LookupItem {
  clue: string;
  pattern: string;
  targetPath: string;
}
