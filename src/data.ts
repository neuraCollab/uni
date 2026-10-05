import repo from './data/repo.json';
import type { RepoData } from './types';

const data = repo as RepoData;

export const SECTIONS = data.sections;
export const NOTES = data.notes;
export const CODE_FILES = data.code;
export const QUESTIONS = data.questions;
export const PATTERN_CLUES = data.patternClues;
export const LOOKUP = data.lookup;
export const CRAM_PLANS = data.cramPlans;

export const noteByPath = new Map(NOTES.map((n) => [n.path, n]));
export const codeByPath = new Map(CODE_FILES.map((c) => [c.path, c]));

export const sectionTitle = (id: string) => SECTIONS.find((s) => s.id === id)?.title ?? humanize(id);

export const humanize = (slug: string) => slug.replace(/[-_]/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

export function shuffle<T>(items: T[]): T[] {
  const a = [...items];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
