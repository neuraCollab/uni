import { useCallback, useEffect, useState } from 'react';
import type { Route, ViewMode } from './types';
import { noteByPath } from './data';

const VIEWS: ViewMode[] = ['browse', 'quiz', 'cram', 'flashcards', 'code', 'lookup'];
const DEFAULT_NOTE = 'algorithms/README.md';

function parseHash(): Route {
  const p = new URLSearchParams(window.location.hash.slice(1));
  const view = p.get('view') as ViewMode;
  const note = p.get('note') ?? '';
  return {
    view: VIEWS.includes(view) ? view : 'browse',
    note: noteByPath.has(note) ? note : DEFAULT_NOTE,
    code: p.get('code') ?? '',
    section: p.get('section') ?? '',
    filter: p.get('filter') ?? '',
  };
}

/** App route, persisted in `location.hash` so every view is linkable. */
export function useRoute() {
  const [route, setRoute] = useState(parseHash);

  useEffect(() => {
    const onHash = () => setRoute(parseHash());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const navigate = useCallback((patch: Partial<Route>) => {
    const next = { ...parseHash(), ...patch };
    // `note` is always kept so returning to Notes reopens the last note.
    const p = new URLSearchParams({ view: next.view, note: next.note });
    if (next.view === 'code' && next.code) p.set('code', next.code);
    if (next.view === 'cram' && next.section) p.set('section', next.section);
    if (next.view === 'flashcards' && next.filter) p.set('filter', next.filter);
    window.location.hash = p.toString();
  }, []);

  return [route, navigate] as const;
}

export function useCopy(): [string | null, (text: string, key?: string) => void] {
  const [copied, setCopied] = useState<string | null>(null);
  const copy = useCallback((text: string, key = text) => {
    navigator.clipboard.writeText(text);
    setCopied(key);
    setTimeout(() => setCopied(null), 2000);
  }, []);
  return [copied, copy];
}
