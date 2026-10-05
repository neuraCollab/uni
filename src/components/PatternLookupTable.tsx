import { useState } from 'react';
import { ArrowRight, Search, TableProperties } from 'lucide-react';
import type { Nav } from '../App';
import { LOOKUP } from '../data';
import { Page, PageHeader } from './ui';

export function PatternLookupTable({ nav }: { nav: Nav }) {
  const [query, setQuery] = useState('');
  const q = query.toLowerCase();
  const rows = LOOKUP.filter((r) => r.clue.toLowerCase().includes(q) || r.pattern.toLowerCase().includes(q));

  return (
    <Page width="max-w-5xl">
      <PageHeader icon={TableProperties} kicker="Algorithms" title="Pattern Lookup Table">
        <p className="text-sm text-neutral-400 mt-2">What the problem statement telegraphs → the technique that solves it.</p>
        <div className="mt-5 max-w-md relative">
          <Search className="w-4 h-4 absolute left-3 top-2.5 text-neutral-400" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter clues or patterns…"
            className="w-full pl-9 pr-4 py-2 rounded-xl bg-neutral-900 border border-neutral-800 text-sm placeholder-neutral-500 focus:outline-none focus:border-amber-500"
          />
        </div>
      </PageHeader>

      <div className="rounded-2xl border border-neutral-800 overflow-hidden bg-neutral-900">
        <table className="w-full text-left text-xs sm:text-sm">
          <thead className="bg-neutral-950/80 text-neutral-400 text-xs uppercase">
            <tr>
              <th className="px-5 py-3.5">If the problem says…</th>
              <th className="px-5 py-3.5">Think</th>
              <th />
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-800/60">
            {rows.map((r) => (
              <tr key={`${r.clue}|${r.pattern}`} onClick={() => r.notePath && nav.openNote(r.notePath)} className="hover:bg-neutral-800/60 cursor-pointer group">
                <td className="px-5 py-3.5 text-neutral-200">{r.clue}</td>
                <td className="px-5 py-3.5 text-amber-400 font-semibold whitespace-nowrap">{r.pattern}</td>
                <td className="px-4 py-3.5 text-neutral-500 group-hover:text-white">
                  <ArrowRight className="w-3.5 h-3.5" />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {rows.length === 0 && <div className="p-8 text-center text-neutral-400 text-xs">No matches.</div>}
      </div>
    </Page>
  );
}
