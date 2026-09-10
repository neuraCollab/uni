import React, { useState, useMemo } from 'react';
import { ALGORITHMS_LOOKUP_TABLE, NoteItem } from '../data/repoData';
import { TableProperties, Search, BookOpen, ArrowRight } from 'lucide-react';

interface PatternLookupTableProps {
  notes: NoteItem[];
  onNavigateNote: (notePath: string) => void;
}

export const PatternLookupTable: React.FC<PatternLookupTableProps> = ({
  notes,
  onNavigateNote,
}) => {
  const [filterQuery, setFilterQuery] = useState('');

  const filteredRows = useMemo(() => {
    return ALGORITHMS_LOOKUP_TABLE.filter(item => {
      const q = filterQuery.toLowerCase();
      return item.clue.toLowerCase().includes(q) || item.pattern.toLowerCase().includes(q);
    });
  }, [filterQuery]);

  const handleRowClick = (item: typeof ALGORITHMS_LOOKUP_TABLE[0]) => {
    if (item.targetPath) {
      onNavigateNote(item.targetPath);
    } else {
      const found = notes.find(n => n.title.toLowerCase() === item.pattern.toLowerCase() && n.section === 'algorithms');
      if (found) {
        onNavigateNote(found.path);
      }
    }
  };

  return (
    <div className="flex-1 overflow-y-auto p-4 sm:p-8 max-w-5xl mx-auto w-full">
      {/* Header */}
      <div className="mb-6 pb-6 border-b border-neutral-800">
        <div className="flex items-center gap-2 text-amber-500 mb-2">
          <TableProperties className="w-5 h-5" />
          <span className="text-xs uppercase font-bold tracking-wider">Cheat Sheet & Decision Matrix</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-neutral-100 tracking-tight">
          Algorithmic Pattern Lookup Table
        </h1>
        <p className="text-sm text-neutral-400 mt-2 max-w-2xl">
          Match what the interview problem statement is telegraphing to the exact algorithmic technique that solves it.
        </p>

        {/* Filter input */}
        <div className="mt-5 max-w-md relative">
          <Search className="w-4 h-4 absolute left-3 top-3 text-neutral-400" />
          <input
            type="text"
            placeholder="Search clues or patterns (e.g., 'contiguous', 'subarray', 'binary search')..."
            value={filterQuery}
            onChange={(e) => setFilterQuery(e.target.value)}
            className="w-full pl-9 pr-4 py-2 rounded-xl bg-neutral-900 border border-neutral-800 text-sm text-neutral-200 placeholder-neutral-400 focus:outline-none focus:border-amber-500"
          />
        </div>
      </div>

      {/* Lookup Table */}
      <div className="rounded-2xl border border-neutral-800 overflow-hidden bg-neutral-900 shadow-xl">
        <table className="w-full text-left border-collapse text-xs sm:text-sm">
          <thead>
            <tr className="bg-neutral-950/80 border-b border-neutral-800 text-neutral-400 text-xs uppercase font-semibold">
              <th className="px-5 py-3.5 w-3/5">If the problem statement says / implies...</th>
              <th className="px-5 py-3.5 w-2/5">Think Pattern...</th>
              <th className="px-4 py-3.5 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-800/60">
            {filteredRows.map((row, idx) => (
              <tr
                key={idx}
                onClick={() => handleRowClick(row)}
                className="hover:bg-neutral-800/60 transition-colors cursor-pointer group"
              >
                <td className="px-5 py-3.5 text-neutral-200 font-medium">
                  {row.clue}
                </td>
                <td className="px-5 py-3.5">
                  <span className="inline-flex items-center gap-1 text-amber-400 font-semibold group-hover:text-amber-300">
                    {row.pattern}
                  </span>
                </td>
                <td className="px-4 py-3.5 text-right">
                  <span className="inline-flex items-center gap-1 text-xs text-neutral-400 group-hover:text-white transition-colors">
                    <span>Note</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {filteredRows.length === 0 && (
          <div className="p-8 text-center text-neutral-400 text-xs">
            No patterns match your search query.
          </div>
        )}
      </div>
    </div>
  );
};
