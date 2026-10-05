import type { ReactNode } from 'react';
import type { LucideIcon } from 'lucide-react';

export function Page({ width = 'max-w-3xl', children }: { width?: string; children: ReactNode }) {
  return <div className={`flex-1 overflow-y-auto p-4 sm:p-8 ${width} mx-auto w-full`}>{children}</div>;
}

export function PageHeader({ icon: Icon, kicker, title, children }: { icon: LucideIcon; kicker: string; title: string; children?: ReactNode }) {
  return (
    <div className="mb-6 pb-6 border-b border-neutral-800">
      <div className="flex items-center gap-2 text-amber-400 mb-2">
        <Icon className="w-5 h-5" />
        <span className="text-xs uppercase font-bold tracking-wider">{kicker}</span>
      </div>
      <h1 className="text-2xl sm:text-3xl font-extrabold text-neutral-100 tracking-tight">{title}</h1>
      {children}
    </div>
  );
}

export function Pill({ active, onClick, children }: { active: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-colors border ${
        active ? 'bg-amber-500/20 text-amber-300 border-amber-500/50' : 'bg-neutral-900 text-neutral-400 border-neutral-800 hover:text-white'
      }`}
    >
      {children}
    </button>
  );
}

export function PillRow({ children }: { children: ReactNode }) {
  return <div className="flex items-center gap-1.5 overflow-x-auto pt-4 pb-1">{children}</div>;
}

export const btnPrimary =
  'flex items-center justify-center gap-1.5 px-4 py-2 rounded-lg bg-amber-500 hover:bg-amber-400 text-neutral-950 font-bold text-xs transition-colors';
export const btnSecondary =
  'flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-neutral-200 text-xs font-medium border border-neutral-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed';
