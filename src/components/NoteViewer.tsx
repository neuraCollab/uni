import { Children, isValidElement, useEffect, useState, type ReactElement, type ReactNode } from 'react';
import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { ArrowLeft, ArrowRight, Check, ChevronDown, ChevronUp, Code2, Copy, ExternalLink, HelpCircle, Info } from 'lucide-react';
import type { Note } from '../types';
import type { Nav } from '../App';
import { NOTES, codeByPath, humanize } from '../data';
import { resolveLink } from '../links';
import { useCopy } from '../hooks';

function textOf(node: ReactNode): string {
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(textOf).join('');
  if (isValidElement(node)) return textOf((node as ReactElement<{ children?: ReactNode }>).props.children);
  return '';
}

// GitHub-compatible heading ids, so `note.md#some-heading` links work.
const slug = (children: ReactNode) =>
  textOf(children)
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s_-]/gu, '')
    .replace(/ /g, '-');

function CopyButton({ text, label = 'Copy' }: { text: string; label?: string }) {
  const [copied, copy] = useCopy();
  return (
    <button onClick={() => copy(text)} className="flex items-center gap-1 px-2 py-0.5 rounded hover:bg-neutral-800 text-neutral-300 text-[11px]">
      {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
      {copied ? 'Copied' : label}
    </button>
  );
}

function CodeBlock({ language, code }: { language: string; code: string }) {
  return (
    <div className="my-5 rounded-xl overflow-hidden border border-neutral-800 bg-neutral-950">
      <div className="flex items-center justify-between px-4 py-1.5 bg-neutral-900 border-b border-neutral-800">
        <span className="font-mono text-[11px] uppercase tracking-wider text-amber-400/90">{language}</span>
        <CopyButton text={code} />
      </div>
      <pre className="p-4 overflow-x-auto text-xs sm:text-sm font-mono text-neutral-200 leading-relaxed">
        <code>{code}</code>
      </pre>
    </div>
  );
}

function markdownComponents(note: Note, nav: Nav): Components {
  const heading = (Tag: 'h1' | 'h2' | 'h3' | 'h4', className: string) =>
    function Heading({ children }: { children?: ReactNode }) {
      return (
        <Tag id={slug(children)} className={`scroll-mt-4 ${className}`}>
          {children}
        </Tag>
      );
    };

  return {
    h1: heading('h1', 'text-2xl sm:text-3xl font-extrabold text-neutral-100 mt-2 mb-6 pb-2 border-b border-neutral-800'),
    h2: heading('h2', 'text-xl sm:text-2xl font-bold text-neutral-100 mt-8 mb-4 pb-1 border-b border-neutral-800/60'),
    h3: heading('h3', 'text-lg font-semibold text-neutral-200 mt-6 mb-3'),
    h4: heading('h4', 'font-semibold text-neutral-200 mt-4 mb-2'),
    p: ({ children }) => <p className="my-3">{children}</p>,
    ul: ({ children }) => <ul className="my-3 space-y-1.5 list-disc pl-5">{children}</ul>,
    ol: ({ children }) => <ol className="my-3 space-y-1.5 list-decimal pl-5">{children}</ol>,
    blockquote: ({ children }) => <blockquote className="my-4 border-l-4 border-amber-500/60 bg-amber-500/5 px-4 py-1 rounded-r-lg">{children}</blockquote>,
    table: ({ children }) => (
      <div className="my-6 overflow-x-auto rounded-lg border border-neutral-800">
        <table className="w-full text-left border-collapse text-xs sm:text-sm">{children}</table>
      </div>
    ),
    thead: ({ children }) => <thead className="bg-neutral-800/80 text-neutral-200">{children}</thead>,
    th: ({ children }) => <th className="px-4 py-2.5 font-semibold">{children}</th>,
    td: ({ children }) => <td className="px-4 py-2.5 border-t border-neutral-800">{children}</td>,
    // Fenced blocks arrive as <pre><code class="language-x">; inline code only hits `code`.
    pre: ({ children }) => {
      const child = Children.only(children) as ReactElement<{ className?: string; children?: ReactNode }>;
      const language = /language-(\w+)/.exec(child.props.className ?? '')?.[1] ?? 'code';
      return <CodeBlock language={language} code={textOf(child.props.children).replace(/\n$/, '')} />;
    },
    code: ({ children }) => <code className="px-1.5 py-0.5 rounded bg-neutral-800 text-amber-300 font-mono text-[0.85em]">{children}</code>,
    a: ({ href = '', children }) => {
      const link = resolveLink(href, note.path);
      const cls = 'text-amber-400 hover:text-amber-300 underline underline-offset-2';
      switch (link.type) {
        case 'note':
          return (
            <button onClick={() => nav.openNote(link.target, link.anchor)} className={`${cls} text-left`}>
              {children}
            </button>
          );
        case 'code':
          return (
            <button onClick={() => nav.openCode(link.target)} className="inline-flex items-center gap-1 text-emerald-400 hover:text-emerald-300 underline underline-offset-2">
              <Code2 className="w-3.5 h-3.5" />
              {children}
            </button>
          );
        case 'anchor':
          return (
            <button onClick={() => document.getElementById(link.target)?.scrollIntoView({ behavior: 'smooth' })} className={`${cls} text-left`}>
              {children}
            </button>
          );
        default:
          return (
            <a href={link.target} target="_blank" rel="noopener noreferrer" className={cls}>
              {children}
              <ExternalLink className="w-3 h-3 inline ml-0.5" />
            </a>
          );
      }
    },
  };
}

function ReferencedCode({ paths, nav }: { paths: string[]; nav: Nav }) {
  const [open, setOpen] = useState<string | null>(null);
  const file = open ? codeByPath.get(open) : undefined;

  return (
    <div className="mb-6 p-4 rounded-xl bg-neutral-900 border border-neutral-800">
      <h3 className="flex items-center gap-2 mb-3 text-xs font-bold uppercase tracking-wider text-neutral-200">
        <Code2 className="w-4 h-4 text-emerald-400" />
        Referenced code ({paths.length})
      </h3>
      <div className="flex flex-wrap gap-2">
        {paths.map((p) => (
          <button
            key={p}
            onClick={() => setOpen(open === p ? null : p)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-mono border transition-colors ${
              open === p ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40' : 'bg-neutral-800 text-neutral-300 hover:text-white border-neutral-700'
            }`}
          >
            {codeByPath.get(p)!.filename}
            {open === p ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </button>
        ))}
      </div>
      {file && (
        <div className="mt-3 pt-3 border-t border-neutral-800">
          <div className="flex items-center justify-between pb-2 text-xs">
            <span className="font-mono text-emerald-400 truncate">{file.path}</span>
            <div className="flex items-center gap-2 shrink-0">
              <CopyButton text={file.content} />
              <button onClick={() => nav.openCode(file.path)} className="flex items-center gap-1 px-2 py-0.5 rounded hover:bg-neutral-800 text-neutral-300 text-[11px]">
                <ExternalLink className="w-3.5 h-3.5" />
                Open
              </button>
            </div>
          </div>
          <pre className="max-h-96 overflow-auto rounded-lg bg-neutral-950 p-3 font-mono text-xs text-neutral-200 border border-neutral-800">{file.content}</pre>
        </div>
      )}
    </div>
  );
}

function NavButton({ note, dir, nav }: { note?: Note; dir: 'prev' | 'next'; nav: Nav }) {
  if (!note) return <div />;
  const Icon = dir === 'prev' ? ArrowLeft : ArrowRight;
  return (
    <button
      onClick={() => nav.openNote(note.path)}
      className={`flex items-center gap-2 px-4 py-2 rounded-lg bg-neutral-900 hover:bg-neutral-800 border border-neutral-800 text-xs max-w-[45%] ${dir === 'next' ? 'flex-row-reverse text-right' : 'text-left'}`}
    >
      <Icon className="w-4 h-4 shrink-0 text-neutral-400" />
      <div className="min-w-0">
        <div className="text-[10px] text-neutral-400 uppercase tracking-wider">{dir === 'prev' ? 'Previous' : 'Next'}</div>
        <div className="font-medium truncate">{note.title}</div>
      </div>
    </button>
  );
}

export function NoteViewer({ note, anchor, nav }: { note: Note; anchor: { id?: string }; nav: Nav }) {
  useEffect(() => {
    if (anchor.id) document.getElementById(anchor.id)?.scrollIntoView();
  }, [anchor]);

  const siblings = NOTES.filter((n) => n.section === note.section);
  const i = siblings.indexOf(note);

  return (
    <div className="flex-1 overflow-y-auto p-4 sm:p-8">
      <div className="max-w-4xl mx-auto">
        <div className="flex flex-wrap items-center justify-between gap-3 pb-4 mb-6 border-b border-neutral-800 text-xs text-neutral-400">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="font-semibold text-amber-400 uppercase tracking-wider">{humanize(note.section)}</span>
            {note.subsection && <span>/ {humanize(note.subsection)}</span>}
            <span className="font-mono">/ {note.filename}</span>
          </div>
          {note.questionCount > 0 && (
            <button
              onClick={() => nav.openFlashcards(note.path)}
              className="flex items-center gap-1 px-2.5 py-1 rounded bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 border border-amber-500/30"
            >
              <HelpCircle className="w-3.5 h-3.5" />
              {note.questionCount} interview Qs
            </button>
          )}
        </div>

        {note.isScaffolding && (
          <div className="mb-6 p-3.5 rounded-lg bg-yellow-950/20 border border-yellow-800/40 text-yellow-300/90 text-xs flex items-start gap-2.5">
            <Info className="w-4 h-4 shrink-0" />
            Work in progress — this section is scaffolding that grows over time.
          </div>
        )}

        {note.codeRefs.length > 0 && <ReferencedCode paths={note.codeRefs} nav={nav} />}

        <article className="text-neutral-300 text-sm sm:text-base leading-relaxed">
          <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]} components={markdownComponents(note, nav)}>
            {note.content}
          </ReactMarkdown>
        </article>

        <div className="mt-12 pt-6 border-t border-neutral-800 flex items-center justify-between gap-4">
          <NavButton note={siblings[i - 1]} dir="prev" nav={nav} />
          <NavButton note={siblings[i + 1]} dir="next" nav={nav} />
        </div>
      </div>
    </div>
  );
}
