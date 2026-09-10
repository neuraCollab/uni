import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { NoteItem, CodeItem } from '../types';
import { resolveMarkdownLink } from '../utils/linkResolver';
import { 
  Copy, 
  Check, 
  Code2, 
  HelpCircle, 
  ArrowLeft, 
  ArrowRight, 
  FileText, 
  ExternalLink,
  Info,
  ChevronDown,
  ChevronUp
} from 'lucide-react';

interface NoteViewerProps {
  note: NoteItem;
  allNotes: NoteItem[];
  allCodeFiles: CodeItem[];
  onNavigateNote: (path: string) => void;
  onNavigateCode: (path: string) => void;
  onOpenFlashcardsForNote: (notePath: string) => void;
}

export const NoteViewer: React.FC<NoteViewerProps> = ({
  note,
  allNotes,
  allCodeFiles,
  onNavigateNote,
  onNavigateCode,
  onOpenFlashcardsForNote,
}) => {
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [activeCodePreview, setActiveCodePreview] = useState<string | null>(null);

  const handleCopy = (text: string, index: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  // Find prev and next note in the same section
  const sectionNotes = allNotes.filter(n => n.section === note.section);
  const currentIndex = sectionNotes.findIndex(n => n.path === note.path);
  const prevNote = currentIndex > 0 ? sectionNotes[currentIndex - 1] : null;
  const nextNote = currentIndex < sectionNotes.length - 1 ? sectionNotes[currentIndex + 1] : null;

  // Resolve referenced code files
  const referencedCodes = (note.codeRefs || [])
    .map(refPath => allCodeFiles.find(c => c.path === refPath || refPath.endsWith(c.path) || c.path.endsWith(refPath)))
    .filter((c): c is CodeItem => c !== undefined);

  return (
    <div className="flex-1 overflow-y-auto p-4 sm:p-8 max-w-4xl mx-auto w-full">
      {/* Top Breadcrumb & Metadata */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-4 mb-6 border-b border-neutral-800 text-xs text-neutral-400">
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="font-semibold text-amber-400 uppercase tracking-wider">
            {note.section.replace(/-/g, ' ')}
          </span>
          {note.subsection && (
            <>
              <span className="text-neutral-400">/</span>
              <span className="text-neutral-300">{note.subsection.replace(/-/g, ' ')}</span>
            </>
          )}
          <span className="text-neutral-400">/</span>
          <span className="text-neutral-400 font-mono text-[11px]">{note.filename}</span>
        </div>

        <div className="flex items-center gap-2">
          {note.interviewQuestionsCount > 0 && (
            <button
              onClick={() => onOpenFlashcardsForNote(note.path)}
              className="flex items-center gap-1 px-2.5 py-1 rounded bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 border border-amber-500/30 transition-colors"
            >
              <HelpCircle className="w-3.5 h-3.5" />
              <span>{note.interviewQuestionsCount} Interview Qs</span>
            </button>
          )}
        </div>
      </div>

      {/* Scaffolding Notice if applicable */}
      {note.isScaffolding && (
        <div className="mb-6 p-3.5 rounded-lg bg-yellow-950/20 border border-yellow-800/40 text-yellow-300/90 text-xs flex items-start gap-2.5">
          <Info className="w-4 h-4 text-yellow-400 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold">Repository Scaffolding / In Progress:</span>
            <p className="mt-0.5 text-yellow-400/80">
              This note is intentionally designated in the repository as scaffolding or growing. The core template and structure are preserved below.
            </p>
          </div>
        </div>
      )}

      {/* Referenced Code Callout if available */}
      {referencedCodes.length > 0 && (
        <div className="mb-6 p-4 rounded-xl bg-neutral-900 border border-neutral-800 shadow-sm">
          <div className="flex items-center justify-between gap-2 mb-2">
            <div className="flex items-center gap-2">
              <Code2 className="w-4 h-4 text-emerald-400" />
              <h3 className="text-xs font-bold uppercase tracking-wider text-neutral-200">
                Referenced Python Implementations ({referencedCodes.length})
              </h3>
            </div>
            <span className="text-[11px] text-neutral-400">Click to view source code</span>
          </div>

          <div className="flex flex-wrap gap-2 pt-1">
            {referencedCodes.map(codeFile => (
              <button
                key={codeFile.path}
                onClick={() => setActiveCodePreview(activeCodePreview === codeFile.path ? null : codeFile.path)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-mono transition-colors border ${
                  activeCodePreview === codeFile.path
                    ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                    : 'bg-neutral-800 text-neutral-300 hover:text-white border-neutral-700 hover:bg-neutral-750'
                }`}
              >
                <span>{codeFile.filename}</span>
                {activeCodePreview === codeFile.path ? (
                  <ChevronUp className="w-3.5 h-3.5" />
                ) : (
                  <ChevronDown className="w-3.5 h-3.5" />
                )}
              </button>
            ))}
          </div>

          {/* Collapsible Inline Code Viewer */}
          {activeCodePreview && (() => {
            const currentCode = referencedCodes.find(c => c.path === activeCodePreview);
            if (!currentCode) return null;
            return (
              <div className="mt-3 pt-3 border-t border-neutral-800">
                <div className="flex items-center justify-between pb-2 text-xs text-neutral-400">
                  <span className="font-mono text-emerald-400">{currentCode.path}</span>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => handleCopy(currentCode.content, 9999)}
                      className="flex items-center gap-1 px-2 py-1 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 transition-colors"
                    >
                      {copiedIndex === 9999 ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                      <span>Copy Code</span>
                    </button>
                    <button
                      onClick={() => onNavigateCode(currentCode.path)}
                      className="flex items-center gap-1 px-2 py-1 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 transition-colors"
                    >
                      <ExternalLink className="w-3 h-3" />
                      <span>Full Vault</span>
                    </button>
                  </div>
                </div>
                <div className="max-h-96 overflow-y-auto rounded-lg bg-neutral-950 p-3 font-mono text-xs text-neutral-200 border border-neutral-800">
                  <pre className="whitespace-pre overflow-x-auto">{currentCode.content}</pre>
                </div>
              </div>
            );
          })()}
        </div>
      )}

      {/* Main Markdown Content */}
      <article className="prose prose-invert max-w-none prose-headings:font-bold prose-headings:tracking-tight prose-a:text-amber-400 hover:prose-a:text-amber-300 prose-code:text-amber-300 prose-pre:bg-neutral-950 prose-pre:border prose-pre:border-neutral-800 text-neutral-200 text-sm sm:text-base leading-relaxed">
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[rehypeKatex]}
          components={{
            h1: ({ children }) => (
              <h1 className="text-2xl sm:text-3xl font-extrabold text-neutral-100 tracking-tight mt-2 mb-6 pb-2 border-b border-neutral-800">
                {children}
              </h1>
            ),
            h2: ({ children }) => (
              <h2 className="text-xl sm:text-2xl font-bold text-neutral-100 mt-8 mb-4 pb-1 border-b border-neutral-800/60 flex items-center gap-2">
                <span className="w-1.5 h-5 bg-amber-500 rounded-sm"></span>
                <span>{children}</span>
              </h2>
            ),
            h3: ({ children }) => (
              <h3 className="text-lg font-semibold text-neutral-200 mt-6 mb-3">
                {children}
              </h3>
            ),
            p: ({ children }) => (
              <p className="my-3 text-neutral-300 leading-relaxed">
                {children}
              </p>
            ),
            ul: ({ children }) => (
              <ul className="my-3 space-y-1.5 list-disc pl-5 text-neutral-300">
                {children}
              </ul>
            ),
            ol: ({ children }) => (
              <ol className="my-3 space-y-1.5 list-decimal pl-5 text-neutral-300">
                {children}
              </ol>
            ),
            li: ({ children }) => (
              <li className="text-neutral-300 leading-relaxed">
                {children}
              </li>
            ),
            blockquote: ({ children }) => (
              <blockquote className="my-4 border-l-4 border-amber-500/60 bg-amber-500/5 px-4 py-2 rounded-r-lg text-neutral-300 italic text-sm">
                {children}
              </blockquote>
            ),
            table: ({ children }) => (
              <div className="my-6 overflow-x-auto rounded-lg border border-neutral-800">
                <table className="w-full text-left border-collapse text-xs sm:text-sm">
                  {children}
                </table>
              </div>
            ),
            thead: ({ children }) => (
              <thead className="bg-neutral-800/80 text-neutral-200 border-b border-neutral-700">
                {children}
              </thead>
            ),
            th: ({ children }) => (
              <th className="px-4 py-2.5 font-semibold text-neutral-200 tracking-wider">
                {children}
              </th>
            ),
            td: ({ children }) => (
              <td className="px-4 py-2.5 border-b border-neutral-800/80 text-neutral-300">
                {children}
              </td>
            ),
            code: ({ className, children, ...props }) => {
              const match = /language-(\w+)/.exec(className || '');
              const isInline = !match && typeof children === 'string' && !children.includes('\n');
              const codeString = String(children).replace(/\n$/, '');

              if (isInline) {
                return (
                  <code className="px-1.5 py-0.5 rounded bg-neutral-800 text-amber-300 font-mono text-xs font-normal border border-neutral-700/60" {...props}>
                    {children}
                  </code>
                );
              }

              const language = match ? match[1] : 'code';
              const codeId = Math.floor(Math.random() * 100000);

              return (
                <div className="my-5 rounded-xl overflow-hidden border border-neutral-800 bg-neutral-950 shadow-md">
                  <div className="flex items-center justify-between px-4 py-1.5 bg-neutral-900/90 border-b border-neutral-800 text-xs text-neutral-400">
                    <span className="font-mono text-[11px] uppercase tracking-wider text-amber-400/90">{language}</span>
                    <button
                      onClick={() => handleCopy(codeString, codeId)}
                      className="flex items-center gap-1 px-2 py-0.5 rounded hover:bg-neutral-800 text-neutral-300 transition-colors"
                      title="Copy code"
                    >
                      {copiedIndex === codeId ? (
                        <>
                          <Check className="w-3.5 h-3.5 text-emerald-400" />
                          <span className="text-[11px] text-emerald-400">Copied!</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3.5 h-3.5" />
                          <span className="text-[11px]">Copy</span>
                        </>
                      )}
                    </button>
                  </div>
                  <pre className="p-4 overflow-x-auto text-xs sm:text-sm font-mono text-neutral-200 leading-relaxed">
                    <code>{children}</code>
                  </pre>
                </div>
              );
            },
            a: ({ href, children }) => {
              if (!href) return <span>{children}</span>;

              const resolved = resolveMarkdownLink(href, note.path);

              if (resolved.type === 'note') {
                return (
                  <button
                    onClick={() => onNavigateNote(resolved.target)}
                    className="inline text-amber-400 hover:text-amber-300 underline underline-offset-2 font-medium cursor-pointer text-left"
                    title={`Go to note: ${resolved.target}`}
                  >
                    {children}
                  </button>
                );
              }

              if (resolved.type === 'code') {
                return (
                  <button
                    onClick={() => onNavigateCode(resolved.target)}
                    className="inline-flex items-center gap-1 text-emerald-400 hover:text-emerald-300 underline underline-offset-2 font-mono text-xs cursor-pointer"
                    title={`View implementation: ${resolved.target}`}
                  >
                    <Code2 className="w-3.5 h-3.5 inline" />
                    <span>{children}</span>
                  </button>
                );
              }

              if (resolved.type === 'anchor') {
                return (
                  <a
                    href={`#${resolved.target}`}
                    className="text-amber-400 hover:text-amber-300 underline underline-offset-2"
                  >
                    {children}
                  </a>
                );
              }

              return (
                <a
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-amber-400 hover:text-amber-300 underline underline-offset-2"
                >
                  <span>{children}</span>
                  <ExternalLink className="w-3 h-3 inline" />
                </a>
              );
            },
          }}
        >
          {note.content}
        </ReactMarkdown>
      </article>

      {/* Footer Navigation (Prev / Next note in section) */}
      <div className="mt-12 pt-6 border-t border-neutral-800 flex items-center justify-between gap-4">
        {prevNote ? (
          <button
            onClick={() => onNavigateNote(prevNote.path)}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-neutral-900 hover:bg-neutral-800 border border-neutral-800 text-xs text-neutral-300 hover:text-white transition-all max-w-[45%]"
          >
            <ArrowLeft className="w-4 h-4 shrink-0 text-neutral-400" />
            <div className="text-left truncate">
              <div className="text-[10px] text-neutral-400 uppercase tracking-wider">Previous</div>
              <div className="font-medium truncate">{prevNote.title}</div>
            </div>
          </button>
        ) : (
          <div></div>
        )}

        {nextNote ? (
          <button
            onClick={() => onNavigateNote(nextNote.path)}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-neutral-900 hover:bg-neutral-800 border border-neutral-800 text-xs text-neutral-300 hover:text-white transition-all max-w-[45%]"
          >
            <div className="text-right truncate">
              <div className="text-[10px] text-neutral-400 uppercase tracking-wider">Next</div>
              <div className="font-medium truncate">{nextNote.title}</div>
            </div>
            <ArrowRight className="w-4 h-4 shrink-0 text-neutral-400" />
          </button>
        ) : (
          <div></div>
        )}
      </div>
    </div>
  );
};
