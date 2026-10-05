import { codeByPath, noteByPath } from './data';

export type ResolvedLink =
  | { type: 'note'; target: string; anchor?: string }
  | { type: 'code'; target: string }
  | { type: 'anchor'; target: string }
  | { type: 'external'; target: string };

const REPO_URL = 'https://github.com/neuraCollab/uni';

function join(fromFile: string, href: string): string {
  const out: string[] = fromFile.split('/').slice(0, -1);
  for (const seg of href.split('/')) {
    if (!seg || seg === '.') continue;
    if (seg === '..') out.pop();
    else out.push(seg);
  }
  return out.join('/');
}

/** Map a relative markdown link inside `fromNote` to an in-app note/code target. */
export function resolveLink(href: string, fromNote: string): ResolvedLink {
  if (href.startsWith('#')) return { type: 'anchor', target: href.slice(1) };
  if (/^[a-z]+:/i.test(href)) return { type: 'external', target: href };

  const [file, anchor] = href.split('#');
  const path = join(fromNote, file);
  if (codeByPath.has(path)) return { type: 'code', target: path };

  for (const candidate of [path, `${path}.md`, `${path}/README.md`]) {
    if (noteByPath.has(candidate)) return { type: 'note', target: candidate, anchor };
  }
  // Folders and other files the app doesn't render: fall back to GitHub.
  return { type: 'external', target: `${REPO_URL}/tree/main/${path}` };
}
