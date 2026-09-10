import { NOTES, CODE_FILES } from '../data/repoData';

export interface ResolvedLink {
  type: 'note' | 'code' | 'anchor' | 'external';
  target: string;
  anchor?: string;
}

export function resolveMarkdownLink(href: string, currentNotePath: string): ResolvedLink {
  if (!href) return { type: 'external', target: '#' };

  if (href.startsWith('http://') || href.startsWith('https://')) {
    return { type: 'external', target: href };
  }

  if (href.startsWith('#')) {
    return { type: 'anchor', target: href.slice(1) };
  }

  const [pathPart, anchorPart] = href.split('#');

  // Normalize relative path
  const currentDir = currentNotePath.includes('/')
    ? currentNotePath.substring(0, currentNotePath.lastIndexOf('/'))
    : '';

  const segments = (currentDir ? currentDir.split('/') : []).concat(pathPart.split('/'));
  const resolvedSegments: string[] = [];

  for (const seg of segments) {
    if (!seg || seg === '.') continue;
    if (seg === '..') {
      resolvedSegments.pop();
    } else {
      resolvedSegments.push(seg);
    }
  }

  const normalized = resolvedSegments.join('/');

  // Check if it's a code file
  if (normalized.endsWith('.py') || normalized.endsWith('.sh') || normalized.includes('/code/')) {
    const codeMatch = CODE_FILES.find(c => c.path === normalized || c.path.endsWith(normalized) || normalized.endsWith(c.path));
    if (codeMatch) {
      return { type: 'code', target: codeMatch.path, anchor: anchorPart };
    }
  }

  // Check if it's a note (with or without .md)
  let cleanNotePath = normalized.endsWith('.md') ? normalized : `${normalized}.md`;
  let noteMatch = NOTES.find(n => n.path === cleanNotePath);

  if (!noteMatch) {
    // try if normalized points to directory containing README.md
    cleanNotePath = `${normalized}/README.md`;
    noteMatch = NOTES.find(n => n.path === cleanNotePath);
  }

  if (!noteMatch) {
    // try matching by filename or id
    const baseName = normalized.split('/').pop()?.replace('.md', '');
    noteMatch = NOTES.find(n => n.filename.replace('.md', '') === baseName || n.id.endsWith(`/${baseName}`));
  }

  if (noteMatch) {
    return { type: 'note', target: noteMatch.path, anchor: anchorPart };
  }

  return { type: 'external', target: href };
}
