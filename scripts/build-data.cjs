const fs = require('fs');
const path = require('path');

const SECTIONS = [
  { id: 'algorithms', title: 'Algorithms', icon: 'Binary', desc: 'Patterns, data structures, sorting, metaheuristics' },
  { id: 'machine-learning', title: 'Machine Learning', icon: 'BrainCircuit', desc: 'Linear models, trees, clustering, evaluation, preprocessing' },
  { id: 'deep-learning', title: 'Deep Learning', icon: 'Layers', desc: 'PyTorch, CNNs, RNNs, VAEs, transformers, regularization' },
  { id: 'python', title: 'Python', icon: 'Code', desc: 'OOP, iterators, async, memory model, typing, common traps' },
  { id: 'sql', title: 'SQL', icon: 'Database', desc: 'Window functions, CTEs, joins, query order, optimization' },
  { id: 'statistics', title: 'Statistics', icon: 'BarChart2', desc: 'Probability, distributions, hypothesis testing, A/B testing' },
];

function walkDir(dir, fileList = []) {
  if (!fs.existsSync(dir)) return fileList;
  const files = fs.readdirSync(dir);
  for (const file of files) {
    if (file === 'node_modules' || file === '.git' || file === 'dist' || file === 'scripts' || file === 'src') continue;
    const fullPath = path.join(dir, file);
    const stat = fs.statSync(fullPath);
    if (stat.isDirectory()) {
      walkDir(fullPath, fileList);
    } else {
      fileList.push(fullPath);
    }
  }
  return fileList;
}

function extractTitle(content, filename) {
  const match = content.match(/^#\s+(.+)$/m);
  if (match) return match[1].trim();
  const base = path.basename(filename, path.extname(filename));
  return base.replace(/[-_]/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function extractExcerpt(content) {
  const lines = content.split('\n');
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed && !trimmed.startsWith('#') && !trimmed.startsWith('```') && !trimmed.startsWith('|') && !trimmed.startsWith('![')) {
      return trimmed.slice(0, 220);
    }
  }
  return '';
}

function extractInterviewQuestions(content, notePath, noteTitle, section) {
  const questions = [];
  const regex = /##\s+Common\s+interview\s+questions([\s\S]*?)(?=\n##\s+|$)/i;
  const match = content.match(regex);
  if (match) {
    const block = match[1];
    const itemRegex = /-\s+([^\n]+)/g;
    let itemMatch;
    while ((itemMatch = itemRegex.exec(block)) !== null) {
      const fullText = itemMatch[1].trim();
      let question = fullText;
      let hint = '';
      const hintMatch = fullText.match(/\*?\((.+?)\)\*?$/);
      if (hintMatch) {
        hint = hintMatch[1];
        question = fullText.replace(/\*?\((.+?)\)\*?$/, '').trim();
      }
      questions.push({
        id: `${notePath}#q${questions.length + 1}`,
        question,
        hint,
        notePath,
        noteTitle,
        section,
      });
    }
  }
  return questions;
}

function extractClues(content, patternName, notePath) {
  const clues = [];
  const match = content.match(/###\s+Key\s+clues([\s\S]*?)(?=\n##\s+|$)/i);
  if (match) {
    const block = match[1];
    const itemRegex = /-\s+([^\n]+)/g;
    let itemMatch;
    while ((itemMatch = itemRegex.exec(block)) !== null) {
      clues.push(itemMatch[1].trim());
    }
  }
  return clues;
}

function extractCodeRefs(content, currentDir) {
  const refs = [];
  const regex = /\[.*?\]\((.*?\.(?:py|sh))\)/g;
  let match;
  while ((match = regex.exec(content)) !== null) {
    const linkPath = match[1].split('#')[0];
    const resolved = path.normalize(path.join(currentDir, linkPath)).replace(/\\/g, '/');
    refs.push(resolved);
  }
  return Array.from(new Set(refs));
}

function parseAlgorithmsLookupTable(readmeContent) {
  const table = [];
  const regex = /##\s+Pattern\s+lookup\s+table([\s\S]*?)(?=\n##\s+|$)/i;
  const match = readmeContent.match(regex);
  if (match) {
    const lines = match[1].split('\n');
    for (const line of lines) {
      if (line.includes('|') && !line.includes('---|---') && !line.includes('If the problem says')) {
        const parts = line.split('|').map(s => s.trim()).filter(Boolean);
        if (parts.length >= 2) {
          const clue = parts[0];
          const patternTarget = parts[1];
          const patternMatch = patternTarget.match(/\[(.*?)\]\((.*?)\)/);
          const patternName = patternMatch ? patternMatch[1] : patternTarget;
          const targetPath = patternMatch ? path.normalize(path.join('algorithms', patternMatch[2])).replace(/\\/g, '/') : '';
          table.push({
            clue,
            pattern: patternName,
            targetPath,
          });
        }
      }
    }
  }
  return table;
}

function extractQuickRevisionOrder(content) {
  const regex = /##\s+Quick\s+revision\s+order([\s\S]*?)(?=\n##\s+|$)/i;
  const match = content.match(regex);
  if (match) {
    return match[1].trim();
  }
  return '';
}

function main() {
  const files = walkDir('.');
  const mdFiles = files.filter(f => f.endsWith('.md') && !f.startsWith('AI_STUDIO') && !f.startsWith('README.md'));
  const pyFiles = files.filter(f => f.endsWith('.py'));

  console.log(`Processing ${mdFiles.length} markdown files and ${pyFiles.length} python files...`);

  const notes = [];
  const allQuestions = [];
  const allClues = [];
  const codeFiles = [];
  let algorithmsLookupTable = [];

  // Parse Python files
  for (const pyPath of pyFiles) {
    const cleanPath = pyPath.replace(/^\.\//, '').replace(/\\/g, '/');
    const content = fs.readFileSync(pyPath, 'utf8');
    const section = cleanPath.split('/')[0];
    const parts = cleanPath.split('/');
    const subsection = parts.length > 2 ? parts[1] : '';
    codeFiles.push({
      id: cleanPath,
      path: cleanPath,
      filename: path.basename(cleanPath),
      section,
      subsection,
      content,
      lines: content.split('\n').length,
    });
  }

  // Parse Markdown files
  for (const mdPath of mdFiles) {
    const cleanPath = mdPath.replace(/^\.\//, '').replace(/\\/g, '/');
    const content = fs.readFileSync(mdPath, 'utf8');
    const parts = cleanPath.split('/');
    const section = parts[0];
    const subsection = parts.length > 2 ? parts[1] : '';
    const filename = path.basename(cleanPath);
    const title = extractTitle(content, filename);
    const excerpt = extractExcerpt(content);
    const isReadme = filename.toLowerCase() === 'readme.md';
    const isScaffolding = cleanPath.includes('leetcode');
    const codeRefs = extractCodeRefs(content, path.dirname(cleanPath));
    const quickRev = isReadme ? extractQuickRevisionOrder(content) : '';

    if (cleanPath === 'algorithms/README.md') {
      algorithmsLookupTable = parseAlgorithmsLookupTable(content);
    }

    const noteQuestions = extractInterviewQuestions(content, cleanPath, title, section);
    allQuestions.push(...noteQuestions);

    if (section === 'algorithms' && cleanPath.includes('patterns/') && !isReadme) {
      const clues = extractClues(content, title, cleanPath);
      if (clues.length > 0) {
        allClues.push({
          pattern: title,
          notePath: cleanPath,
          clues,
        });
      }
    }

    notes.push({
      id: cleanPath.replace(/\.md$/, ''),
      path: cleanPath,
      title,
      section,
      subsection,
      filename,
      isReadme,
      isScaffolding,
      excerpt,
      content,
      codeRefs,
      quickRevisionOrder: quickRev,
      interviewQuestionsCount: noteQuestions.length,
    });
  }

  // Build output TS
  const outDir = path.join(__dirname, '..', 'src', 'data');
  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
  }

  const outputTs = `// Automatically generated by scripts/build-data.js
// Do not edit manually. Total notes: ${notes.length}, Code files: ${codeFiles.length}

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

export const SECTIONS: SectionMeta[] = ${JSON.stringify(SECTIONS, null, 2)};

export const NOTES: NoteItem[] = ${JSON.stringify(notes, null, 2)};

export const CODE_FILES: CodeItem[] = ${JSON.stringify(codeFiles, null, 2)};

export const ALL_QUESTIONS: InterviewQuestion[] = ${JSON.stringify(allQuestions, null, 2)};

export const PATTERN_CLUES: PatternClueItem[] = ${JSON.stringify(allClues, null, 2)};

export const ALGORITHMS_LOOKUP_TABLE: LookupItem[] = ${JSON.stringify(algorithmsLookupTable, null, 2)};
`;

  fs.writeFileSync(path.join(outDir, 'repoData.ts'), outputTs, 'utf8');
  console.log(`Wrote repoData.ts successfully! Generated ${notes.length} notes, ${codeFiles.length} code files, ${allQuestions.length} interview questions, ${allClues.length} pattern recognition items, ${algorithmsLookupTable.length} lookup rows.`);
}

main();
