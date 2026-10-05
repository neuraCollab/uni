// Builds src/data/repo.json from the markdown notes and Python code in this repo.
// Runs automatically before `dev`, `build` and `lint` (see package.json).
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const OUT = path.join(ROOT, 'src', 'data', 'repo.json');

const SECTIONS = [
  { id: 'algorithms', title: 'Algorithms', desc: 'Patterns, data structures, sorting, metaheuristics' },
  { id: 'machine-learning', title: 'Machine Learning', desc: 'Linear models, trees, clustering, evaluation, preprocessing' },
  { id: 'deep-learning', title: 'Deep Learning', desc: 'PyTorch, CNNs, RNNs, VAEs, transformers, regularization' },
  { id: 'python', title: 'Python', desc: 'OOP, iterators, async, memory model, typing, common traps' },
  { id: 'sql', title: 'SQL', desc: 'Window functions, CTEs, joins, query order, optimization' },
  { id: 'statistics', title: 'Statistics', desc: 'Probability, distributions, hypothesis testing, A/B testing' },
];

const posix = (p) => p.split(path.sep).join('/');

function walk(dir, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full, out);
    else out.push(posix(path.relative(ROOT, full)));
  }
  return out;
}

// Body of a `## Heading` (or `### Heading`) section, up to the next `##`.
function section(content, heading, level = 2) {
  const re = new RegExp(`^${'#'.repeat(level)}\\s+${heading}\\s*$([\\s\\S]*?)(?=^##\\s|(?![\\s\\S]))`, 'im');
  return content.match(re)?.[1] ?? '';
}

const bullets = (block) => [...block.matchAll(/^\s*-\s+(.+)$/gm)].map((m) => m[1].trim());

// Resolve a relative markdown link against the note's directory.
const resolve = (fromFile, href) => posix(path.normalize(path.join(path.dirname(fromFile), href.split('#')[0])));

function title(content, file) {
  const h1 = content.match(/^#\s+(.+)$/m);
  if (h1) return h1[1].trim();
  return path.basename(file, '.md').replace(/[-_]/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

// First prose paragraph, hard-wrapped lines joined.
function excerpt(content) {
  const para = content
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .find((p) => p && !/^(#|```|\||!\[|-|\d+\.)/.test(p));
  const text = (para ?? '').replace(/\s+/g, ' ');
  return text.length > 220 ? `${text.slice(0, 220).trimEnd()}…` : text;
}

function questions(content, note) {
  return bullets(section(content, 'Common interview questions')).map((text, i) => {
    const hint = text.match(/\*?\((.+?)\)\*?$/);
    return {
      id: `${note.path}#q${i + 1}`,
      question: hint ? text.slice(0, hint.index).trim() : text,
      hint: hint ? hint[1] : '',
      notePath: note.path,
      noteTitle: note.title,
      section: note.section,
    };
  });
}

function lookupTable(content, file) {
  return section(content, 'Pattern lookup table')
    .split('\n')
    .filter((l) => l.startsWith('|') && !/^\|\s*-/.test(l))
    .slice(1) // header row
    .map((l) => l.split('|').map((s) => s.trim()).filter(Boolean))
    .filter((cells) => cells.length >= 2)
    .map(([clue, target]) => {
      const link = target.match(/\[(.*?)\]\((.*?)\)/);
      return { clue, pattern: link ? link[1] : target, notePath: link ? resolve(file, link[2]) : '' };
    });
}

function reviewOrder(content, file) {
  const block = section(content, 'Suggested review order');
  return [...block.matchAll(/\]\(([^)]+\.md)\)/g)].map((m) => resolve(file, m[1]));
}

function main() {
  const files = walk(ROOT).filter((f) => SECTIONS.some((s) => f.startsWith(`${s.id}/`)));

  const code = files
    .filter((f) => f.endsWith('.py'))
    .map((f) => ({ path: f, filename: path.basename(f), section: f.split('/')[0], content: fs.readFileSync(path.join(ROOT, f), 'utf8') }));
  const codePaths = new Set(code.map((c) => c.path));

  const notes = [];
  const allQuestions = [];
  const patternClues = [];
  const cramPlans = [];
  let lookup = [];

  for (const file of files.filter((f) => f.endsWith('.md') && path.basename(f) !== 'template.md')) {
    const content = fs.readFileSync(path.join(ROOT, file), 'utf8');
    const parts = file.split('/');
    const note = {
      path: file,
      title: title(content, file),
      section: parts[0],
      subsection: parts.length > 2 ? parts[1] : '',
      filename: parts[parts.length - 1],
      isReadme: parts[parts.length - 1] === 'README.md',
      isScaffolding: file.includes('/leetcode/'),
      excerpt: excerpt(content),
      content,
      codeRefs: [...new Set([...content.matchAll(/\]\(([^)]+\.py)(?:#[^)]*)?\)/g)].map((m) => resolve(file, m[1])))].filter((p) => codePaths.has(p)),
      questionCount: 0,
    };

    const qs = questions(content, note);
    note.questionCount = qs.length;
    allQuestions.push(...qs);

    if (note.isReadme && parts.length === 2) {
      const steps = reviewOrder(content, file);
      if (steps.length) cramPlans.push({ section: note.section, steps });
      if (note.section === 'algorithms') lookup = lookupTable(content, file);
    }

    if (file.startsWith('algorithms/patterns/')) {
      const clues = bullets(section(content, 'Key clues', 3));
      if (clues.length) patternClues.push({ pattern: note.title, notePath: file, clues });
    }

    notes.push(note);
  }

  const notePaths = new Set(notes.map((n) => n.path));
  for (const plan of cramPlans) {
    const missing = plan.steps.filter((s) => !notePaths.has(s));
    if (missing.length) throw new Error(`${plan.section}/README.md review order links to missing notes: ${missing.join(', ')}`);
  }

  // Keep cram plans in sidebar section order.
  cramPlans.sort((a, b) => SECTIONS.findIndex((s) => s.id === a.section) - SECTIONS.findIndex((s) => s.id === b.section));

  const data = { sections: SECTIONS, notes, code, questions: allQuestions, patternClues, lookup, cramPlans };
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, JSON.stringify(data));
  console.log(
    `repo.json: ${notes.length} notes, ${code.length} code files, ${allQuestions.length} questions, ` +
      `${patternClues.length} patterns, ${lookup.length} lookup rows, ${cramPlans.length} cram plans`,
  );
}

main();
