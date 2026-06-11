import { useEffect, useRef, useState } from 'react';

const BADGE = {
  mcq: 'bg-blue-100 text-blue-800 border-blue-200',
  tf: 'bg-green-100 text-green-800 border-green-200',
};

function parseQuiz(text) {
  const lines = text.trim().split('\n');
  const out = [];
  let cur = null;
  let pending = '';
  let id = 1;

  const flush = () => {
    if (!cur?.question) return;
    const t = (cur.type || '').toLowerCase();
    if (t.includes('fill') || /_{3,}/.test(cur.question)) return;
    out.push({ ...cur, id: id++ });
    cur = null;
  };

  const field = (line, names) => {
    for (const n of names) {
      const m = line.match(new RegExp(`^${n}\\s*:?\\s*`, 'i'));
      if (m) return line.slice(m[0].length).trim();
    }
    return null;
  };

  const opts = (s) => {
    const trimmed = s.trim();
    if (!trimmed) return [];
    if (/[A-Da-d][).]/.test(trimmed)) {
      const parts = trimmed
        .split(/(?:,\s*|\s+)(?=[A-Da-d][).])/)
        .map((x) => x.trim())
        .filter(Boolean);
      if (parts.length > 1) return parts;
    }
    return [trimmed];
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line || /^here (are|is)/i.test(line)) continue;
    if (/^(\*\*)?(multiple choice|true or false)/i.test(line)) continue;
    const md = line.match(/^\*\*(.+)\*\*$/);
    if (md) { pending = md[1]; continue; }

    const type = field(line, ['Question Type', 'Type']);
    if (type !== null) { flush(); pending = type; cur = { type, question: '', options: [], answer: '' }; continue; }

    const q = field(line, ['Question']);
    if (q !== null) {
      if (!cur) cur = { type: pending, question: '', options: [], answer: '' };
      cur.question = q;
      if (!cur.type) cur.type = pending;
      continue;
    }

    const o = field(line, ['Options']);
    if (o !== null) {
      if (!cur) cur = { type: pending, question: '', options: [], answer: '' };
      cur.options.push(...opts(o));
      while (i + 1 < lines.length) {
        const next = lines[i + 1].trim();
        if (!next || field(next, ['Answer', 'Question Type', 'Type', 'Question']) !== null || /^\*\*/.test(next)) break;
        cur.options.push(...opts(next));
        i += 1;
      }
      continue;
    }

    const a = field(line, ['Answer']);
    if (a !== null) {
      if (!cur) cur = { type: pending, question: '', options: [], answer: '' };
      cur.answer = a;
      flush();
      pending = '';
    }
  }
  flush();
  return out;
}

function qType(q) {
  const t = (q.type || '').toLowerCase();
  if (t.includes('mcq') || t.includes('multiple')) return { label: 'Multiple Choice', badge: 'mcq' };
  if (t.includes('true') || t.includes('false')) return { label: 'True or False', badge: 'tf' };
  if (q.options.length) return { label: 'Multiple Choice', badge: 'mcq' };
  return { label: 'True or False', badge: 'tf' };
}

function qOptions(q) {
  if (q.options.length) return q.options;
  return ['True', 'False'];
}

function match(a, b) {
  const s = (a || '').toLowerCase().trim();
  const c = (b || '').toLowerCase().trim();
  if (!s || !c) return false;
  if (s === c || c.startsWith(s)) return true;
  const sl = s.match(/^([a-d])[).]/);
  const cl = c.match(/^([a-d])[).]/);
  return sl && cl && sl[1] === cl[1];
}

function Question({ q, show, pick, selected }) {
  const { label, badge } = qType(q);
  const options = qOptions(q);
  const ok = show && match(selected, q.answer);

  return (
    <div className="mb-6 p-4 border border-gray-200 rounded-lg shadow-sm text-left">
      <div className="flex gap-2 mb-2">
        <span className="text-sm text-gray-500">Q{q.id}</span>
        <span className={`text-xs font-semibold px-2 py-0.5 rounded-full border ${BADGE[badge] || ''}`}>{label}</span>
      </div>
      <p className="font-medium text-gray-800 mb-3">{q.question}</p>
      <ul className="space-y-2 mb-3">
        {options.map((opt, idx) => {
          const sel = selected === opt;
          const correct = show && match(opt, q.answer);
          let cls = 'w-full border rounded-md px-3 py-2 text-left transition-colors ';
          if (show && correct) cls += 'bg-green-50 border-green-400 text-green-800';
          else if (show && sel && !correct) cls += 'bg-red-50 border-red-400';
          else if (sel) cls += 'bg-purple-50 border-purple-400';
          else cls += 'bg-gray-50 border-gray-100 hover:border-purple-300 cursor-pointer';
          return (
            <li key={`${q.id}-${idx}`}>
              <button type="button" disabled={show} className={cls} onClick={() => pick(q.id, opt)}>{opt}</button>
            </li>
          );
        })}
      </ul>
      {show && (
        <div className={`p-3 rounded-md border text-sm ${ok ? 'bg-green-50 border-green-200' : 'bg-amber-50 border-amber-200'}`}>
          {!selected && <p><b>Answer:</b> {q.answer}</p>}
          {selected && ok && <p className="text-green-800 font-semibold">Correct!</p>}
          {selected && !ok && <p><span className="text-red-800 font-semibold">Incorrect.</span> <b>Answer:</b> {q.answer}</p>}
        </div>
      )}
    </div>
  );
}

export default function QuizPanel({ text }) {
  const [show, setShow] = useState(false);
  const [picks, setPicks] = useState({});
  const scroll = useRef(false);
  const questions = parseQuiz(text);

  useEffect(() => { setShow(false); setPicks({}); }, [text]);
  useEffect(() => {
    if (show && scroll.current) {
      scroll.current = false;
      document.getElementById('quiz')?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [show]);

  if (!questions.length) return <pre className="whitespace-pre-wrap text-gray-800 text-left">{text}</pre>;

  const answered = questions.filter((q) => picks[q.id]).length;
  const score = show ? questions.filter((q) => match(picks[q.id], q.answer)).length : 0;

  return (
    <>
      <div className="mb-4 pb-4 border-b text-sm text-gray-600 text-left">
        {questions.length} questions{answered > 0 && !show && ` · ${answered} answered`}
        {show && <p className="font-semibold text-purple-700 mt-1">Score: {score} / {questions.length}</p>}
      </div>
      {questions.map((q) => (
        <Question key={q.id} q={q} show={show} selected={picks[q.id] || ''} pick={(id, v) => !show && setPicks((p) => ({ ...p, [id]: v }))} />
      ))}
      <div className="flex justify-center pt-2 border-t">
        <button
          type="button"
          className="bg-purple-600 hover:bg-purple-700 text-white text-sm font-medium px-6 py-2 rounded-lg"
          onClick={() => {
            if (show) setShow(false);
            else { scroll.current = true; setShow(true); }
          }}
        >
          {show ? 'Hide Answers' : 'Check Answers'}
        </button>
      </div>
    </>
  );
}
