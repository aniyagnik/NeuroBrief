import { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import logo from './logo.png';

axios.defaults.timeout = 30000;

const STAGE_LABEL = {
  pending: 'Queued…',
  downloading: 'Downloading audio from YouTube…',
  extracting: 'Extracting audio from your video…',
  transcribing: 'Transcribing audio — this may take a few minutes.',
  summarizing: 'Writing summary and quiz…',
};

function parseQuiz(text) {
  const out = [];
  let cur = null;
  let pending = '';
  let id = 1;
  const field = (line, names) => {
    for (const n of names) {
      const m = line.match(new RegExp(`^${n}\\s*:?\\s*`, 'i'));
      if (m) return line.slice(m[0].length).trim();
    }
    return null;
  };
  const splitOpts = (s) => {
    const t = s.trim();
    if (!t) return [];
    const parts = t.split(/(?:,\s*|\s+)(?=[A-Da-d][).])/).map((x) => x.trim()).filter(Boolean);
    return parts.length > 1 ? parts : [t];
  };
  const flush = () => {
    if (cur?.question && !/fill|_{3,}/i.test((cur.type || '') + cur.question)) out.push({ ...cur, id: id++ });
    cur = null;
  };

  for (const raw of text.trim().split('\n')) {
    const line = raw.trim();
    if (!line || /^here (are|is)/i.test(line) || /^(\*\*)?(multiple choice|true or false)/i.test(line)) continue;
    const md = line.match(/^\*\*(.+)\*\*$/);
    if (md) { pending = md[1]; continue; }
    const type = field(line, ['Question Type', 'Type']);
    if (type !== null) { flush(); cur = { type, question: '', options: [], answer: '' }; pending = type; continue; }
    const q = field(line, ['Question']);
    if (q !== null) { if (!cur) cur = { type: pending, question: '', options: [], answer: '' }; cur.question = q; continue; }
    const o = field(line, ['Options']);
    if (o !== null) { if (!cur) cur = { type: pending, question: '', options: [], answer: '' }; cur.options.push(...splitOpts(o)); continue; }
    const a = field(line, ['Answer']);
    if (a !== null) { if (!cur) cur = { type: pending, question: '', options: [], answer: '' }; cur.answer = a; flush(); pending = ''; }
  }
  flush();
  return out;
}

function quizType(q) {
  const t = (q.type || '').toLowerCase();
  if (t.includes('mcq') || t.includes('multiple') || q.options.length) return ['Multiple Choice', 'mcq'];
  return ['True or False', 'tf'];
}

function matchAnswer(a, b) {
  const s = (a || '').toLowerCase().trim();
  const c = (b || '').toLowerCase().trim();
  if (!s || !c) return false;
  if (s === c || c.startsWith(s)) return true;
  const sl = s.match(/^([a-d])[).]/);
  const cl = c.match(/^([a-d])[).]/);
  return sl && cl && sl[1] === cl[1];
}

function Quiz({ text }) {
  const [show, setShow] = useState(false);
  const [picks, setPicks] = useState({});
  const scroll = useRef(false);
  const qs = parseQuiz(text);

  useEffect(() => { setShow(false); setPicks({}); }, [text]);
  useEffect(() => {
    if (show && scroll.current) { scroll.current = false; document.getElementById('quiz')?.scrollIntoView({ behavior: 'smooth' }); }
  }, [show]);

  if (!qs.length) return <pre className="whitespace-pre-wrap text-gray-800">{text}</pre>;

  const score = show ? qs.filter((q) => matchAnswer(picks[q.id], q.answer)).length : 0;
  const badge = { mcq: 'bg-blue-100 text-blue-800 border-blue-200', tf: 'bg-green-100 text-green-800 border-green-200' };

  return (
    <>
      <p className="mb-4 pb-4 border-b text-sm text-gray-600">
        {qs.length} questions{show && <span className="block font-semibold text-purple-700 mt-1">Score: {score} / {qs.length}</span>}
      </p>
      {qs.map((q) => {
        const [label, kind] = quizType(q);
        const options = q.options.length ? q.options : ['True', 'False'];
        const ok = show && matchAnswer(picks[q.id], q.answer);
        return (
          <div key={q.id} className="mb-6 p-4 border rounded-lg text-left">
            <div className="flex gap-2 mb-2">
              <span className="text-sm text-gray-500">Q{q.id}</span>
              <span className={`text-xs px-2 py-0.5 rounded-full border ${badge[kind]}`}>{label}</span>
            </div>
            <p className="font-medium mb-3">{q.question}</p>
            <ul className="space-y-2 mb-3">
              {options.map((opt, i) => {
                const sel = picks[q.id] === opt;
                const good = show && matchAnswer(opt, q.answer);
                let cls = 'w-full border rounded-md px-3 py-2 text-left ';
                if (show && good) cls += 'bg-green-50 border-green-400';
                else if (show && sel) cls += 'bg-red-50 border-red-400';
                else if (sel) cls += 'bg-purple-50 border-purple-400';
                else cls += 'bg-gray-50 hover:border-purple-300 cursor-pointer';
                return (
                  <li key={i}>
                    <button type="button" disabled={show} className={cls} onClick={() => setPicks((p) => ({ ...p, [q.id]: opt }))}>{opt}</button>
                  </li>
                );
              })}
            </ul>
            {show && (
              <p className={`text-sm p-3 rounded border ${ok ? 'bg-green-50' : 'bg-amber-50'}`}>
                {ok ? 'Correct!' : <>Incorrect. <b>Answer:</b> {q.answer}</>}
              </p>
            )}
          </div>
        );
      })}
      <div className="flex justify-center pt-2 border-t">
        <button type="button" className="bg-purple-600 text-white px-6 py-2 rounded-lg" onClick={() => { if (show) setShow(false); else { scroll.current = true; setShow(true); } }}>
          {show ? 'Hide Answers' : 'Check Answers'}
        </button>
      </div>
    </>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [url, setUrl] = useState('');
  const [level, setLevel] = useState('medium');
  const [summary, setSummary] = useState('');
  const [quiz, setQuiz] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState('');
  const [stage, setStage] = useState('');
  const [done, setDone] = useState(false);
  const [jobId, setJobId] = useState(null);

  useEffect(() => {
    if (!jobId) return;
    let stop = false;
    (async () => {
      for (let t = Date.now(); !stop && Date.now() - t < 1200000; ) {
        try {
          const { data: j } = await axios.get(`/api/jobs/${jobId}`);
          setStage(j.stage || j.status || '');
          setMsg(j.status_message || 'Working…');
          if (j.status === 'completed') {
            setSummary((j.summary || '').trimStart());
            setQuiz(j.quiz || '');
            setDone(true);
            setLoading(false);
            setJobId(null);
            setTimeout(() => document.getElementById('results')?.scrollIntoView({ behavior: 'smooth' }), 100);
            return;
          }
          if (j.status === 'failed') throw new Error(j.error || 'Failed');
        } catch (e) {
          if (!stop) { setError(e.response?.data?.error || e.message); setLoading(false); setJobId(null); }
          return;
        }
        await new Promise((r) => setTimeout(r, 2000));
      }
      if (!stop) { setError('Timed out.'); setLoading(false); setJobId(null); }
    })();
    return () => { stop = true; };
  }, [jobId]);

  const submit = async (e) => {
    e.preventDefault();
    if (loading || (!file && !url.trim())) { setError('Add a video or YouTube URL.'); return; }
    setError('');
    setSummary('');
    setQuiz('');
    setDone(false);
    setLoading(true);
    setMsg('Submitting…');
    const form = new FormData();
    if (file) form.append('video_file', file);
    if (url.trim()) form.append('youtube_url', url.trim());
    form.append('level', level);
    try {
      const { data } = await axios.post('/process', form);
      setJobId(data.job_id);
    } catch (e) {
      setError(e.response?.data?.error || e.message);
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-blue-100 p-6 flex flex-col items-center">
      <img src={logo} alt="" className="w-36 md:w-48 mb-4" />
      <h1 className="text-3xl font-bold text-blue-700 mb-6">🧠 NeuroBrief</h1>

      <form onSubmit={submit} className="w-full max-w-lg bg-white p-6 rounded-xl shadow space-y-4 text-left">
        <label className="block">Upload Video<input type="file" accept="video/*" className="w-full border rounded px-3 py-2 mt-1" onChange={(e) => setFile(e.target.files[0] || null)} /></label>
        <label className="block">YouTube URL<input type="url" className="w-full border rounded px-3 py-2 mt-1" value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://youtube.com/..." /></label>
        <label className="block">Difficulty
          <select className="w-full border rounded px-3 py-2 mt-1" value={level} onChange={(e) => setLevel(e.target.value)}>
            <option value="easy">Easy</option><option value="medium">Medium</option><option value="high">High</option>
          </select>
        </label>
        <button disabled={loading} className="bg-blue-600 text-white w-full py-2 rounded disabled:opacity-50">{loading ? 'Working…' : 'Generate Summary & Quiz'}</button>
      </form>

      {done && (
        <div className="mt-4 w-full max-w-lg flex items-center gap-3 px-4 py-3 bg-white border border-green-200 rounded-xl shadow-sm text-green-800">
          <span className="text-lg">✓</span>
          <p className="text-sm font-medium">Finished — scroll down for results.</p>
        </div>
      )}
      {loading && !done && (
        <div className="mt-4 w-full max-w-lg flex items-center gap-3 px-4 py-3 bg-white border border-gray-200 rounded-xl shadow-sm">
          <div className="h-5 w-5 shrink-0 rounded-full border-2 border-blue-200 border-t-blue-600 animate-spin" aria-hidden="true" />
          <p className="text-sm text-gray-700">{STAGE_LABEL[stage] || msg || 'Processing…'}</p>
        </div>
      )}
      {error && <p className="mt-4 text-red-600 text-sm">{error}</p>}

      <div id="results" className="w-full max-w-3xl mt-6">
        {summary && (
          <div className="bg-white p-6 rounded-lg shadow mb-6 text-left">
            <h2 className="text-xl font-semibold text-blue-600 mb-2">📄 Summary</h2>
            <p className="whitespace-pre-line">{summary}</p>
          </div>
        )}
        {quiz && (
          <div id="quiz" className="bg-white p-6 rounded-lg shadow text-left">
            <h2 className="text-xl font-semibold text-purple-600 mb-4">🧠 Quiz</h2>
            <Quiz text={quiz} />
          </div>
        )}
      </div>
    </div>
  );
}
