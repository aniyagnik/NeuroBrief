import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import neuroBriefLogo from './logo.png';

axios.defaults.timeout = 30000;

const STAGE_HINTS = {
  pending: 'Waiting to start…',
  downloading: 'Fetching audio from YouTube…',
  extracting: 'Pulling audio from your video…',
  transcribing: 'Transcribing audio — often 2–10 minutes for long videos.',
  summarizing: 'Generating summary and quiz…',
};

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

const TYPE_BADGE = {
  mcq: 'bg-blue-100 text-blue-800 border-blue-200',
  tf: 'bg-green-100 text-green-800 border-green-200',
  fill: 'bg-amber-100 text-amber-800 border-amber-200',
  default: 'bg-purple-100 text-purple-800 border-purple-200',
};

function normalizeQuestionType(raw) {
  const t = (raw || '').toLowerCase().replace(/\*\*/g, '').trim();
  if (!t) return { label: 'Question', badge: 'default' };
  if (t.includes('mcq') || t.includes('multiple choice')) {
    return { label: 'Multiple Choice', badge: 'mcq' };
  }
  if (t.includes('true') || t.includes('false') || t.includes('t/f')) {
    return { label: 'True or False', badge: 'tf' };
  }
  if (t.includes('fill')) {
    return { label: 'Fill in the Blank', badge: 'fill' };
  }
  return { label: raw.replace(/\*\*/g, '').trim(), badge: 'default' };
}

function normalizeAnswer(text) {
  return (text || '').toLowerCase().replace(/\s+/g, ' ').trim();
}

function answersMatch(selected, correct) {
  const s = normalizeAnswer(selected);
  const c = normalizeAnswer(correct);
  if (!s || !c) return false;
  if (s === c) return true;

  const selectedLetter = s.match(/^([a-d])[).]/);
  const correctLetter = c.match(/^([a-d])[).]/);
  if (selectedLetter && correctLetter && selectedLetter[1] === correctLetter[1]) {
    return true;
  }
  if (/^[a-d]$/.test(c) && selectedLetter && selectedLetter[1] === c) {
    return true;
  }

  if ((s === 'true' || s === 'false') && c.startsWith(s)) {
    return true;
  }

  return false;
}

function parseOptionsLine(text) {
  const trimmed = text.trim();
  if (!trimmed) return [];
  if (/,\s*[A-Da-d][).]/.test(trimmed)) {
    return trimmed.split(/,\s*(?=[A-Da-d][).])/).map((o) => o.trim()).filter(Boolean);
  }
  return [trimmed];
}

function extractQuizField(line, prefixes) {
  const trimmed = line.trimStart();
  for (const prefix of prefixes) {
    const re = new RegExp(`^${prefix}\\s*:?\\s*`, 'i');
    if (re.test(trimmed)) {
      return trimmed.replace(re, '').trim();
    }
  }
  return null;
}

function parseQuizText(quizText) {
  const lines = quizText.trim().split('\n');
  const questions = [];
  let current = null;
  let pendingType = '';
  let id = 1;

  const flush = () => {
    if (current?.question) {
      questions.push({ ...current, id: id++ });
    }
    current = null;
  };

  for (let i = 0; i < lines.length; i++) {
    const trimmed = lines[i].trim();
    if (!trimmed) continue;
    if (/^here are (the )?quiz questions/i.test(trimmed)) continue;

    const mdHeader = trimmed.match(/^\*\*(.+?)\*\*$/);
    if (mdHeader) {
      pendingType = mdHeader[1].trim();
      continue;
    }

    const typeVal = extractQuizField(trimmed, ['Question Type', 'Type']);
    if (typeVal !== null) {
      flush();
      pendingType = typeVal;
      current = { type: typeVal, question: '', options: [], answer: '' };
      continue;
    }

    const questionVal = extractQuizField(trimmed, ['Question']);
    if (questionVal !== null) {
      if (!current) {
        current = { type: pendingType, question: '', options: [], answer: '' };
      }
      current.question = questionVal;
      if (!current.type && pendingType) current.type = pendingType;
      continue;
    }

    const optionsVal = extractQuizField(trimmed, ['Options']);
    if (optionsVal !== null) {
      if (!current) {
        current = { type: pendingType, question: '', options: [], answer: '' };
      }
      current.options.push(...parseOptionsLine(optionsVal));
      i += 1;
      while (i < lines.length) {
        const next = lines[i].trim();
        if (!next) {
          i += 1;
          continue;
        }
        if (extractQuizField(next, ['Answer', 'Question Type', 'Type', 'Question']) !== null) {
          i -= 1;
          break;
        }
        if (/^\*\*.+\*\*$/.test(next)) {
          i -= 1;
          break;
        }
        current.options.push(...parseOptionsLine(next));
        i += 1;
      }
      continue;
    }

    const answerVal = extractQuizField(trimmed, ['Answer']);
    if (answerVal !== null) {
      if (!current) {
        current = { type: pendingType, question: '', options: [], answer: '' };
      }
      current.answer = answerVal;
      flush();
      pendingType = '';
    }
  }

  flush();
  return questions;
}

function QuizQuestion({ q, showAnswer, selected, onSelect }) {
  const { label, badge } = normalizeQuestionType(q.type);
  const badgeClass = TYPE_BADGE[badge] || TYPE_BADGE.default;
  const isFillBlank = badge === 'fill';
  const displayOptions =
    q.options.length > 0 ? q.options : label === 'True or False' ? ['True', 'False'] : [];
  const isCorrect = showAnswer && answersMatch(selected, q.answer);

  return (
    <div className="mb-6 p-4 border border-gray-200 rounded-lg shadow-sm">
      <div className="flex flex-wrap items-center gap-2 mb-2">
        <span className="text-sm font-medium text-gray-500">Q{q.id}</span>
        <span
          className={`text-xs font-semibold px-2.5 py-0.5 rounded-full border ${badgeClass}`}
        >
          {label}
        </span>
      </div>
      <p className="font-medium text-gray-800 mb-3">{q.question}</p>
      {isFillBlank ? (
        <input
          type="text"
          value={selected || ''}
          onChange={(e) => onSelect(q.id, e.target.value)}
          disabled={showAnswer}
          placeholder="Type your answer…"
          className="w-full border border-gray-200 rounded-md px-3 py-2 text-gray-800 mb-3 focus:outline-none focus:ring-2 focus:ring-purple-300 disabled:bg-gray-50"
        />
      ) : (
        displayOptions.length > 0 && (
          <ul className="space-y-2 mb-3">
            {displayOptions.map((opt, idx) => {
              const isSelected = selected === opt;
              const isCorrectOption = showAnswer && answersMatch(opt, q.answer);
              const isWrongSelection = showAnswer && isSelected && !answersMatch(opt, q.answer);

              let optionClass =
                'text-gray-700 border rounded-md px-3 py-2 text-left w-full transition-colors';
              if (showAnswer && isCorrectOption) {
                optionClass += ' bg-green-50 border-green-400 text-green-800';
              } else if (isWrongSelection) {
                optionClass += ' bg-red-50 border-red-400 text-red-800';
              } else if (isSelected) {
                optionClass += ' bg-purple-50 border-purple-400 ring-1 ring-purple-300';
              } else {
                optionClass += ' bg-gray-50 border-gray-100 hover:border-purple-300 hover:bg-purple-50/50 cursor-pointer';
              }

              return (
                <li key={idx}>
                  <button
                    type="button"
                    onClick={() => onSelect(q.id, opt)}
                    disabled={showAnswer}
                    className={optionClass}
                  >
                    {opt}
                  </button>
                </li>
              );
            })}
          </ul>
        )
      )}
      {showAnswer && (
        <div
          className={`mt-2 p-3 rounded-md border ${
            !selected
              ? 'bg-gray-50 border-gray-200'
              : isCorrect
                ? 'bg-green-50 border-green-200'
                : 'bg-amber-50 border-amber-200'
          }`}
        >
          {!selected && (
            <p className="text-sm text-gray-700">
              <span className="font-semibold">Correct answer:</span> {q.answer}
            </p>
          )}
          {selected && isCorrect && (
            <p className="text-sm text-green-800 font-semibold">Correct!</p>
          )}
          {selected && !isCorrect && (
            <>
              <p className="text-sm text-red-800 font-semibold">Incorrect.</p>
              <p className="text-sm text-green-800 mt-1">
                <span className="font-semibold">Correct answer:</span> {q.answer}
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function QuizPanel({ quizText }) {
  const [showAllAnswers, setShowAllAnswers] = useState(false);
  const [selections, setSelections] = useState({});
  const scrollToQuizRef = useRef(false);

  useEffect(() => {
    setShowAllAnswers(false);
    setSelections({});
  }, [quizText]);

  useEffect(() => {
    if (!showAllAnswers || !scrollToQuizRef.current) return;
    scrollToQuizRef.current = false;
    document.getElementById('quiz')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [showAllAnswers]);

  const handleSelect = (questionId, value) => {
    if (showAllAnswers) return;
    setSelections((prev) => ({ ...prev, [questionId]: value }));
  };

  const handleCheckAnswers = () => {
    if (showAllAnswers) {
      setShowAllAnswers(false);
      return;
    }
    scrollToQuizRef.current = true;
    setShowAllAnswers(true);
  };

  try {
    const questions = parseQuizText(quizText);

    if (questions.length === 0) {
      return <pre className="whitespace-pre-wrap text-gray-800">{quizText}</pre>;
    }

    const answeredCount = questions.filter((q) => selections[q.id]?.trim()).length;
    const score = showAllAnswers
      ? questions.filter((q) => answersMatch(selections[q.id], q.answer)).length
      : 0;

    return (
      <>
        <div className="mb-4 pb-4 border-b border-gray-100 text-sm text-gray-600 text-left">
          <p>
            {questions.length} question{questions.length === 1 ? '' : 's'}
            {answeredCount > 0 && !showAllAnswers && (
              <span className="text-purple-700"> · {answeredCount} answered</span>
            )}
          </p>
          {showAllAnswers && (
            <p className="font-semibold text-purple-700 mt-1">
              Score: {score} / {questions.length}
            </p>
          )}
        </div>
        {questions.map((q) => (
          <QuizQuestion
            key={q.id}
            q={q}
            showAnswer={showAllAnswers}
            selected={selections[q.id] || ''}
            onSelect={handleSelect}
          />
        ))}
        <div className="flex justify-center pt-2 border-t border-gray-100">
          <button
            type="button"
            onClick={handleCheckAnswers}
            className="bg-purple-600 hover:bg-purple-700 text-white text-sm font-medium px-6 py-2 rounded-lg shadow transition-colors"
          >
            {showAllAnswers ? 'Hide Answers' : 'Check Answers'}
          </button>
        </div>
      </>
    );
  } catch (err) {
    console.error('QuizPanel crashed:', err);
    return <pre className="whitespace-pre-wrap text-gray-800">{quizText}</pre>;
  }
}

function App() {
  const [uid, setUid] = useState('');
  const [videoFile, setVideoFile] = useState(null);
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [difficulty, setDifficulty] = useState('medium');
  const [summary, setSummary] = useState('');
  const [quiz, setQuiz] = useState('');
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [stage, setStage] = useState('');
  const [done, setDone] = useState(false);
  const [activeJobId, setActiveJobId] = useState(null);

  useEffect(() => {
    if (!activeJobId) return undefined;

    let cancelled = false;

    const poll = async () => {
      const deadline = Date.now() + 20 * 60 * 1000;

      while (!cancelled && Date.now() < deadline) {
        try {
          const statusRes = await axios.get(`/api/jobs/${activeJobId}`, {
            headers: { 'Cache-Control': 'no-cache' },
          });
          if (cancelled) return;

          const job = statusRes.data;
          const currentStage = job.stage || job.status || '';
          setStage(currentStage);
          setStatusMessage(job.status_message || job.status || 'Working…');

          if (String(job.status).toLowerCase() === 'completed') {
            const nextSummary = (job.summary || '').trimStart();
            const nextQuiz = job.quiz || '';
            if (!nextSummary && !nextQuiz) {
              throw new Error('Job finished but the server returned no summary or quiz.');
            }
            setSummary(nextSummary);
            setQuiz(nextQuiz);
            setUid(job.uid || activeJobId);
            setStatusMessage('Done! Scroll down for results.');
            setDone(true);
            setLoading(false);
            setActiveJobId(null);
            setTimeout(() => {
              document.getElementById('results')?.scrollIntoView({ behavior: 'smooth' });
            }, 100);
            return;
          }

          if (String(job.status).toLowerCase() === 'failed') {
            throw new Error(job.error || 'Job failed');
          }
        } catch (err) {
          if (cancelled) return;
          console.error(err);
          setError(err.response?.data?.error || err.message || 'An error occurred.');
          setStatusMessage('');
          setLoading(false);
          setActiveJobId(null);
          return;
        }

        await sleep(2000);
      }

      if (!cancelled) {
        setError(
          'Timed out after 20 minutes. Check the server terminal — the job may still be running.'
        );
        setStatusMessage('');
        setLoading(false);
        setActiveJobId(null);
      }
    };

    poll();
    return () => {
      cancelled = true;
    };
  }, [activeJobId]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (loading) return;

    setError(null);
    setSummary('');
    setQuiz('');
    setUid('');
    setDone(false);

    if (!videoFile && !youtubeUrl.trim()) {
      setError('Choose a video file or paste a YouTube URL.');
      return;
    }

    setLoading(true);
    setStatusMessage('Submitting…');

    const formData = new FormData();
    if (videoFile) formData.append('video_file', videoFile);
    if (youtubeUrl.trim()) formData.append('youtube_url', youtubeUrl.trim());
    formData.append('level', difficulty);

    try {
      const res = await axios.post('/process', formData);
      const jobId = res.data.job_id;
      if (!jobId) {
        throw new Error('Server did not return a job id.');
      }
      setActiveJobId(jobId);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.error || err.message || 'An error occurred.');
      setStatusMessage('');
      setLoading(false);
    }
  };

  const downloadFile = async (type) => {
    if (!uid) return;
    try {
      const res = await axios.get(`/download/${uid}/${type}`, {
        responseType: 'blob',
      });
      const url = window.URL.createObjectURL(new Blob([res.data], { type: 'text/plain' }));
      const link = document.createElement('a');
      link.href = url;
      link.download = `${type}.txt`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error(err);
      setError(`Failed to download ${type}. Try again.`);
    }
  };

  const renderDownloadButtons = () => {
    const types = ['summary', 'quiz', 'transcript'];

    return (
      <div className="mt-8 flex flex-wrap justify-center gap-4">
        {types.map((type) => (
          <button
            key={type}
            type="button"
            onClick={() => downloadFile(type)}
            className="bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 text-white font-medium px-6 py-2 rounded-full shadow transition-transform transform hover:scale-105"
          >
            📥 Download {type.charAt(0).toUpperCase() + type.slice(1)}
          </button>
        ))}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-blue-100 p-6 flex flex-col items-center text-center">
      <img
        src={neuroBriefLogo}
        alt="NeuroBrief Logo"
        className="w-32 h-32 md:w-48 md:h-48 mb-4 object-contain"
      />

      <h1 className="text-3xl md:text-4xl font-bold text-blue-700 mb-2">🧠 NeuroBrief</h1>

      <p className="text-gray-700 text-sm md:text-base max-w-2xl mb-6 px-4">
        Upload a video or paste a YouTube link. Status updates show while each step runs.
      </p>

      <form
        onSubmit={handleSubmit}
        className="w-full max-w-lg bg-white p-6 rounded-xl shadow-md mb-4 space-y-4"
      >
        <div className="text-left">
          <label className="block font-medium text-gray-700 mb-1">Upload Video</label>
          <input
            type="file"
            accept="video/*"
            onChange={(e) => setVideoFile(e.target.files[0] || null)}
            className="w-full border rounded px-3 py-2"
          />
        </div>

        <div className="text-left">
          <label className="block font-medium text-gray-700 mb-1">Or YouTube URL</label>
          <input
            type="url"
            value={youtubeUrl}
            onChange={(e) => setYoutubeUrl(e.target.value)}
            placeholder="https://www.youtube.com/watch?v=..."
            className="w-full border rounded px-3 py-2"
          />
        </div>

        <div className="text-left">
          <label className="block font-medium text-gray-700 mb-1">Quiz Difficulty</label>
          <select
            value={difficulty}
            onChange={(e) => setDifficulty(e.target.value)}
            className="w-full border rounded px-3 py-2"
          >
            <option value="easy">Easy</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 w-full disabled:opacity-50"
        >
          {loading ? 'Working…' : 'Generate Summary & Quiz'}
        </button>
      </form>

      {done && (
        <div className="w-full max-w-lg mb-4 p-4 bg-green-50 border border-green-300 rounded-lg text-left">
          <p className="font-semibold text-green-800">✓ Finished processing</p>
          <p className="text-sm text-green-700 mt-1">Results are below. The server is idle — not stuck.</p>
        </div>
      )}

      {loading && statusMessage && !done && (
        <div className="w-full max-w-lg mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg text-left">
          <p className="font-semibold text-blue-800">{statusMessage}</p>
          <p className="text-sm text-gray-600 mt-1">
            {STAGE_HINTS[stage] || 'The server is processing. Watch the terminal for step logs.'}
          </p>
        </div>
      )}

      {error && <p className="text-red-600 mb-4 max-w-lg">{error}</p>}

      <div id="results" className="w-full max-w-3xl">
      {summary && (
        <div className="w-full max-w-3xl bg-white p-6 rounded-lg shadow-md mb-6 text-left">
          <h2 className="text-2xl font-semibold text-blue-600 mb-2">📄 Summary</h2>
          <p className="text-gray-800 whitespace-pre-line leading-relaxed">{summary}</p>
        </div>
      )}

      {quiz && (
        <div id="quiz" className="w-full max-w-3xl bg-white p-6 rounded-lg shadow-md text-left">
          <h2 className="text-2xl font-semibold text-purple-600 mb-4">🧠 Quiz</h2>
          <QuizPanel quizText={quiz} />
          {uid && renderDownloadButtons()}
        </div>
      )}
      </div>
    </div>
  );
}

export default App;
