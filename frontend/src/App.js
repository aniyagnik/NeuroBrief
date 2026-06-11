import { useEffect, useState } from 'react';
import axios from 'axios';
import neuroBriefLogo from './logo.png';
import QuizPanel from './quiz';

axios.defaults.timeout = 30000;

const STAGE_HINTS = {
  pending: 'Waiting to start…',
  downloading: 'Fetching audio from YouTube…',
  extracting: 'Pulling audio from your video…',
  transcribing: 'Transcribing audio — often 2–10 minutes for long videos.',
  summarizing: 'Generating summary and quiz…',
};

export default function App() {
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
  const [jobId, setJobId] = useState(null);

  useEffect(() => {
    if (!jobId) return undefined;
    let stop = false;

    (async () => {
      const end = Date.now() + 20 * 60 * 1000;
      while (!stop && Date.now() < end) {
        try {
          const { data: job } = await axios.get(`/api/jobs/${jobId}`, { headers: { 'Cache-Control': 'no-cache' } });
          setStage(job.stage || job.status || '');
          setStatusMessage(job.status_message || job.status || 'Working…');

          if (job.status === 'completed') {
            setSummary((job.summary || '').trimStart());
            setQuiz(job.quiz || '');
            setUid(job.uid || jobId);
            setDone(true);
            setLoading(false);
            setJobId(null);
            setTimeout(() => document.getElementById('results')?.scrollIntoView({ behavior: 'smooth' }), 100);
            return;
          }
          if (job.status === 'failed') throw new Error(job.error || 'Job failed');
        } catch (err) {
          if (stop) return;
          setError(err.response?.data?.error || err.message || 'An error occurred.');
          setLoading(false);
          setJobId(null);
          return;
        }
        await new Promise((r) => setTimeout(r, 2000));
      }
      if (!stop) {
        setError('Timed out after 20 minutes.');
        setLoading(false);
        setJobId(null);
      }
    })();

    return () => { stop = true; };
  }, [jobId]);

  const download = async (type) => {
    try {
      const res = await axios.get(`/download/${uid}/${type}`, { responseType: 'blob' });
      const url = URL.createObjectURL(new Blob([res.data], { type: 'text/plain' }));
      const a = document.createElement('a');
      a.href = url;
      a.download = `${type}.txt`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      setError(`Failed to download ${type}.`);
    }
  };

  const submit = async (e) => {
    e.preventDefault();
    if (loading) return;
    if (!videoFile && !youtubeUrl.trim()) {
      setError('Choose a video file or paste a YouTube URL.');
      return;
    }

    setError(null);
    setSummary('');
    setQuiz('');
    setUid('');
    setDone(false);
    setLoading(true);
    setStatusMessage('Submitting…');

    const form = new FormData();
    if (videoFile) form.append('video_file', videoFile);
    if (youtubeUrl.trim()) form.append('youtube_url', youtubeUrl.trim());
    form.append('level', difficulty);

    try {
      const { data } = await axios.post('/process', form);
      if (!data.job_id) throw new Error('No job id returned.');
      setJobId(data.job_id);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'An error occurred.');
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-blue-100 p-6 flex flex-col items-center text-center">
      <img src={neuroBriefLogo} alt="NeuroBrief" className="w-32 h-32 md:w-48 md:h-48 mb-4 object-contain" />
      <h1 className="text-3xl md:text-4xl font-bold text-blue-700 mb-2">🧠 NeuroBrief</h1>
      <p className="text-gray-700 text-sm md:text-base max-w-2xl mb-6 px-4">
        Upload a video or paste a YouTube link.
      </p>

      <form onSubmit={submit} className="w-full max-w-lg bg-white p-6 rounded-xl shadow-md mb-4 space-y-4 text-left">
        <div>
          <label className="block font-medium text-gray-700 mb-1">Upload Video</label>
          <input type="file" accept="video/*" onChange={(e) => setVideoFile(e.target.files[0] || null)} className="w-full border rounded px-3 py-2" />
        </div>
        <div>
          <label className="block font-medium text-gray-700 mb-1">Or YouTube URL</label>
          <input type="url" value={youtubeUrl} onChange={(e) => setYoutubeUrl(e.target.value)} placeholder="https://www.youtube.com/watch?v=..." className="w-full border rounded px-3 py-2" />
        </div>
        <div>
          <label className="block font-medium text-gray-700 mb-1">Quiz Difficulty</label>
          <select value={difficulty} onChange={(e) => setDifficulty(e.target.value)} className="w-full border rounded px-3 py-2">
            <option value="easy">Easy</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
        </div>
        <button type="submit" disabled={loading} className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 w-full disabled:opacity-50">
          {loading ? 'Working…' : 'Generate Summary & Quiz'}
        </button>
      </form>

      {done && (
        <div className="w-full max-w-lg mb-4 p-4 bg-green-50 border border-green-300 rounded-lg text-left">
          <p className="font-semibold text-green-800">✓ Finished processing</p>
        </div>
      )}
      {loading && statusMessage && !done && (
        <div className="w-full max-w-lg mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg text-left">
          <p className="font-semibold text-blue-800">{statusMessage}</p>
          <p className="text-sm text-gray-600 mt-1">{STAGE_HINTS[stage] || 'Processing…'}</p>
        </div>
      )}
      {error && <p className="text-red-600 mb-4 max-w-lg">{error}</p>}

      <div id="results" className="w-full max-w-3xl">
        {summary && (
          <div className="bg-white p-6 rounded-lg shadow-md mb-6 text-left">
            <h2 className="text-2xl font-semibold text-blue-600 mb-2">📄 Summary</h2>
            <p className="text-gray-800 whitespace-pre-line">{summary}</p>
          </div>
        )}
        {quiz && (
          <div id="quiz" className="bg-white p-6 rounded-lg shadow-md text-left">
            <h2 className="text-2xl font-semibold text-purple-600 mb-4">🧠 Quiz</h2>
            <QuizPanel text={quiz} />
            {uid && (
              <div className="mt-8 flex flex-wrap justify-center gap-4">
                {['summary', 'quiz', 'transcript'].map((type) => (
                  <button key={type} type="button" onClick={() => download(type)} className="bg-green-600 hover:bg-green-700 text-white font-medium px-6 py-2 rounded-full">
                    📥 Download {type.charAt(0).toUpperCase() + type.slice(1)}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
