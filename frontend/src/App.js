import { useCallback, useState } from 'react';
import { submitJob } from './api/jobs';
import Header from './components/Header';
import UploadForm from './components/UploadForm';
import StatusBanner from './components/StatusBanner';
import ResultsSection from './components/ResultsSection';
import { useJobPolling } from './hooks/useJobPolling';

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

  const handleProgress = useCallback(({ stage: nextStage, statusMessage: nextMessage }) => {
    setStage(nextStage);
    setStatusMessage(nextMessage);
  }, []);

  const handleComplete = useCallback(({ summary: nextSummary, quiz: nextQuiz, uid: nextUid }) => {
    setSummary(nextSummary);
    setQuiz(nextQuiz);
    setUid(nextUid);
    setStatusMessage('Done! Scroll down for results.');
    setDone(true);
    setLoading(false);
    setActiveJobId(null);
    setTimeout(() => {
      document.getElementById('results')?.scrollIntoView({ behavior: 'smooth' });
    }, 100);
  }, []);

  const handlePollError = useCallback((message) => {
    setError(message);
    setStatusMessage('');
    setLoading(false);
    setActiveJobId(null);
  }, []);

  useJobPolling(activeJobId, {
    onProgress: handleProgress,
    onComplete: handleComplete,
    onError: handlePollError,
  });

  const resetResults = () => {
    setError(null);
    setSummary('');
    setQuiz('');
    setUid('');
    setDone(false);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (loading) return;

    resetResults();

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
      const data = await submitJob(formData);
      if (!data.job_id) {
        throw new Error('Server did not return a job id.');
      }
      setActiveJobId(data.job_id);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.error || err.message || 'An error occurred.');
      setStatusMessage('');
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-blue-100 p-6 flex flex-col items-center text-center">
      <Header />
      <UploadForm
        videoFile={videoFile}
        youtubeUrl={youtubeUrl}
        difficulty={difficulty}
        loading={loading}
        onVideoChange={setVideoFile}
        onYoutubeChange={setYoutubeUrl}
        onDifficultyChange={setDifficulty}
        onSubmit={handleSubmit}
      />
      <StatusBanner
        done={done}
        loading={loading}
        statusMessage={statusMessage}
        stage={stage}
        error={error}
      />
      <ResultsSection summary={summary} quiz={quiz} uid={uid} onError={setError} />
    </div>
  );
}

export default App;
