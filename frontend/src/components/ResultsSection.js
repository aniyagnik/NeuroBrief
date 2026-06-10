import SummaryCard from './SummaryCard';
import DownloadButtons from './DownloadButtons';
import QuizPanel from './quiz/QuizPanel';

export default function ResultsSection({ summary, quiz, uid, onError }) {
  if (!summary && !quiz) return null;

  return (
    <div id="results" className="w-full max-w-3xl">
      <SummaryCard summary={summary} />
      {quiz && (
        <div id="quiz" className="w-full max-w-3xl bg-white p-6 rounded-lg shadow-md text-left">
          <h2 className="text-2xl font-semibold text-purple-600 mb-4">🧠 Quiz</h2>
          <QuizPanel quizText={quiz} />
          <DownloadButtons uid={uid} onError={onError} />
        </div>
      )}
    </div>
  );
}
