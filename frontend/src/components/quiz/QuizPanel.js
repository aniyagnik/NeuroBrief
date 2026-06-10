import { useEffect, useRef, useState } from 'react';
import { answersMatch } from '../../utils/quiz/answersMatch';
import { parseQuizText } from '../../utils/quiz/parseQuiz';
import QuizQuestion from './QuizQuestion';

export default function QuizPanel({ quizText }) {
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
