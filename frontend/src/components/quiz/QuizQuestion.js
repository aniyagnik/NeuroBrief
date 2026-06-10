import { answersMatch } from '../../utils/quiz/answersMatch';
import { getBadgeClass, normalizeQuestionType } from '../../utils/quiz/questionTypes';

export default function QuizQuestion({ q, showAnswer, selected, onSelect }) {
  const { label, badge } = normalizeQuestionType(q.type);
  const badgeClass = getBadgeClass(badge);
  const isFillBlank = badge === 'fill';
  const displayOptions =
    q.options.length > 0 ? q.options : label === 'True or False' ? ['True', 'False'] : [];
  const isCorrect = showAnswer && answersMatch(selected, q.answer);

  return (
    <div className="mb-6 p-4 border border-gray-200 rounded-lg shadow-sm">
      <div className="flex flex-wrap items-center gap-2 mb-2">
        <span className="text-sm font-medium text-gray-500">Q{q.id}</span>
        <span className={`text-xs font-semibold px-2.5 py-0.5 rounded-full border ${badgeClass}`}>
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
                optionClass +=
                  ' bg-gray-50 border-gray-100 hover:border-purple-300 hover:bg-purple-50/50 cursor-pointer';
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
