export default function SummaryCard({ summary }) {
  if (!summary) return null;

  return (
    <div className="w-full max-w-3xl bg-white p-6 rounded-lg shadow-md mb-6 text-left">
      <h2 className="text-2xl font-semibold text-blue-600 mb-2">📄 Summary</h2>
      <p className="text-gray-800 whitespace-pre-line leading-relaxed">{summary}</p>
    </div>
  );
}
