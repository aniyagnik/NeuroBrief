import { STAGE_HINTS } from '../constants';

export default function StatusBanner({ done, loading, statusMessage, stage, error }) {
  if (error) {
    return <p className="text-red-600 mb-4 max-w-lg">{error}</p>;
  }

  if (done) {
    return (
      <div className="w-full max-w-lg mb-4 p-4 bg-green-50 border border-green-300 rounded-lg text-left">
        <p className="font-semibold text-green-800">✓ Finished processing</p>
        <p className="text-sm text-green-700 mt-1">
          Results are below. The server is idle — not stuck.
        </p>
      </div>
    );
  }

  if (loading && statusMessage) {
    return (
      <div className="w-full max-w-lg mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg text-left">
        <p className="font-semibold text-blue-800">{statusMessage}</p>
        <p className="text-sm text-gray-600 mt-1">
          {STAGE_HINTS[stage] || 'The server is processing. Watch the terminal for step logs.'}
        </p>
      </div>
    );
  }

  return null;
}
