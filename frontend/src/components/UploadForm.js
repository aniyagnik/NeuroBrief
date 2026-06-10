export default function UploadForm({
  videoFile,
  youtubeUrl,
  difficulty,
  loading,
  onVideoChange,
  onYoutubeChange,
  onDifficultyChange,
  onSubmit,
}) {
  return (
    <form
      onSubmit={onSubmit}
      className="w-full max-w-lg bg-white p-6 rounded-xl shadow-md mb-4 space-y-4"
    >
      <div className="text-left">
        <label className="block font-medium text-gray-700 mb-1">Upload Video</label>
        <input
          type="file"
          accept="video/*"
          onChange={(e) => onVideoChange(e.target.files[0] || null)}
          className="w-full border rounded px-3 py-2"
        />
      </div>

      <div className="text-left">
        <label className="block font-medium text-gray-700 mb-1">Or YouTube URL</label>
        <input
          type="url"
          value={youtubeUrl}
          onChange={(e) => onYoutubeChange(e.target.value)}
          placeholder="https://www.youtube.com/watch?v=..."
          className="w-full border rounded px-3 py-2"
        />
      </div>

      <div className="text-left">
        <label className="block font-medium text-gray-700 mb-1">Quiz Difficulty</label>
        <select
          value={difficulty}
          onChange={(e) => onDifficultyChange(e.target.value)}
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
  );
}
