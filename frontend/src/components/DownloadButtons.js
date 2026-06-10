import { DOWNLOAD_TYPES } from '../constants';
import { downloadArtifact } from '../api/jobs';

export default function DownloadButtons({ uid, onError }) {
  if (!uid) return null;

  const handleDownload = async (type) => {
    try {
      const blob = await downloadArtifact(uid, type);
      const url = window.URL.createObjectURL(new Blob([blob], { type: 'text/plain' }));
      const link = document.createElement('a');
      link.href = url;
      link.download = `${type}.txt`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error(err);
      onError(`Failed to download ${type}. Try again.`);
    }
  };

  return (
    <div className="mt-8 flex flex-wrap justify-center gap-4">
      {DOWNLOAD_TYPES.map((type) => (
        <button
          key={type}
          type="button"
          onClick={() => handleDownload(type)}
          className="bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 text-white font-medium px-6 py-2 rounded-full shadow transition-transform transform hover:scale-105"
        >
          📥 Download {type.charAt(0).toUpperCase() + type.slice(1)}
        </button>
      ))}
    </div>
  );
}
