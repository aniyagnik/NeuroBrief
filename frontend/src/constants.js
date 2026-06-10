export const STAGE_HINTS = {
  pending: 'Waiting to start…',
  downloading: 'Fetching audio from YouTube…',
  extracting: 'Pulling audio from your video…',
  transcribing: 'Transcribing audio — often 2–10 minutes for long videos.',
  summarizing: 'Generating summary and quiz…',
};

export const DOWNLOAD_TYPES = ['summary', 'quiz', 'transcript'];

export const TYPE_BADGE = {
  mcq: 'bg-blue-100 text-blue-800 border-blue-200',
  tf: 'bg-green-100 text-green-800 border-green-200',
  fill: 'bg-amber-100 text-amber-800 border-amber-200',
  default: 'bg-purple-100 text-purple-800 border-purple-200',
};
