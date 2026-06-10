import { useEffect } from 'react';
import { fetchJob } from '../api/jobs';
import { sleep } from '../utils/sleep';

const POLL_INTERVAL_MS = 2000;
const TIMEOUT_MS = 20 * 60 * 1000;

export function useJobPolling(activeJobId, { onComplete, onError, onProgress }) {
  useEffect(() => {
    if (!activeJobId) return undefined;

    let cancelled = false;

    const poll = async () => {
      const deadline = Date.now() + TIMEOUT_MS;

      while (!cancelled && Date.now() < deadline) {
        try {
          const job = await fetchJob(activeJobId);
          if (cancelled) return;

          onProgress({
            stage: job.stage || job.status || '',
            statusMessage: job.status_message || job.status || 'Working…',
          });

          const status = String(job.status).toLowerCase();

          if (status === 'completed') {
            const summary = (job.summary || '').trimStart();
            const quiz = job.quiz || '';
            if (!summary && !quiz) {
              throw new Error('Job finished but the server returned no summary or quiz.');
            }
            onComplete({
              summary,
              quiz,
              uid: job.uid || activeJobId,
            });
            return;
          }

          if (status === 'failed') {
            throw new Error(job.error || 'Job failed');
          }
        } catch (err) {
          if (cancelled) return;
          onError(err.response?.data?.error || err.message || 'An error occurred.');
          return;
        }

        await sleep(POLL_INTERVAL_MS);
      }

      if (!cancelled) {
        onError(
          'Timed out after 20 minutes. Check the server terminal — the job may still be running.'
        );
      }
    };

    poll();
    return () => {
      cancelled = true;
    };
  }, [activeJobId, onComplete, onError, onProgress]);
}
