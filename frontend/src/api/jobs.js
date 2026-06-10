import client from './client';

export async function submitJob(formData) {
  const res = await client.post('/process', formData);
  return res.data;
}

export async function fetchJob(jobId) {
  const res = await client.get(`/api/jobs/${jobId}`, {
    headers: { 'Cache-Control': 'no-cache' },
  });
  return res.data;
}

export async function downloadArtifact(uid, type) {
  const res = await client.get(`/download/${uid}/${type}`, {
    responseType: 'blob',
  });
  return res.data;
}
