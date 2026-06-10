import { TYPE_BADGE } from '../../constants';

export function normalizeQuestionType(raw) {
  const t = (raw || '').toLowerCase().replace(/\*\*/g, '').trim();
  if (!t) return { label: 'Question', badge: 'default' };
  if (t.includes('mcq') || t.includes('multiple choice')) {
    return { label: 'Multiple Choice', badge: 'mcq' };
  }
  if (t.includes('true') || t.includes('false') || t.includes('t/f')) {
    return { label: 'True or False', badge: 'tf' };
  }
  if (t.includes('fill')) {
    return { label: 'Fill in the Blank', badge: 'fill' };
  }
  return { label: raw.replace(/\*\*/g, '').trim(), badge: 'default' };
}

export function getBadgeClass(badge) {
  return TYPE_BADGE[badge] || TYPE_BADGE.default;
}
