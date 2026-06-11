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
  return { label: raw.replace(/\*\*/g, '').trim(), badge: 'default' };
}

function isTrueFalseAnswer(answer) {
  const a = (answer || '').toLowerCase().trim();
  return a === 'true' || a === 'false' || a.startsWith('true') || a.startsWith('false');
}

export function resolveQuestionType(question) {
  const fromType = normalizeQuestionType(question.type);
  if (fromType.badge !== 'default') return fromType;

  if (question.options?.length > 0) {
    return { label: 'Multiple Choice', badge: 'mcq' };
  }

  if (isTrueFalseAnswer(question.answer)) {
    return { label: 'True or False', badge: 'tf' };
  }

  if (!question.options?.length) {
    return { label: 'True or False', badge: 'tf' };
  }

  return fromType;
}

export function getDisplayOptions(question) {
  const { badge } = resolveQuestionType(question);
  if (question.options?.length > 0) return question.options;
  if (badge === 'tf' || badge === 'default') return ['True', 'False'];
  return [];
}

export function getBadgeClass(badge) {
  return TYPE_BADGE[badge] || TYPE_BADGE.default;
}
