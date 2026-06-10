function normalizeAnswer(text) {
  return (text || '').toLowerCase().replace(/\s+/g, ' ').trim();
}

export function answersMatch(selected, correct) {
  const s = normalizeAnswer(selected);
  const c = normalizeAnswer(correct);
  if (!s || !c) return false;
  if (s === c) return true;

  const selectedLetter = s.match(/^([a-d])[).]/);
  const correctLetter = c.match(/^([a-d])[).]/);
  if (selectedLetter && correctLetter && selectedLetter[1] === correctLetter[1]) {
    return true;
  }
  if (/^[a-d]$/.test(c) && selectedLetter && selectedLetter[1] === c) {
    return true;
  }

  if ((s === 'true' || s === 'false') && c.startsWith(s)) {
    return true;
  }

  return false;
}
