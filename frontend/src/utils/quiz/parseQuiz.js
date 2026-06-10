function parseOptionsLine(text) {
  const trimmed = text.trim();
  if (!trimmed) return [];
  if (/,\s*[A-Da-d][).]/.test(trimmed)) {
    return trimmed.split(/,\s*(?=[A-Da-d][).])/).map((o) => o.trim()).filter(Boolean);
  }
  return [trimmed];
}

function extractQuizField(line, prefixes) {
  const trimmed = line.trimStart();
  for (const prefix of prefixes) {
    const re = new RegExp(`^${prefix}\\s*:?\\s*`, 'i');
    if (re.test(trimmed)) {
      return trimmed.replace(re, '').trim();
    }
  }
  return null;
}

export function parseQuizText(quizText) {
  const lines = quizText.trim().split('\n');
  const questions = [];
  let current = null;
  let pendingType = '';
  let id = 1;

  const flush = () => {
    if (current?.question) {
      questions.push({ ...current, id: id++ });
    }
    current = null;
  };

  for (let i = 0; i < lines.length; i++) {
    const trimmed = lines[i].trim();
    if (!trimmed) continue;
    if (/^here are (the )?quiz questions/i.test(trimmed)) continue;

    const mdHeader = trimmed.match(/^\*\*(.+?)\*\*$/);
    if (mdHeader) {
      pendingType = mdHeader[1].trim();
      continue;
    }

    const typeVal = extractQuizField(trimmed, ['Question Type', 'Type']);
    if (typeVal !== null) {
      flush();
      pendingType = typeVal;
      current = { type: typeVal, question: '', options: [], answer: '' };
      continue;
    }

    const questionVal = extractQuizField(trimmed, ['Question']);
    if (questionVal !== null) {
      if (!current) {
        current = { type: pendingType, question: '', options: [], answer: '' };
      }
      current.question = questionVal;
      if (!current.type && pendingType) current.type = pendingType;
      continue;
    }

    const optionsVal = extractQuizField(trimmed, ['Options']);
    if (optionsVal !== null) {
      if (!current) {
        current = { type: pendingType, question: '', options: [], answer: '' };
      }
      current.options.push(...parseOptionsLine(optionsVal));
      i += 1;
      while (i < lines.length) {
        const next = lines[i].trim();
        if (!next) {
          i += 1;
          continue;
        }
        if (extractQuizField(next, ['Answer', 'Question Type', 'Type', 'Question']) !== null) {
          i -= 1;
          break;
        }
        if (/^\*\*.+\*\*$/.test(next)) {
          i -= 1;
          break;
        }
        current.options.push(...parseOptionsLine(next));
        i += 1;
      }
      continue;
    }

    const answerVal = extractQuizField(trimmed, ['Answer']);
    if (answerVal !== null) {
      if (!current) {
        current = { type: pendingType, question: '', options: [], answer: '' };
      }
      current.answer = answerVal;
      flush();
      pendingType = '';
    }
  }

  flush();
  return questions;
}
