import { tokenize } from "./words.js";

// Like spaCy's rule-based sentencizer: a token made only of sentence-final
// punctuation ends a sentence. Periods that belong to a token ("Dr.", "Ph.D.",
// "e.g.") never do, since the tokenizer keeps them attached. The same holds for
// a sentence-final "U.S.", which is a known miss.
const TERMINAL = /^[.!?…。？！]+$/;
// Closing quotes and brackets right after the terminal stay in that sentence:
// He said "stop." Then ...
const CLOSING = /^[)\]}"'”’»」』]+$/;
const PARAGRAPH_BREAK = /\n\s*\n/;

// Paragraph text -> array of sentences, each an array of word strings.
// Whitespace tokens are dropped; a blank line also ends a sentence.
export function splitSentences(text) {
  const doc = tokenize(text);
  const sentences = [];
  let current = [];
  let closing = false;

  const flush = () => {
    if (current.length) sentences.push(current);
    current = [];
    closing = false;
  };

  for (const tok of doc.tokens) {
    const word = doc.getText(tok);
    if (!word.trim()) {
      if (closing || PARAGRAPH_BREAK.test(word)) flush();
      continue;
    }
    if (closing && !TERMINAL.test(word) && !CLOSING.test(word)) flush();
    current.push(word);
    if (TERMINAL.test(word)) closing = true;
  }
  flush();
  return sentences;
}
