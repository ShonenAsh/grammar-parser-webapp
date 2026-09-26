// Must match shape_of in util/prepare_data.py, which the model was trained
// with. That function is built on Python's str.isdigit/isupper/istitle/islower,
// so those are reproduced here with the same Unicode semantics. In particular a
// string with no cased letters (punctuation, "5.00") is neither lower nor
// upper and falls through to OTHER.

const TITLE = 0, UPPER = 1, LOWER = 2, DIGIT = 3, OTHER = 4;

// Python's isdigit: Numeric_Type Decimal (\p{Nd}) or Digit. The second class
// lists the Digit-only characters (superscripts, circled digits, ...),
// generated from Python 3.13 / Unicode 15.1.
const DIGITS = new RegExp(
  "^[\\p{Nd}\\u{B2}-\\u{B3}\\u{B9}\\u{1369}-\\u{1371}\\u{19DA}\\u{2070}\\u{2074}-\\u{2079}"
  + "\\u{2080}-\\u{2089}\\u{2460}-\\u{2468}\\u{2474}-\\u{247C}\\u{2488}-\\u{2490}\\u{24EA}"
  + "\\u{24F5}-\\u{24FD}\\u{24FF}\\u{2776}-\\u{277E}\\u{2780}-\\u{2788}\\u{278A}-\\u{2792}"
  + "\\u{10A40}-\\u{10A43}\\u{10E60}-\\u{10E68}\\u{11052}-\\u{1105A}\\u{1F100}-\\u{1F10A}]+$",
  "u",
);
const ALPHA = /\p{L}/u;
const IS_LOWER = /\p{Lowercase}/u;
const IS_UPPER = /\p{Uppercase}/u;
const IS_TITLE = /\p{Lt}/u;

// The three case checks follow CPython's unicode_islower_impl etc.
function isLower(word) {
  let cased = false;
  for (const ch of word) {
    if (IS_UPPER.test(ch) || IS_TITLE.test(ch)) return false;
    if (IS_LOWER.test(ch)) cased = true;
  }
  return cased;
}

function isUpper(word) {
  let cased = false;
  for (const ch of word) {
    if (IS_LOWER.test(ch) || IS_TITLE.test(ch)) return false;
    if (IS_UPPER.test(ch)) cased = true;
  }
  return cased;
}

function isTitle(word) {
  let cased = false, prevCased = false;
  for (const ch of word) {
    if (IS_UPPER.test(ch) || IS_TITLE.test(ch)) {
      if (prevCased) return false;
      prevCased = cased = true;
    } else if (IS_LOWER.test(ch)) {
      if (!prevCased) return false;
      prevCased = cased = true;
    } else {
      prevCased = false;
    }
  }
  return cased;
}

export function shapeOf(word) {
  if (DIGITS.test(word)) return DIGIT;
  if (isUpper(word) && ALPHA.test(word)) return UPPER;
  if (isTitle(word)) return TITLE;
  if (isLower(word)) return LOWER;
  return OTHER;
}
