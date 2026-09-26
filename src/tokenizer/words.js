/*
    Tokenizer.
    Port of spaCy's tokenizer for the English language.
    Regex patterns are mostly identical to spacy's.
    Exceptions (Ph.D., Dr., don't -> do n't, emoticons) come from spaCy's
    tokenizer rules and are applied at the same points spaCy v3 applies them.
*/

import { EXCEPTIONS } from "./exceptions.js";

const merge_chars = (s) => s.trim().replaceAll(" ", "|");
const split_chars = (s) => s.trim().split(" ");
const group_chars = (s) => s.trim().replaceAll(" ", "");

const ALPHA = "A-Za-z";
const ALPHA_LOWER = "a-z";
const ALPHA_UPPER = "A-Z";
const QUOTES = String.raw`\' " ” “ ` + "` ‘ ´ ’ ‚ , „ » « 「 」 『 』 （ ） 〔 〕 【 】 《 》 〈 〉 〈 〉  ⟦ ⟧";
const PUNCTS = String.raw`… …… , : ; \! \? ¿ ؟ ¡ \( \) \[ \] \{ \} < > _ # \* & 。 ？ ！ ， 、 ； ： ～ · । ، ۔ ؛ ٪`;
const HYPHENS = "- – — -- --- —— ~";
const CURRENCY = String.raw`\$ £ € ¥ ฿ US\$ C\$ A\$ ₽ ﷼ ₴ ₠ ₡ ₢ ₣ ₤ ₥ ₦ ₧ ₨ ₩ ₪ ₫ € ₭ ₮ ₯ ₰ `
    + "₱ ₲ ₳ ₴ ₵ ₶ ₷ ₸ ₹ ₺ ₻ ₼ ₽ ₾ ₿";
// Copied verbatim from spaCy's char_classes._units, including its missing
// space between "тб" and "كم".
const UNITS = "km km² km³ m m² m³ dm dm² dm³ cm cm² cm³ mm mm² mm³ ha µm nm yd in ft "
    + "kg g mg µg t lb oz m/s km/h kmh mph hPa Pa mbar mb MB kb KB gb GB tb "
    + "TB T G M K % км км² км³ м м² м³ дм дм² дм³ см см² см³ мм мм² мм³ нм "
    + "кг г мг м/с км/ч кПа Па мбар Кб КБ кб Мб МБ мб Гб ГБ гб Тб ТБ тб"
    + "كم كم² كم³ م م² م³ سم سم² سم³ مم مم² مم³ كم غرام جرام جم كغ ملغ كوب اكواب";

const MERGE_QUOTES = merge_chars(QUOTES);
const GROUP_QUOTES = group_chars(QUOTES);
const MERGE_PUNCTS = merge_chars(PUNCTS);
const MERGE_HYPHENS = merge_chars(HYPHENS);
const MERGE_CURRENCY = merge_chars(CURRENCY);
const MERGE_UNITS = merge_chars(UNITS);

// spaCy's compile_prefix_regex/compile_suffix_regex skip blank entries. The
// double space in QUOTES yields one; left in, it becomes an empty alternative
// that matches before "\$" and the other currency prefixes are tried.
const nonBlank = (pieces) => pieces.filter((p) => p.trim());

const LIST_ELLIPSES = [String.raw`\.\.+`, "…"];

const LIST_INFIX_RE = [
    ...(LIST_ELLIPSES.map(ele => new RegExp(ele))),
    new RegExp(`(?<=[0-9])[+\\-\\*^](?=[0-9-])`),
    new RegExp(`(?<=[${ALPHA_LOWER}${GROUP_QUOTES}])\\.(?=[${ALPHA_UPPER}${GROUP_QUOTES}])`),
    new RegExp(`(?<=[${ALPHA}]),(?=[${ALPHA}])`),
    new RegExp(`(?<=[${ALPHA}0-9])(?:${MERGE_HYPHENS})(?=[${ALPHA}])`),
    new RegExp(`(?<=[${ALPHA}0-9])[:<>=/](?=[${ALPHA}])`),
];

const INFIX_RE = new RegExp(
    LIST_INFIX_RE.map(r => r.source).join('|'),
    'g'
);

const LIST_PUNCT = split_chars(PUNCTS);
const LIST_QUOTES = split_chars(QUOTES);
const LIST_CURRENCY = split_chars(CURRENCY);

const LIST_PREFIXES = [
    ...["§", "%", "=", "—", "–", String.raw`\+(?![0-9])`],
    ...LIST_PUNCT,
    ...LIST_ELLIPSES,
    ...LIST_QUOTES,
    ...LIST_CURRENCY,
]

const PREFIX_RE = new RegExp(
    `^(?:${nonBlank(LIST_PREFIXES).join('|')})`
);

const LIST_SUFFIXES = [
    ...LIST_PUNCT,
    ...LIST_ELLIPSES,
    ...LIST_QUOTES,
    "'s", "'S", "’s", "’S", "—", "–",
    String.raw`(?<=[0-9])\+`,
    String.raw`(?<=°[FfCcKk])\.`,
    String.raw`(?<=[0-9])(?:${MERGE_CURRENCY})`,
    String.raw`(?<=[0-9])(?:${MERGE_UNITS})`,
    String.raw`(?<=[0-9${ALPHA_LOWER}%²\-\+${MERGE_PUNCTS}(?:${GROUP_QUOTES})])\.`,
    String.raw`(?<=[${ALPHA_UPPER}][${ALPHA_UPPER}])\.`,
];

const SUFFIX_RE = new RegExp(
    `(?:${nonBlank(LIST_SUFFIXES).join('|')})$`
);

const URL_MATCH = /http:\/\//
const TEXT_ENCODER = new TextEncoder();

class Token {
    constructor(start, end, spacy = false) {
        this.start = start;
        this.end = end;
        this.spacy = spacy;
    }
}

class Doc {
    constructor(text) {
        this.text = text;
        this.tokens = [];
    }

    addToken(start, end, spacy = false) {
        this.tokens.push(new Token(start, end, spacy));
    }

    getText(token) {
        return this.text.substring(token.start, token.end);
    }

    lastToken() {
        return this.tokens[this.tokens.length - 1];
    }

    toBuffer() {
        const utf8Text = TEXT_ENCODER.encode(this.text);
        const charToByteOffset = this._buildOffsetMap();
        const tokenCount = this.tokens.length;
        const totalSize = 4 + utf8Text.length + 4 + (tokenCount * 12);
        const buffer = new ArrayBuffer(totalSize);
        const view = new DataView(buffer);
        let offset = 0;

        view.setUint32(offset, utf8Text.length, true);
        offset += 4;
        new Uint8Array(buffer, offset, utf8Text.length).set(utf8Text);
        offset += utf8Text.length;

        view.setUint32(offset, tokenCount, true);
        offset += 4;

        for (const token of this.tokens) {
            view.setUint32(offset, charToByteOffset[token.start], true);
            offset += 4;
            view.setUint32(offset, charToByteOffset[token.end], true);
            offset += 4;
            view.setUint32(offset, token.spacy ? 1 : 0, true);
            offset += 4;
        }

        return buffer;
    }

    _buildOffsetMap() {
        const map = new Uint32Array(this.text.length + 1);
        let byteOffset = 0;

        for (let i = 0; i < this.text.length; i++) {
            map[i] = byteOffset;
            const code = this.text.codePointAt(i);

            if (code <= 0x7f) {
                byteOffset += 1;
            } else if (code <= 0x7ff) {
                byteOffset += 2;
            } else if (code <= 0xffff) {
                byteOffset += 3;
            } else {
                byteOffset += 4;
                i++;
                map[i] = byteOffset;
            }
        }

        map[this.text.length] = byteOffset;
        return map;
    }
}

export function tokenize(text) {
    const doc = tokenizeWith(text, true);
    applySpecialCases(doc);
    return doc;
}

// withSpecials = false is spaCy's affix-only tokenization, used to build the
// token patterns that applySpecialCases looks for.
function tokenizeWith(text, withSpecials) {
    const doc = new Doc(text);

    if (!text || text.length === 0) return doc;

    let i = 0;
    let start = 0;
    let inWs = /\s/.test(text[0]);

    for (const c of text) {
        if (/\s/.test(c) !== inWs) {
            if (start < i) {
                const span = text.substring(start, i);
                tokenizeSpan(doc, span, start, withSpecials);
            }
            if (c === ' ') {
                if (doc.tokens.length > 0) {
                    doc.lastToken().spacy = true;
                }
                start = i + 1;
            } else {
                start = i;
            }
            inWs = !inWs;
        }
        // for...of yields code points; astral ones (emoji) are 2 UTF-16 units,
        // and offsets index UTF-16 units.
        i += c.length;
    }

    if (start < i) {
        const span = text.substring(start);
        tokenizeSpan(doc, span, start, withSpecials);
        if (doc.tokens.length > 0) {
            doc.lastToken().spacy = (text[text.length - 1] === ' ' && !inWs);
        }
    }

    return doc;
}

const isSpecial = (s) => Object.hasOwn(EXCEPTIONS, s);

// Emit an exception's pieces as consecutive tokens starting at offset st.
function addSpecial(doc, st, key) {
    for (const piece of EXCEPTIONS[key]) {
        doc.addToken(st, st + piece.length);
        st += piece.length;
    }
}

function tokenizeSpan(doc, span, baseOffset, withSpecials) {
    if (withSpecials && isSpecial(span)) {
        addSpecial(doc, baseOffset, span);
        return;
    }
    const { prefixes, coreSt, coreEnd, suffixes } = splitAffixes(span, baseOffset, withSpecials);
    attachTokens(doc, prefixes, coreSt, coreEnd, suffixes, withSpecials);
}

/** Find and strip prefixes and suffixes layer by layer
 *  While loop strips prefixes and suffixes like layers of an onion.
*/
function splitAffixes(str, baseOffset, withSpecials) {
    const prefixes = [];
    const suffixes = [];
    let st = baseOffset;
    let end = baseOffset + str.length;
    let lastSize = 0;
    const current = () => str.substring(st - baseOffset, end - baseOffset);

    // lastSize is used as a stop condition,
    // i.e. we stop when there are no new prefixes/suffixes found
    while (st < end && (end - st) !== lastSize) {
        // Stop peeling once what's left is an exception, e.g. "(Dr." -> "(" "Dr."
        if (withSpecials && isSpecial(current())) break;
        lastSize = end - st;
        const chunk = current();
        const preLen = findPrefix(chunk);

        if (withSpecials && preLen && preLen < chunk.length
            && isSpecial(chunk.substring(preLen))) {
            prefixes.push([st, st + preLen]);
            st += preLen;
            break;
        }

        // suffix after prefix is removed
        let sufLen = findSuffix(chunk.substring(preLen));

        // Like spaCy, the suffix is tested against the chunk with the prefix
        // still attached.
        if (withSpecials && sufLen && sufLen < chunk.length
            && isSpecial(chunk.substring(0, chunk.length - sufLen))) {
            suffixes.push([end - sufLen, end]);
            end -= sufLen;
            break;
        }

        if (preLen && sufLen && (preLen + sufLen) <= end - st) {
            prefixes.push([st, st + preLen]);
            suffixes.push([end - sufLen, end]);
            st += preLen;
            end -= sufLen;
        } else if (preLen) {
            prefixes.push([st, st + preLen]);
            st += preLen;
        } else if (sufLen) {
            suffixes.push([end - sufLen, end]);
            end -= sufLen;
        }
    }

    return { prefixes, coreSt: st, coreEnd: end, suffixes };
}

function findPrefix(string) {
    const match = string.match(PREFIX_RE);
    return match ? match[0].length : 0;
}

function findSuffix(string) {
    const match = string.match(SUFFIX_RE);
    return match ? match[0].length : 0;
}

// With start and end offsets identified, attach info to each token
function attachTokens(doc, prefixes, coreSt, coreEnd, suffixes, withSpecials) {
    for (const [s, e] of prefixes) {
        doc.addToken(s, e);
    }

    if (coreSt < coreEnd) {
        const core = doc.text.substring(coreSt, coreEnd);
        if (withSpecials && isSpecial(core)) {
            addSpecial(doc, coreSt, core);
        } else if (URL_MATCH.test(core)) { // TODO: fix later
            doc.addToken(coreSt, coreEnd);
        } else {
            splitInfixes(doc, coreSt, coreEnd);
        }
    }
    //  suffixes should be assembled from right to left (reverse)
    for (let i = suffixes.length - 1; i >= 0; i--) {
        doc.addToken(suffixes[i][0], suffixes[i][1]);
    }
}

function splitInfixes(doc, st, end) {
    if (st >= end) return;

    const core = doc.text.substring(st, end);
    const matches = [...core.matchAll(INFIX_RE)];
    if (matches.length === 0) {
        doc.addToken(st, end);
        return;
    }

    let offset = 0;
    for (const match of matches) {
        if (offset === 0 && match.index === 0) continue;

        if (match.index > offset) {
            doc.addToken(st + offset, st + match.index);
        }
        // zero-width lookahead/lookbehind re infix patterns to split
        // without consuming the chars.
        // (acc. to spacy, empty infix tokens are useful e.g. split ab12 into ab 12)
        if (match[0].length > 0) { // split was on non-empty chars
            doc.addToken(st + match.index, st + match.index + match[0].length);
        }
        offset = match.index + match[0].length;
    }

    // remaining chunk
    if (offset < core.length) {
        doc.addToken(st + offset, end);
    }
}

/*  spaCy's second pass (Tokenizer._apply_special_cases). Exceptions that
    contain affix characters, like "):" or ":))", only show up after affix
    splitting has already broken them apart ("3):" -> "3" ")" ":"). Each such
    exception's affix-only tokenization is a pattern; runs of adjacent tokens
    with no whitespace between them that match a pattern are replaced by the
    exception's pieces. Longest matches win, then leftmost, without overlap.
*/
const SPECIAL_PATTERNS = new Map(); // first token text -> [{ texts, key }]
for (const key of Object.keys(EXCEPTIONS)) {
    const doc = tokenizeWith(key, false);
    const texts = doc.tokens.map((t) => doc.getText(t));
    if (texts.length === 1 && texts[0] === key) continue;
    if (!SPECIAL_PATTERNS.has(texts[0])) SPECIAL_PATTERNS.set(texts[0], []);
    SPECIAL_PATTERNS.get(texts[0]).push({ texts, key });
}

function applySpecialCases(doc) {
    const toks = doc.tokens;
    const texts = toks.map((t) => doc.getText(t));
    const matches = [];
    for (let i = 0; i < toks.length; i++) {
        for (const { texts: pat, key } of SPECIAL_PATTERNS.get(texts[i]) ?? []) {
            const n = pat.length;
            if (i + n > toks.length) continue;
            let ok = true;
            for (let k = 0; k < n && ok; k++) {
                ok = texts[i + k] === pat[k]
                    && (k === 0 || toks[i + k - 1].end === toks[i + k].start);
            }
            if (ok) matches.push({ i, n, key });
        }
    }
    if (matches.length === 0) return;

    matches.sort((a, b) => b.n - a.n || a.i - b.i);
    const taken = new Uint8Array(toks.length);
    const chosen = [];
    for (const m of matches) {
        let free = true;
        for (let k = m.i; k < m.i + m.n && free; k++) free = !taken[k];
        if (!free) continue;
        taken.fill(1, m.i, m.i + m.n);
        chosen.push(m);
    }
    chosen.sort((a, b) => a.i - b.i);

    const out = [];
    let next = 0;
    for (const { i, n, key } of chosen) {
        out.push(...toks.slice(next, i));
        let st = toks[i].start;
        for (const piece of EXCEPTIONS[key]) {
            out.push(new Token(st, st + piece.length));
            st += piece.length;
        }
        out[out.length - 1].spacy = toks[i + n - 1].spacy;
        next = i + n;
    }
    out.push(...toks.slice(next));
    doc.tokens = out;
}
