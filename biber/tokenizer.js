/*
    Something here
*/


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

const MERGE_QUOTES = merge_chars(QUOTES);
const GROUP_QUOTES = group_chars(QUOTES);
const MERGE_PUNCTS = merge_chars(PUNCTS);
const MERGE_HYPHENS = merge_chars(HYPHENS);
const MERGE_CURRENCY = merge_chars(CURRENCY);

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
    `^(?:${LIST_PREFIXES.join('|')})`
);

const LIST_SUFFIXES = [
    ...LIST_PUNCT,
    ...LIST_ELLIPSES,
    ...LIST_QUOTES,
    "'s", "'S", "’s", "’S", "—", "–",
    String.raw`(?<=[0-9])\+`,
    String.raw`(?<=°[FfCcKk])\.`,
    String.raw`(?<=[0-9])(?:${MERGE_CURRENCY})`,
    String.raw`(?<=[0-9${ALPHA_LOWER}%²\-\+${MERGE_PUNCTS}(?:${GROUP_QUOTES})])\.`,
    String.raw`(?<=[${ALPHA_UPPER}][${ALPHA_UPPER}])\.`,
];

const SUFFIX_RE = new RegExp(
    `(?:${LIST_SUFFIXES.join('|')})$`
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

function tokenize(text) {
    const doc = new Doc(text);

    if (!text || text.length === 0) return doc;

    let i = 0;
    let start = 0;
    let inWs = /\s/.test(text[0]);

    for (const c of text) {
        if (/\s/.test(c) !== inWs) {
            if (start < i) {
                const span = text.substring(start, i);
                tokenizeSpan(doc, span, start);
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
        i++;
    }

    if (start < i) {
        const span = text.substring(start);
        tokenizeSpan(doc, span, start);
        if (doc.tokens.length > 0) {
            doc.lastToken().spacy = (text[text.length - 1] === ' ' && !inWs);
        }
    }

    return doc;
}

function tokenizeSpan(doc, span, baseOffset) {
    const { prefixes, coreSt, coreEnd, suffixes } = splitAffixes(span, baseOffset);
    attachTokens(doc, prefixes, coreSt, coreEnd, suffixes);
}

function splitAffixes(str, baseOffset) {
    const prefixes = [];
    const suffixes = [];
    let st = baseOffset;
    let end = baseOffset + str.length;
    let lastSize = 0;

    while (st < end && (end - st) !== lastSize) {
        lastSize = end - st;
        const chunk = str.substring(st - baseOffset, end - baseOffset);
        const preLen = findPrefix(chunk);

        // suffix after prefix is removed
        let sufLen = findSuffix(chunk.substring(preLen));

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

function attachTokens(doc, prefixes, coreSt, coreEnd, suffixes) {
    for (const [s, e] of prefixes) {
        doc.addToken(s, e);
    }

    if (coreSt < coreEnd) {
        const core = doc.text.substring(coreSt, coreEnd);
        if (URL_MATCH.test(core)) { // TODO: fix later
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


