const merge_chars = (s) => s.trim().replaceAll(" ", "|");
const split_chars = (s) => s.trim().split(" ");
const group_chars = (s) => s.trim().replaceAll(" ", "");

const ALPHA = "A-Za-z";
const ALPHA_LOWER = "a-z";
const ALPHA_UPPER = "A-Z";
const QUOTES = String.raw`\' " ” “ `+ "` ‘ ´ ’ ‚ , „ » « 「 」 『 』 （ ） 〔 〕 【 】 《 》 〈 〉 〈 〉  ⟦ ⟧";
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

