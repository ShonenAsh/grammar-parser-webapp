// @ts-check
// spaCy's rule-based English lemmatizer (Lemmatizer.rule_lemmatize with
// EnglishLemmatizer.is_base_form): suffix rules checked against a per-POS word
// index, with an exceptions table taking priority. Lemmas set by the attribute
// ruler (src/parser/attributeRuler.js) take precedence over this, as in spaCy.

/** Python's str.isalpha(): every character is a Unicode letter. */
function isAlpha(/** @type {string} */ str) {
  return /^\p{L}+$/u.test(str);
}

/**
 * spaCy's is_base_form: an uninflected form, which is only lowercased.
 * @param {string} pos lowercased coarse POS
 * @param {Map<string, string>} morph
 */
function isBaseForm(pos, morph) {
  if (pos === "noun" && morph.get("Number") === "Sing") return true;
  if (pos === "verb" && morph.get("VerbForm") === "Inf") return true;
  // VBP: finite present tense without number.
  if (pos === "verb" && morph.get("VerbForm") === "Fin" && morph.get("Tense") === "Pres"
      && !morph.has("Number")) return true;
  if (pos === "adj" && morph.get("Degree") === "Pos") return true;
  return morph.get("VerbForm") === "Inf" || morph.get("VerbForm") === "None"
    || morph.get("Degree") === "Pos";
}

/** "Number=Sing|Degree=Pos" -> Map */
function parseMorph(/** @type {string} */ morph) {
  const out = new Map();
  for (const feature of morph.split("|")) {
    const eq = feature.indexOf("=");
    if (eq > 0) out.set(feature.slice(0, eq), feature.slice(eq + 1));
  }
  return out;
}

export class Lemmatizer {
  /** @param {string} dataUrl */
  async init(dataUrl) {
    const res = await fetch(dataUrl);
    if (!res.ok) throw new Error(`${dataUrl}: HTTP ${res.status}`);
    const data = await res.json();
    /** @type {Record<string, Record<string, string[]>>} */
    this.exc = data["en_lemma_exc"];
    /** @type {Record<string, [string, string][]>} */
    this.rules = data["en_lemma_rules"];
    /** @type {Record<string, Set<string>>} */
    this.index = Object.fromEntries(
      Object.entries(/** @type {Record<string, string[]>} */ (data["en_lemma_index"]))
        .map(([pos, words]) => [pos, new Set(words)]),
    );
  }

  /**
   * @param {string} word
   * @param {string} pos spaCy coarse POS (NOUN, VERB, PROPN, ...)
   * @param {string} [morph] spaCy morphology, e.g. "Number=Sing"
   * @returns {string}
   */
  lemmatize(word, pos, morph = "") {
    const univ = pos.toLowerCase();
    if (univ === "" || univ === "eol" || univ === "space") return word.toLowerCase();
    if (isBaseForm(univ, parseMorph(morph))) return word.toLowerCase();

    const index = this.index?.[univ];
    const exceptions = this.exc?.[univ];
    const rules = this.rules?.[univ];
    if (!index?.size && !(exceptions && Object.keys(exceptions).length) && !rules?.length) {
      return univ === "propn" ? word : word.toLowerCase();
    }

    const lower = word.toLowerCase();
    /** @type {string[]} */
    let forms = [];
    /** @type {string[]} */
    const oovForms = [];
    for (const [oldSuffix, newSuffix] of rules ?? []) {
      if (lower.endsWith(oldSuffix)) {
        const form = lower.substring(0, lower.length - oldSuffix.length) + newSuffix;
        if (!form) continue;
        if (index?.has(form)) forms.unshift(form);
        else if (!isAlpha(form)) forms.push(form);
        else oovForms.push(form);
      }
    }
    // preserve order but remove duplicates
    forms = [...new Set(forms)];
    if (exceptions && Object.hasOwn(exceptions, lower)) {
      for (const e of exceptions[lower] ?? []) {
        if (!forms.includes(e)) forms.unshift(e);
      }
    }
    return forms[0] ?? oovForms[0] ?? word;
  }
}
