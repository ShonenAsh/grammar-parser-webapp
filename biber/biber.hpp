#ifndef BIBER_H
#define BIBER_H

#include <cstdint>

/**
 * Fine-grained Part-of-Speech tags from en_core_web_sm (OntoNotes 5)
 * mapped to fixed integers for SIMD-compatible processing.
 */
enum class PosTag : int32_t {
  UNKNOWN = 0,
  CURRENCY = 1,    // $
  QUOTE_CLOSE = 2, // ''
  COMMA = 3,       // ,
  PUNCT_OPEN = 4,  // -LRB-
  PUNCT_CLOSE = 5, // -RRB-
  PERIOD = 6,      // .
  COLON = 7,       // :
  ADD = 8,         // Email/URL/Address
  AFX = 9,         // Affix
  CC = 10,         // Conjunction, coordinating
  CD = 11,         // Cardinal number
  DT = 12,         // Determiner
  EX = 13,         // Existential there
  FW = 14,         // Foreign word
  HYPH = 15,       // Hyphen
  IN = 16,         // Preposition or subordinating conjunction
  JJ = 17,         // Adjective
  JJR = 18,        // Adjective, comparative
  JJS = 19,        // Adjective, superlative
  LS = 20,         // List item marker
  MD = 21,         // Modal
  NFP = 22,        // Superfluous punctuation
  NN = 23,         // Noun, singular or mass
  NNP = 24,        // Proper noun, singular
  NNPS = 25,       // Proper noun, plural
  NNS = 26,        // Noun, plural
  PDT = 27,        // Predeterminer
  POS = 28,        // Possessive ending
  PRP = 29,        // Personal pronoun
  PRP_POSS = 30,   // Possessive pronoun (PRP$)
  RB = 31,         // Adverb
  RBR = 32,        // Adverb, comparative
  RBS = 33,        // Adverb, superlative
  RP = 34,         // Particle
  SYM = 35,        // Symbol
  TO = 36,         // to
  UH = 37,         // Interjection
  VB = 38,         // Verb, base form
  VBD = 39,        // Verb, past tense
  VBG = 40,        // Verb, gerund or present participle
  VBN = 41,        // Verb, past participle
  VBP = 42,        // Verb, non-3rd person singular present
  VBZ = 43,        // Verb, 3rd person singular present
  WDT = 44,        // Wh-determiner
  WP = 45,         // Wh-pronoun
  WP_POSS = 46,    // Possessive wh-pronoun (WP$)
  WRB = 47,        // Wh-adverb
  XX = 48,         // Unknown
  SPACE = 49,      // _SP
  QUOTE_OPEN = 50  // ``
};

enum class UPos : int32_t {
  ADJ,
  ADP,
  ADV,
  AUX,
  CONJ,
  CCONJ,
  DET,
  INTJ,
  NOUN,
  NUM,
  PART,
  PRON,
  PROPN,
  PUNCT,
  SCONJ,
  SYM,
  VERB,
  X,
  SPACE
};

enum class DepRel : int32_t {
  ROOT = 0,
  ACL,
  ACOMP,
  ADVCL,
  ADVMOD,
  AGENT,
  AMOD,
  APPOS,
  ATTR,
  AUX,
  AUXPASS,
  CASE,
  CC,
  CCOMP,
  COMPOUND,
  CONJ,
  CSUBJ,
  CSUBJPASS,
  DATIVE,
  DEP,
  DET,
  DOBJ,
  EXPL,
  INTJ,
  MARK,
  META,
  NEG,
  NMOD,
  NPADVMOD,
  NSUBJ,
  NSUBJPASS,
  NUMMOD,
  OPRD,
  PARATAXIS,
  PCOMP,
  POBJ,
  POSS,
  PRECONJ,
  PREDET,
  PREP,
  PRT,
  PUNCT,
  QUANTMOD,
  RELCL,
  XCOMP,
  UNKNOWN // Safety value
};

/**
 * Major POS categories categories based on tags.
 */
namespace PosTagCategory {
inline bool is_noun(PosTag t) {
  return (t == PosTag::NN || t == PosTag::NNS || t == PosTag::NNP ||
          t == PosTag::NNPS);
}

inline bool is_verb(PosTag t) { return (t >= PosTag::VB && t <= PosTag::VBZ); }

inline bool is_wh_word(PosTag t) {
  return (t == PosTag::WDT || t == PosTag::WP || t == PosTag::WP_POSS ||
          t == PosTag::WRB);
}
} // namespace PosTagCategory

UPos resolve_upos(int32_t tag_idx, int32_t dep_ids);

#endif // BIBER_H
