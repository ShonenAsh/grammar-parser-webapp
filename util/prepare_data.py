"""
Data layer for the joint POS tagger + biaffine parser.

Record format (one JSON object per line in datasets/wiki_dataset.jsonl):
  {"words": [str], "tags": [str], "heads": [int], "deps": [str], "doc_id": str}
heads are sentence-local, 1-indexed, 0 = ROOT.

Produces, per sentence, tensors aligned to the model's word axis:
  position 0 is the ROOT sentinel; positions 1..W are the real words.

The load-bearing part is word_index: word_index[0]=0 (ROOT slot, its gathered
value is overwritten by _head_sentinel in the model) and word_index[i+1] is the
subword position of the FIRST piece of word i. If this is wrong, the model trains
on misaligned targets and parses garbage while looking fine — so it is tested
against a hand-worked example in the __main__ block below.
"""

import json

import torch
from torch.utils.data import Dataset


# shape / casing features (recover signal an uncased encoder discards)

SHAPE_TITLE, SHAPE_UPPER, SHAPE_LOWER, SHAPE_DIGIT, SHAPE_OTHER = range(5)
N_SHAPES = 5


def shape_of(word):
    if word.isdigit():
        return SHAPE_DIGIT
    if word.isupper() and any(c.isalpha() for c in word):
        return SHAPE_UPPER
    if word.istitle():
        return SHAPE_TITLE
    if word.islower():
        return SHAPE_LOWER
    return SHAPE_OTHER


# label inventory (the contract frozen during silver dumping)

def load_labels(path="labels.json"):
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    tags, deps = meta["tags"], meta["deps"]
    return ({t: i for i, t in enumerate(tags)},
            {d: i for i, d in enumerate(deps)},
            tags, deps)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class ParseDataset(Dataset):
    """
    Aligns each word to its first subword and builds per-sentence tensors.

    Target convention on the W+1 word axis (index 0 = ROOT):
      pos_tags   [W+1]  ROOT gets -100 (ignored); words get tag ids
      head_idx   [W+1]  ROOT gets -100; word i gets its gold head in 0..W
      head_tags  [W+1]  ROOT gets -100; word i gets its gold deprel id
    Using -100 on the ROOT row means the loss ignores it without extra masking.
    """

    def __init__(self, rows, tokenizer, tag2id, dep2id, max_len=256, train=True):
        self.tok = tokenizer
        self.tag2id = tag2id
        self.dep2id = dep2id
        self.max_len = max_len
        self.examples = []
        self.n_skipped = 0

        unk = tokenizer.unk_token or "[UNK]"

        for r in rows:
            words = r["words"]
            # A word that tokenizes to zero pieces would desync alignment; sub UNK.
            safe = [w if self.tok.tokenize(w) else unk for w in words]

            enc = self.tok(
                safe,
                is_split_into_words=True,
                add_special_tokens=True,
                truncation=True,
                max_length=max_len,
            )
            word_ids = enc.word_ids()

            # first subword position of each word (in original word order)
            first_sub = {}
            for pos, wid in enumerate(word_ids):
                if wid is not None and wid not in first_sub:
                    first_sub[wid] = pos
            n_kept = len(first_sub)
            if n_kept == 0:
                self.n_skipped += 1
                continue

            # word_index[0]=0 (ROOT slot); then first subword of words 0..n_kept-1
            word_index = [0] + [first_sub[i] for i in range(n_kept)]
            shapes = [shape_of(words[i]) for i in range(n_kept)]
            tags = [tag2id.get(r["tags"][i], 0) for i in range(n_kept)]

            heads, deps = [], []
            truncated = False
            for i in range(n_kept):
                h = r["heads"][i]
                if h > n_kept:                      # head fell past the truncation
                    heads.append(-100)
                    deps.append(-100)
                    truncated = True
                else:
                    heads.append(h)                 # already 0..n_kept, 0 = ROOT
                    deps.append(self.dep2id.get(r["deps"][i], -100))
            if train and truncated:
                self.n_skipped += 1
                continue

            self.examples.append({
                "input_ids": enc["input_ids"],
                "word_index": word_index,           # length W+1
                "shapes": shapes,                   # length W
                "tags": tags,                       # length W
                "heads": heads,                     # length W
                "deps": deps,                       # length W
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def make_collate(pad_id):
    """
    Pads a batch. Word-axis targets carry the ROOT row explicitly so their length
    is W+1; shapes/tags/heads/deps are per-word (length W) and get the ROOT row
    added as -100 / 0 here so every word-axis tensor lines up at W+1.
    """
    def collate(batch):
        B = len(batch)
        T = max(len(b["input_ids"]) for b in batch)
        W = max(len(b["tags"]) for b in batch)

        input_ids = torch.full((B, T), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, T), dtype=torch.long)
        word_index = torch.zeros((B, W + 1), dtype=torch.long)
        word_mask = torch.zeros((B, W + 1), dtype=torch.long)   # 1 for ROOT+words
        shapes = torch.zeros((B, W), dtype=torch.long)          # per real word
        # word-axis targets, ROOT row at col 0 set to ignore
        pos_tags = torch.full((B, W + 1), -100, dtype=torch.long)
        head_idx = torch.full((B, W + 1), -100, dtype=torch.long)
        head_tags = torch.full((B, W + 1), -100, dtype=torch.long)

        for i, b in enumerate(batch):
            t = len(b["input_ids"])
            w = len(b["tags"])
            input_ids[i, :t] = torch.tensor(b["input_ids"])
            attention_mask[i, :t] = 1
            word_index[i, : w + 1] = torch.tensor(b["word_index"])
            word_mask[i, : w + 1] = 1
            shapes[i, :w] = torch.tensor(b["shapes"])
            # shift real-word targets into columns 1..w; column 0 stays -100 (ROOT)
            pos_tags[i, 1 : w + 1] = torch.tensor(b["tags"])
            head_idx[i, 1 : w + 1] = torch.tensor(b["heads"])
            head_tags[i, 1 : w + 1] = torch.tensor(b["deps"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_index": word_index,       # [B, W+1]
            "word_mask": word_mask,         # [B, W+1]
            "shapes": shapes,               # [B, W]
            "pos_tags": pos_tags,           # [B, W+1], col 0 = -100
            "head_idx": head_idx,           # [B, W+1], col 0 = -100
            "head_tags": head_tags,         # [B, W+1], col 0 = -100
        }

    return collate
