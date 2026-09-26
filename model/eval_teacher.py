"""
Teacher-agreement evaluation: measures how well the student model reproduces
spaCy trf's parses on the same dev sentences.

This is the honest metric for distillation: instead of scoring against silver
labels (which are trf's own output), we re-run trf on the raw words and measure
direct agreement between student predictions and fresh teacher predictions.

The gap between this number and the formal UAS/LAS tells you how much of your
eval score was "getting trf's systematic errors right" vs actual parsing quality.

Usage:
    python teacher_agreement.py \
        --run runs/electra \
        --data datasets/wiki_dataset.jsonl \
        --labels labels.json \
        --teacher en_core_web_trf

Reports:
    UAS_agree    % of arcs where student head == teacher head
    LAS_agree    % of arcs where student head AND label == teacher
    POS_agree    % of tokens where student POS == teacher POS
    (no-punct variants of each)
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer
from spacy.tokens import Doc

PUNCT_TAGS = {",", ".", ":", "``", "''", "HYPH", "#", "$", "SYM",
              "-LRB-", "-RRB-", "NFP", "ADD", "GW", "XX", "SP"}


# ---- CLE (same as eval.py, their convention: score_matrix[head][dep]) -------

def _find_cycle(parents, length, current_nodes):
    added = [False] * length; added[0] = True
    has_cycle = False; cycle = set()
    for i in range(1, length):
        if has_cycle or added[i] or not current_nodes[i]:
            continue
        this_cycle = set(); this_cycle.add(i); added[i] = True
        has_cycle = True; next_node = i
        while parents[next_node] not in this_cycle:
            next_node = parents[next_node]
            if added[next_node]: has_cycle = False; break
            added[next_node] = True; this_cycle.add(next_node)
        if has_cycle:
            original = next_node; cycle.add(original)
            next_node = parents[original]
            while next_node != original:
                cycle.add(next_node); next_node = parents[next_node]
            break
    return has_cycle, list(cycle)


def _cle(length, s, cn, fe, oi, oo, reps):
    parents = [-1]
    for n1 in range(1, length):
        parents.append(0)
        if not cn[n1]: continue
        ms = s[0, n1]
        for n2 in range(1, length):
            if n2 == n1 or not cn[n2]: continue
            if s[n2, n1] > ms: ms = s[n2, n1]; parents[n1] = n2
    hc, cyc = _find_cycle(parents, length, cn)
    if not hc:
        fe[0] = -1
        for node in range(1, length):
            if cn[node]: fe[oo[parents[node], node]] = oi[parents[node], node]
        return
    cw = sum(s[parents[c], c] for c in cyc); cr = cyc[0]; cs = set(cyc)
    for node in range(length):
        if not cn[node] or node in cs: continue
        bi, bis = -1, float("-inf"); bo, bos = -1, float("-inf")
        for c in cyc:
            if s[c, node] > bis: bis, bi = s[c, node], c
            sc = cw + s[node, c] - s[parents[c], c]
            if sc > bos: bos, bo = sc, c
        s[cr, node] = bis; oi[cr, node] = oi[bi, node]; oo[cr, node] = oo[bi, node]
        s[node, cr] = bos; oo[node, cr] = oo[node, bo]; oi[node, cr] = oi[node, bo]
    considered = []
    for i, c in enumerate(cyc):
        considered.append(set())
        if i > 0: cn[c] = False
        for node in reps[c]:
            considered[i].add(node)
            if i > 0: reps[cr].add(node)
    _cle(length, s, cn, fe, oi, oo, reps)
    found, kn = False, -1
    for i, node in enumerate(cyc):
        for rep in considered[i]:
            if rep in fe: kn = node; found = True; break
        if found: break
    prev = parents[kn]
    while prev != kn:
        fe[oo[parents[prev], prev]] = oi[parents[prev], prev]
        prev = parents[prev]


def decode_mst(scores, length):
    """scores must be [head, dep] convention (transpose attended_arcs before calling)."""
    s = np.array(scores[:length, :length], copy=True, dtype=np.float64)
    oi = np.zeros((length, length), dtype=np.int32)
    oo = np.zeros((length, length), dtype=np.int32)
    cn = [True] * length
    reps = [set([i]) for i in range(length)]
    for n1 in range(length):
        s[n1, n1] = 0.0
        for n2 in range(n1 + 1, length):
            oi[n1, n2] = n1; oo[n1, n2] = n2
            oi[n2, n1] = n2; oo[n2, n1] = n1
    fe = {}
    _cle(length, s, cn, fe, oi, oo, reps)
    heads = np.zeros(length, dtype=np.int32)
    for d, p in fe.items(): heads[d] = p
    heads[0] = 0
    return heads


# ---- load student model -------------------------------------------------------

def load_student(run_dir, device):
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import Encoder

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder(
                meta["encoder"], len(meta["tags"]), len(meta["deps"]),
                arc_representation_dim=meta["arc_dim"],
                tag_representation_dim=meta["tag_dim"],
            )
            self.tag_bilinear = nn.Bilinear(
                meta["tag_dim"], meta["tag_dim"], len(meta["deps"]))

        def forward(self, input_ids, attention_mask, word_index, shapes):
            return self.encoder(input_ids, attention_mask, word_index, shapes)

    model = Model().to(device)
    state = torch.load(os.path.join(run_dir, "model.pt"),
                       map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, meta


# ---- shape features (must match data_layer.py) --------------------------------

N_SHAPES = 5
SHAPE_TITLE, SHAPE_UPPER, SHAPE_LOWER, SHAPE_DIGIT, SHAPE_OTHER = range(5)

def shape_of(word):
    if word.isdigit():                              return SHAPE_DIGIT
    if word.isupper() and any(c.isalpha() for c in word): return SHAPE_UPPER
    if word.istitle():                              return SHAPE_TITLE
    if word.islower():                              return SHAPE_LOWER
    return SHAPE_OTHER


# ---- encode one sentence for student -----------------------------------------

def encode_sentence(words, tok, device, max_len=256):
    unk = tok.unk_token or "[UNK]"
    safe = [w if tok.tokenize(w) else unk for w in words]
    enc = tok(safe, is_split_into_words=True, add_special_tokens=True,
               truncation=True, max_length=max_len)
    word_ids = enc.word_ids()
    first_sub = {}
    for pos, wid in enumerate(word_ids):
        if wid is not None and wid not in first_sub:
            first_sub[wid] = pos
    n = len(first_sub)
    word_index = torch.tensor(
        [[0] + [first_sub[i] for i in range(n)]], dtype=torch.long, device=device)
    shapes = torch.tensor(
        [[shape_of(words[i]) for i in range(n)]], dtype=torch.long, device=device)
    input_ids = torch.tensor([enc["input_ids"]], dtype=torch.long, device=device)
    attn = torch.tensor([enc["attention_mask"]], dtype=torch.long, device=device)
    return input_ids, attn, word_index, shapes, n


# ---- get teacher parse for one sentence ---------------------------------------
def teacher_parse(words, nlp, dep2id, tag2id):
    doc = Doc(nlp.vocab, words=words)
    # run all components in pipeline order, skip what we don't need
    for name, pipe in nlp.pipeline:
        if name in ("ner", "lemmatizer", "attribute_ruler"):
            continue
        pipe(doc)
    heads, deps, tags = [], [], []
    for tok in doc:
        h = 0 if tok.head.i == tok.i else tok.head.i - doc[0].i + 1
        heads.append(h)
        deps.append(dep2id.get(tok.dep_, -1))
        tags.append(tag2id.get(tok.tag_, -1))
    return heads, deps, tags


# ---- main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="datasets/wiki_dataset.jsonl")
    ap.add_argument("--labels", default="labels.json")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--teacher", default="en_core_web_trf")
    ap.add_argument("--dev-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--max-sents", type=int, default=2000,
                    help="cap dev sentences for speed (trf is slow)")
    args = ap.parse_args()

    import spacy
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load teacher
    print(f"loading teacher {args.teacher} ...")
    nlp = spacy.load(args.teacher, disable=["ner", "lemmatizer"])

    # load student
    print(f"loading student from {args.run} ...")
    student, meta = load_student(args.run, device)
    tok_name = args.tokenizer or meta["encoder"]
    tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)

    tag_list = meta["tags"]
    dep_list = meta["deps"]
    tag2id = {t: i for i, t in enumerate(tag_list)}
    dep2id = {d: i for i, d in enumerate(dep_list)}
    punct_ids = {tag2id[t] for t in PUNCT_TAGS if t in tag2id}

    # reproduce dev split
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from prepare_data import read_jsonl
    rows = read_jsonl(args.data)
    by_doc = {}
    for r in rows:
        by_doc.setdefault(r.get("doc_id", id(r)), []).append(r)
    docs = list(by_doc)
    random.Random(args.seed).shuffle(docs)
    n_dev = max(1, int(len(docs) * args.dev_frac))
    dev_docs = set(docs[:n_dev])
    dev_rows = [r for d in dev_docs for r in by_doc[d]]
    random.Random(args.seed).shuffle(dev_rows)
    dev_rows = dev_rows[:args.max_sents]
    print(f"evaluating on {len(dev_rows)} dev sentences  device={device}\n")

    n_tok = n_tok_np = 0
    uas = las = uas_np = las_np = pos_ok = pos_ok_np = 0
    n_skipped = 0

    with torch.no_grad():
        for r in tqdm(dev_rows, desc="teacher-agree", unit="sent",
                      dynamic_ncols=True):
            words = r["words"]
            if len(words) < 2:
                continue

            # teacher parse
            t_heads, t_deps, t_tags = teacher_parse(words, nlp, dep2id, tag2id)
            if len(t_heads) != len(words):
                n_skipped += 1
                continue

            # student parse
            try:
                input_ids, attn, word_index, shapes, n = \
                    encode_sentence(words, tok, device, args.max_len)
            except Exception:
                n_skipped += 1
                continue
            if n != len(words):
                # truncation changed word count — skip for fair comparison
                n_skipped += 1
                continue

            pos_logits, attended_arcs, head_tag, dept_tag = \
                student(input_ids, attn, word_index, shapes)

            arc_np = attended_arcs[0].float().cpu().numpy()   # [W+1, W+1]
            s_heads = decode_mst(arc_np.T, n + 1)             # [W+1]

            ht = head_tag[0]; dt = dept_tag[0]
            s_deps = []
            for d in range(1, n + 1):
                h = int(s_heads[d])
                logits = student.tag_bilinear(
                    ht[h].unsqueeze(0), dt[d].unsqueeze(0))
                s_deps.append(int(logits.argmax(-1).item()))
            s_pos = pos_logits[0].argmax(-1).cpu().numpy()    # [W]

            # compare student vs teacher token by token
            for i in range(n):
                th = t_heads[i]     # teacher head (1-indexed, 0=ROOT)
                td = t_deps[i]      # teacher dep label id
                tp = t_tags[i]      # teacher pos id
                sh = int(s_heads[i + 1])
                sd = s_deps[i]
                sp = int(s_pos[i])
                is_punct = (tp in punct_ids)

                if td < 0 or tp < 0:   # unknown label/tag in teacher output
                    continue

                n_tok += 1
                pos_ok += (sp == tp)
                head_hit = (sh == th)
                uas += head_hit
                las += head_hit and (sd == td)
                if not is_punct:
                    n_tok_np += 1
                    pos_ok_np += (sp == tp)
                    uas_np += head_hit
                    las_np += head_hit and (sd == td)

    def pct(a, b): return 100.0 * a / max(b, 1)
    print(f"\n{'Metric':<14}  {'Score':>7}")
    print("-" * 24)
    print(f"{'POS_agree':<14}  {pct(pos_ok, n_tok):>7.2f}%")
    print(f"{'UAS_agree':<14}  {pct(uas, n_tok):>7.2f}%")
    print(f"{'LAS_agree':<14}  {pct(las, n_tok):>7.2f}%")
    print(f"{'UAS_agree_np':<14}  {pct(uas_np, n_tok_np):>7.2f}%")
    print(f"{'LAS_agree_np':<14}  {pct(las_np, n_tok_np):>7.2f}%")
    print(f"\ntokens: {n_tok:,}  no-punct: {n_tok_np:,}  skipped: {n_skipped}")
    print(f"\nFor context, formal eval against silver dev labels gave:")
    print(f"  UAS 95.28%  LAS 94.17%  POS 98.41%")
    print(f"Teacher-agreement is the fairer number — it measures how closely")
    print(f"the student reproduces fresh trf parses on unseen sentences.")


if __name__ == "__main__":
    main()
