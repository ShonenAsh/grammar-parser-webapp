"""
Formal evaluation: UAS / LAS / POS accuracy on the dev set.

Uses Chu-Liu-Edmonds (reference: paolo-gajo/EfficientSDP GraphDecoder) for valid trees,
matching the repo's _mst_decode path.

Usage:
    python eval.py --run runs/electra --data datasets/wiki_dataset.jsonl --labels labels.json

Metrics reported:
    UAS      unlabelled attachment score  (all tokens, excl. ROOT row)
    LAS      labelled attachment score    (all tokens, excl. ROOT row)
    UAS_np   UAS excluding punctuation
    LAS_np   LAS excluding punctuation
    POS      XPOS tag accuracy            (all tokens)

These match how spaCy reports DEP_UAS / DEP_LAS / TAG_ACC so you can compare
directly against en_core_web_sm's published numbers.
"""

import argparse
import json
import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# PTB punct tags to exclude from UAS/LAS (spaCy convention)
PUNCT_TAGS = {",", ".", ":", "``", "''", "HYPH", "#", "$", "SYM",
              "-LRB-", "-RRB-", "NFP", "ADD", "GW", "XX", "SP"}


# Convention: score_matrix[i][j] = score of arc j -> i (j is head of i)

def _find_cycle(parents, length, current_nodes):
    added = [False] * length
    added[0] = True
    has_cycle = False
    cycle = set()
    for i in range(1, length):
        if has_cycle or added[i] or not current_nodes[i]:
            continue
        this_cycle = set()
        this_cycle.add(i)
        added[i] = True
        has_cycle = True
        next_node = i
        while parents[next_node] not in this_cycle:
            next_node = parents[next_node]
            if added[next_node]:
                has_cycle = False
                break
            added[next_node] = True
            this_cycle.add(next_node)
        if has_cycle:
            original = next_node
            cycle.add(original)
            next_node = parents[original]
            while next_node != original:
                cycle.add(next_node)
                next_node = parents[next_node]
            break
    return has_cycle, list(cycle)


def _chu_liu_edmonds(length, score_matrix, current_nodes,
                     final_edges, old_input, old_output, representatives):
    parents = [-1]
    for node1 in range(1, length):
        parents.append(0)
        if not current_nodes[node1]:
            continue
        max_score = score_matrix[0, node1]
        for node2 in range(1, length):
            if node2 == node1 or not current_nodes[node2]:
                continue
            if score_matrix[node2, node1] > max_score:
                max_score = score_matrix[node2, node1]
                parents[node1] = node2

    has_cycle, cycle = _find_cycle(parents, length, current_nodes)
    if not has_cycle:
        final_edges[0] = -1
        for node in range(1, length):
            if not current_nodes[node]:
                continue
            parent = old_input[parents[node], node]
            dept = old_output[parents[node], node]
            final_edges[dept] = parent
        return

    cycle_weight = sum(score_matrix[parents[n], n] for n in cycle)
    cycle_rep = cycle[0]
    for node in range(length):
        if not current_nodes[node] or node in set(cycle):
            continue
        best_in, best_in_score = -1, float("-inf")
        best_out, best_out_score = -1, float("-inf")
        for c in cycle:
            if score_matrix[c, node] > best_in_score:
                best_in_score, best_in = score_matrix[c, node], c
            score = (cycle_weight + score_matrix[node, c]
                     - score_matrix[parents[c], c])
            if score > best_out_score:
                best_out_score, best_out = score, c
        score_matrix[cycle_rep, node] = best_in_score
        old_input[cycle_rep, node] = old_input[best_in, node]
        old_output[cycle_rep, node] = old_output[best_in, node]
        score_matrix[node, cycle_rep] = best_out_score
        old_output[node, cycle_rep] = old_output[node, best_out]
        old_input[node, cycle_rep] = old_input[node, best_out]

    considered = []
    for i, c in enumerate(cycle):
        considered.append(set())
        if i > 0:
            current_nodes[c] = False
        for node in representatives[c]:
            considered[i].add(node)
            if i > 0:
                representatives[cycle_rep].add(node)

    _chu_liu_edmonds(length, score_matrix, current_nodes,
                     final_edges, old_input, old_output, representatives)

    found, key_node = False, -1
    for i, node in enumerate(cycle):
        for rep in considered[i]:
            if rep in final_edges:
                key_node = node
                found = True
                break
        if found:
            break

    prev = parents[key_node]
    while prev != key_node:
        dept = old_output[parents[prev], prev]
        parent = old_input[parents[prev], prev]
        final_edges[dept] = parent
        prev = parents[prev]


def decode_mst(scores, length):
    """
    Maximum spanning arborescence via Chu-Liu-Edmonds.

    Convention follows the EfficientSDP GraphDecoder (AllenNLP):
      scores[i][j] = score of arc FROM j TO i  (j is head of i)
    BUT their _chu_liu_edmonds inner loop reads score_matrix[node2, node1]
    where node2 iterates as candidate head and node1 is the dependent.
    So the matrix fed here must be score_matrix[head][dep], i.e. TRANSPOSED
    relative to the model's attended_arcs[dep][head] convention.
    Caller must transpose before calling: decode_mst(arc_np[b].T, n)

    Returns heads array of length `length`, heads[0] = 0 (ROOT sentinel).
    """
    s = np.array(scores[:length, :length], copy=True, dtype=np.float64)
    old_input = np.zeros((length, length), dtype=np.int32)
    old_output = np.zeros((length, length), dtype=np.int32)
    current_nodes = [True] * length
    representatives = [set([i]) for i in range(length)]
    for n1 in range(length):
        s[n1, n1] = 0.0
        for n2 in range(n1 + 1, length):
            old_input[n1, n2] = n1; old_output[n1, n2] = n2
            old_input[n2, n1] = n2; old_output[n2, n1] = n1
    final_edges = {}
    _chu_liu_edmonds(length, s, current_nodes, final_edges,
                     old_input, old_output, representatives)
    heads = np.zeros(length, dtype=np.int32)
    for dept, parent in final_edges.items():
        heads[dept] = parent
    heads[0] = 0
    return heads


# ---- model loading (mirrors train.py) ----

def load_model(run_dir, device):
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


# ---- evaluation loop ----

@torch.no_grad()
def evaluate(model, dev_dl, meta, device):
    tag_list = meta["tags"]
    tag2id = {t: i for i, t in enumerate(tag_list)}
    punct_ids = {tag2id[t] for t in PUNCT_TAGS if t in tag2id}

    n_tok = n_tok_np = 0
    uas = las = uas_np = las_np = pos_ok = 0

    for batch in tqdm(dev_dl, desc="eval", unit="batch", dynamic_ncols=True):
        batch = {k: v.to(device) for k, v in batch.items()}
        pos_logits, attended_arcs, head_tag, dept_tag = model(
            batch["input_ids"], batch["attention_mask"],
            batch["word_index"], batch["shapes"])

        gold_heads = batch["head_idx"].cpu().numpy()    # [B, W+1]
        gold_tags = batch["head_tags"].cpu().numpy()    # [B, W+1]
        gold_pos = batch["pos_tags"].cpu().numpy()      # [B, W+1]
        word_mask = batch["word_mask"].cpu().numpy()    # [B, W+1]

        arc_np = attended_arcs.float().cpu().numpy()    # [B, W+1, W+1]
        pos_pred = pos_logits.argmax(-1).cpu().numpy()  # [B, W]

        B = arc_np.shape[0]
        for b in range(B):
            # real word count for this sentence
            L = int(word_mask[b].sum()) - 1   # subtract ROOT
            if L <= 0:
                continue
            n = L + 1   # includes ROOT at pos 0

            # CLE decode: transpose because decode_mst expects [head, dep]
            # but attended_arcs is [dep, head] (model convention)
            pred_heads = decode_mst(arc_np[b, :n, :n].T, n)  # [n]

            # label: score at predicted head for each dependent
            ht = head_tag[b]   # [W+1, tag_dim]
            dt = dept_tag[b]
            pred_deps = []
            for d in range(1, n):
                h = int(pred_heads[d])
                logits = model.tag_bilinear(
                    ht[h].unsqueeze(0), dt[d].unsqueeze(0))  # [1, n_deps]
                pred_deps.append(int(logits.argmax(-1).item()))

            # score against gold (cols 1..L of the W+1 axis)
            for i in range(L):
                col = i + 1
                gh = gold_heads[b, col]
                gd = gold_tags[b, col]
                gp = gold_pos[b, col]
                if gh == -100:          # truncated / padding
                    continue
                ph = int(pred_heads[i + 1])
                pd = pred_deps[i]
                pp = int(pos_pred[b, i])
                is_punct = (gp in punct_ids)

                n_tok += 1
                pos_ok += (pp == gp)
                head_hit = (ph == gh)
                uas += head_hit
                las += head_hit and (pd == gd)
                if not is_punct:
                    n_tok_np += 1
                    uas_np += head_hit
                    las_np += head_hit and (pd == gd)

    def pct(a, b): return 100.0 * a / max(b, 1)
    return {
        "POS":    pct(pos_ok, n_tok),
        "UAS":    pct(uas,    n_tok),
        "LAS":    pct(las,    n_tok),
        "UAS_np": pct(uas_np, n_tok_np),
        "LAS_np": pct(las_np, n_tok_np),
        "n_tokens": n_tok,
        "n_tokens_nopunct": n_tok_np,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="datasets/wiki_dataset.jsonl")
    ap.add_argument("--labels", default="labels.json")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--dev-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_model(args.run, device)

    tok_name = args.tokenizer or meta["encoder"]
    tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from prepare_data import ParseDataset, load_labels, make_collate, read_jsonl

    tag2id, dep2id, _, _ = load_labels(args.labels)
    rows = read_jsonl(args.data)

    # reproduce the same dev split as training
    by_doc = {}
    for r in rows:
        by_doc.setdefault(r.get("doc_id", id(r)), []).append(r)
    docs = list(by_doc)
    random.Random(args.seed).shuffle(docs)
    n_dev = max(1, int(len(docs) * args.dev_frac))
    dev_docs = set(docs[:n_dev])
    dev_rows = [r for d in dev_docs for r in by_doc[d]]

    dev_ds = ParseDataset(dev_rows, tok, tag2id, dep2id, args.max_len, train=False)
    dev_dl = DataLoader(dev_ds, args.batch_size, shuffle=False,
                        collate_fn=make_collate(tok.pad_token_id),
                        num_workers=args.num_workers, pin_memory=True)

    print(f"dev sentences: {len(dev_ds)}  device: {device}")
    m = evaluate(model, dev_dl, meta, device)

    print(f"\n{'Metric':<12}  {'Score':>7}")
    print("-" * 22)
    for k in ["POS", "UAS", "LAS", "UAS_np", "LAS_np"]:
        print(f"{k:<12}  {m[k]:>7.2f}%")
    print(f"\ntokens: {m['n_tokens']:,}  "
          f"no-punct: {m['n_tokens_nopunct']:,}")
    print("\nen_core_web_sm baselines (OntoNotes, for rough comparison):")
    print("  TAG_ACC  97.29   UAS  91.77   LAS  89.92")
    print("  Note: sm uses XPOS on OntoNotes/ClearNLP; your labels are silver.")
    print("  Agreement with spaCy trf on your own data is the fairer comparison.")


if __name__ == "__main__":
    main()
