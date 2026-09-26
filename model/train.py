"""
Train the joint POS tagger + biaffine parser on datasets/wiki_dataset.jsonl.

  python train.py --data datasets/wiki_dataset.jsonl --labels labels.json --out runs/electra

Pieces this wires together:
  GraphEncoder (graph_encoder.py)  -> pos_logits, attended_arcs, head_tag, dept_tag
  get_head_tags + construct_loss (parser_loss.py) -> arc_nll, tag_nll  (labels at GOLD heads)
  a POS cross-entropy on pos_logits
  the label biaffine (tag_bilinear) lives HERE, not in the model, because it is
  plain nn.Bilinear (unnormalized) and is only needed for loss/decode.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from model import Encoder, construct_loss, get_head_tags
from prepare_data import ParseDataset, load_labels, make_collate, read_jsonl
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup


def split_by_doc(rows, dev_frac, seed):
    """Hold out whole documents so correlated sentences don't leak into dev."""
    by_doc = {}
    for r in rows:
        by_doc.setdefault(r.get("doc_id", id(r)), []).append(r)
    docs = list(by_doc)
    random.Random(seed).shuffle(docs)
    n_dev = max(1, int(len(docs) * dev_frac))
    dev_docs = set(docs[:n_dev])
    train = [r for d, rs in by_doc.items() if d not in dev_docs for r in rs]
    dev = [r for d in dev_docs for r in by_doc[d]]
    return train, dev


class Model(nn.Module):
    """GraphEncoder + the label biaffine, so one module holds all parameters."""

    def __init__(
        self,
        encoder_name,
        n_pos,
        n_edge_labels,
        arc_representation_dim=384,
        tag_representation_dim=128,
    ):
        super().__init__()
        self.encoder = Encoder(
            encoder_name,
            n_pos,
            n_edge_labels,
            arc_representation_dim=arc_representation_dim,
            tag_representation_dim=tag_representation_dim,
        )
        # label scorer: plain (unnormalized) bilinear over the label projections
        self.tag_bilinear = nn.Bilinear(
            tag_representation_dim, tag_representation_dim, n_edge_labels
        )

    def forward(self, input_ids, attention_mask, word_index, shapes):
        return self.encoder(input_ids, attention_mask, word_index, shapes)


def compute_loss(model, out, batch):
    pos_logits, attended_arcs, head_tag, dept_tag = out
    gold_heads = batch["head_idx"]  # [B, W+1], col 0 = -100
    gold_tags = batch["head_tags"]  # [B, W+1]
    gold_pos = batch["pos_tags"]  # [B, W+1]
    word_mask = batch["word_mask"]  # [B, W+1]

    # label logits scored at GOLD heads (clamp -100 ROOT/pad to 0 for the gather;
    # those rows are ignored by the loss anyway)
    safe_heads = gold_heads.clamp(min=0)
    head_tag_logits = get_head_tags(head_tag, dept_tag, safe_heads, model.tag_bilinear)

    arc_nll, tag_nll = construct_loss(
        attended_arcs, head_tag_logits, safe_heads, gold_tags, word_mask
    )

    # POS head: pos_logits is [B, W] (ROOT excluded by the model), gold_pos is
    # [B, W+1] with col 0 = ROOT. Compare pos_logits against gold_pos[:, 1:].
    pos_nll = F.cross_entropy(
        pos_logits.reshape(-1, pos_logits.size(-1)),
        gold_pos[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    return arc_nll + tag_nll + pos_nll, (arc_nll.item(), tag_nll.item(), pos_nll.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/wiki_dataset.jsonl")
    ap.add_argument("--labels", default="labels.json")
    ap.add_argument("--encoder", default="prajjwal1/bert-mini")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", default="runs/bert-mini")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--accum", type=int, default=2, help="effective batch = bs*accum")
    ap.add_argument("--encoder-lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--dev-frac", type=float, default=0.05)
    ap.add_argument("--arc-dim", type=int, default=384)
    ap.add_argument("--tag-dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    try:
        load_dotenv()
    except ValueError:
        print("could not find env file containing HF_TOKEN, skipping")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tag2id, dep2id, tags, deps = load_labels(args.labels)

    tok_name = args.tokenizer or args.encoder
    tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)

    assert tok.is_fast, "need a fast tokenizer for word_ids() alignment"

    rows = read_jsonl(args.data)
    train_rows, dev_rows = split_by_doc(rows, args.dev_frac, args.seed)
    train_ds = ParseDataset(train_rows, tok, tag2id, dep2id, args.max_len, True)
    dev_ds = ParseDataset(dev_rows, tok, tag2id, dep2id, args.max_len, False)
    print(
        f"{len(rows)} sentences -> train {len(train_ds)} "
        f"({train_ds.n_skipped} skipped), dev {len(dev_ds)} ({dev_ds.n_skipped} skipped)"
    )
    print(f"{len(tags)} POS tags, {len(deps)} edge labels")

    collate = make_collate(tok.pad_token_id)
    train_dl = DataLoader(
        train_ds,
        args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
        persistent_workers=True,
        drop_last=True,
        pin_memory=True,
    )
    dev_dl = DataLoader(
        dev_ds,
        args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        collate_fn=collate,
    )

    model = Model(args.encoder, len(tags), len(deps), args.arc_dim, args.tag_dim).to(
        device
    )

    # split LR: pretrained encoder low, everything else (heads, biaffines,
    # shape emb, sentinel, label bilinear) high
    enc_ids = {id(p) for p in model.encoder.encoder.parameters()}
    enc_p = [p for p in model.parameters() if id(p) in enc_ids]
    head_p = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW(
        [
            {"params": enc_p, "lr": args.encoder_lr},
            {"params": head_p, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )

    steps = (len(train_dl) // args.accum) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(steps * args.warmup), steps)
    use_amp = (not args.no_amp) and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"device: {device}  AMP: {use_amp}")

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(
            {
                "encoder": args.encoder,
                "tags": tags,
                "deps": deps,
                "arc_dim": args.arc_dim,
                "tag_dim": args.tag_dim,
            },
            f,
            indent=2,
        )
    tok.save_pretrained(args.out)

    # sanity check: one forward pass before committing to the full run
    model.eval()
    with torch.no_grad():
        batch = next(iter(train_dl))
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["word_index"],
            batch["shapes"],
        )
        loss, (a, t, p) = compute_loss(model, out, batch)
        print(f"sanity  loss {loss.item():.3f}  arc {a:.3f}  tag {t:.3f}  pos {p:.3f}")
    model.train()

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        run_loss = run_arc = run_tag = run_pos = 0.0
        bar = tqdm(train_dl, desc=f"epoch {ep:2d}", unit="batch", dynamic_ncols=True)
        for step, batch in enumerate(bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", enabled=use_amp):
                out = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["word_index"],
                    batch["shapes"],
                )
                loss, (a, t, p) = compute_loss(model, out, batch)
                loss = loss / args.accum
            scaler.scale(loss).backward()
            run_loss += loss.item() * args.accum
            run_arc += a
            run_tag += t
            run_pos += p
            if (step + 1) % args.accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            bar.set_postfix(
                loss=f"{loss.item() * args.accum:.3f}",
                arc=f"{a:.3f}",
                refresh=False,
            )
        bar.close()
        nb = len(train_dl)
        print(
            f"epoch {ep:2d}  loss {run_loss / nb:.3f}  "
            f"(arc {run_arc / nb:.3f}  tag {run_tag / nb:.3f}  pos {run_pos / nb:.3f})  "
            f"{time.time() - t0:.0f}s"
        )
        torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))

    print("done. eval/decode is a separate step.")


if __name__ == "__main__":
    main()
