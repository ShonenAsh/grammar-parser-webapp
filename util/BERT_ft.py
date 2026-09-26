import argparse
import json

import spacy
from tqdm import tqdm
from dotenv import load_dotenv

WIKI_STOP_SECTIONS = {
    "references",
    "external links",
    "see also",
    "further reading",
    "bibliography",
    "notes",
    "citations",
    "sources",
    "footnotes",
}


# for dependency parsing, so we skip short documents
def iter_wikipedia(dataset, config, split, limit, min_len_filter=2000):
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split, streaming=True)
    i = 0
    for row in ds:
        title = row.get("title", "")
        if title.startswith(("List of", "Index of", "Timeline of")):
            continue
        if "(disambiguation)" in title:
            continue
        text = row.get("text", "")
        if len(text) < min_len_filter:
            continue
        # yield wiki-doc_id, text
        yield f"wiki-{row.get('id', i)}", text
        i += 1
        if limit and i >= limit:
            return


def parse_wikipedia_paragraphs(text):
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        # detect sections and cut the article
        if len(line) < 50 and not line.endswith((".", "?", "!", '"')):
            if line.strip(" =").lower() in WIKI_STOP_SECTIONS:
                return
            continue
        yield line


def para_stream(args):
    for doc_id, text in iter_wikipedia(
        args.dataset, args.config, args.split, args.max_docs
    ):
        for para in parse_wikipedia_paragraphs(text):
            yield para, doc_id


def parse_with_spacy(doc, doc_id, stats, source="Wikipedia", min_len=4, max_len=100):
    res = []

    for sent in doc.sents:
        stats["seen"] += 1
        kept = [t for t in sent if not t.is_space]
        if len(kept) < min_len:
            stats["too_short"] += 1
            continue
        if len(kept) > max_len:
            stats["too_long"] += 1
            continue
        old2new = {t.i: i + 1 for i, t in enumerate(kept)}

        words, tags, heads, deps = [], [], [], []
        n_root = 0
        broken = False
        for t in kept:
            if t.head.i == t.i:
                h = 0
                n_root += 1
            elif t.head.i in old2new:
                h = old2new[t.head.i]
            else:
                broken = True
                break
            words.append(t.text)
            tags.append(t.tag_)
            heads.append(h)
            deps.append(t.dep_)

        if broken:
            stats["space_head"] += 1
            continue

        if n_root != 1:
            stats["multi_root"] += 1
            continue

        stats["kept"] += 1
        res.append(
            {
                "doc_id": doc_id,
                "source": source,
                "words": words,
                "tags": tags,
                "heads": heads,
                "deps": deps,
            }
        )
    return res


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="en_core_web_trf")
    ap.add_argument("--max-sentences", type=int, default=150000)
    ap.add_argument("--max-docs", type=int, default=0, help="0 = unlimited")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--dataset", default="wikimedia/wikipedia")
    ap.add_argument("--config", default="20231101.en")
    ap.add_argument("--split", default="train")
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=100)
    args = ap.parse_args()
    
    try:
        load_dotenv()
    except ValueError:
        print("could not find env file containing HF_TOKEN, skipping")


    if args.gpu:
        spacy.require_gpu()

    nlp = spacy.load(args.model, disable=["ner", "lemmatizer"])
    tag_labels = sorted(nlp.get_pipe("tagger").labels)
    dep_labels = sorted(nlp.get_pipe("parser").labels)
    label_file = {"tags": tag_labels, "deps": dep_labels}
    # dump labels
    with open("labels.json", "w", encoding="utf-8") as f:
        json.dump(label_file, f, indent=2)

    n = 0
    stats = {
        "seen": 0,
        "too_short": 0,
        "too_long": 0,
        "space_head": 0,
        "multi_root": 0,
        "kept": 0,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        bar = tqdm(
            total=args.max_sentences,
            unit="sent",
            desc="wikipedia",
            smoothing=0.1,
            dynamic_ncols=True,
        )

        for doc, doc_id in nlp.pipe(
            para_stream(args), as_tuples=True, batch_size=args.batch_size
        ):
            recs = parse_with_spacy(
                doc,
                doc_id,
                stats,
                min_len=args.min_len,
                max_len=args.max_len,
            )
            before = n
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
            bar.update(min(n, args.max_sentences) - before)
            bar.set_postfix(
                kept=f"{100.0 * stats['kept'] / max(stats['seen'], 1):.0f}%",
                docs=(str(doc_id).split("-", 1)[-1]),
                refresh=False,
            )
            if n >= args.max_sentences:
                break

        bar.close()
    print(f"wrote {n} records to {args.out}")


if __name__ == "__main__":
    main()
