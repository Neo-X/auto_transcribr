#!/usr/bin/env python3
"""
Add names/jargon to vocabulary.txt, the hotword list dictate.py gives Qwen3-ASR / Granite.

dictate.py re-reads the file before every transcription, so new terms apply to the next
recording without a restart. Terms already present (case-insensitive) are skipped.

Usage:
    python3 add_vocab.py Faisal "Rebekah Thompson" LIBERO-90
    python3 add_vocab.py --list
"""
import argparse
import os

VOCAB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocabulary.txt")


def read_terms():
    if not os.path.exists(VOCAB_FILE):
        return []
    with open(VOCAB_FILE) as f:
        return [t.strip() for t in f.read().replace("\n", ",").split(",") if t.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("terms", nargs="*", help="Terms to add (quote multi-word terms)")
    parser.add_argument("--list", action="store_true", help="Print the current vocabulary")
    args = parser.parse_args()

    terms = read_terms()
    if args.list or not args.terms:
        print(f"{len(terms)} terms: " + ", ".join(terms))
        return
    known = {t.lower() for t in terms}
    added = []
    for t in (t.strip() for t in args.terms):
        if t and t.lower() not in known:
            terms.append(t)
            known.add(t.lower())
            added.append(t)
    if added:
        with open(VOCAB_FILE, "w") as f:
            f.write(", ".join(terms) + "\n")
    print(f"added: {', '.join(added) or '(none, already present)'} -> {len(terms)} terms")


if __name__ == "__main__":
    main()
