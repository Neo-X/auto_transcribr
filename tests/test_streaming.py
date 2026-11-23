#!/usr/bin/env python3
"""Regression test for dictate.py: accuracy of the speech model, and of transcribing while recording.

Run from the repository root (it loads the real model on the GPU, so it takes minutes):

    uv run --no-sync python tests/test_streaming.py --quick            # about 5 minutes
    uv run --no-sync python tests/test_streaming.py                    # every clip, 40+ minutes
    uv run --no-sync python tests/test_streaming.py --update-baseline  # accept today's accuracy

Clips:
  * labelled  — ~/playground/asr-whisper-finetuning/data/dictation: dictations with a corrected
                reference text (these are clips the model once got wrong, so their word error
                rate is high by construction; what matters is that it does not get worse).
  * staged    — /tmp/dictate-clips: recent dictations with no reference (wiped on reboot). Only
                clips of at least --min-staged-s are used; they are there to compare the
                while-recording text with the one-go text on long recordings.

--quick runs a fixed sample of the labelled clips, sized to take about five minutes on the
laptop and spread over clip lengths (listed in tests/quick_clips.json, which is written the
first time and then kept, so every quick run uses the same clips; delete it to re-draw).
Run it first; run every clip only when the quick run looks good.

For each clip the audio is transcribed in one go, then again the way dictate.py does it during
a recording: the test replays the clip against a clock, giving dictate._Stream the audio
recorded "so far" at the moments the real background thread would have asked for it.

Checks (exit status 1 if any fails):
  1. One-go word error rate on the labelled clips is no worse than tests/baseline.json
     (compared on the clips both runs share).
  2. While-recording word error rate is less than 1 point above one-go on the same labelled clips.
  3. While-recording text stays close to the one-go text on all long clips.
  4. Clips too short for a pass during recording give exactly the one-go text.
  5. On long clips the wait after release is much shorter than in one go.
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
p.add_argument("--dataset", default=os.path.expanduser("~/playground/asr-whisper-finetuning/data/dictation"))
p.add_argument("--staged", default="/tmp/dictate-clips")
p.add_argument("--min-staged-s", type=float, default=20.0, help="shortest staged clip to use")
p.add_argument("--limit", type=int, default=0, help="use only the N longest clips of each set")
p.add_argument("--quick", action="store_true", help="fixed five-minute sample of the labelled clips")
p.add_argument("--quick-budget-s", type=float, default=310.0,
               help="estimated transcription time to fill when drawing the quick sample")
p.add_argument("--update-baseline", action="store_true", help="write tests/baseline.json from this run")
p.add_argument("--out", default=os.path.join(ROOT, "tests", "results"))
p.add_argument("--wer-slack", type=float, default=0.5,
               help="allowed rise in one-go word error rate over the baseline, in points")
p.add_argument("--stream-slack", type=float, default=1.0,
               help="allowed rise in word error rate from transcribing while recording, in points "
                    "(under 1 point is worth the shorter wait)")
p.add_argument("--max-drift", type=float, default=4.0,
               help="allowed word difference between while-recording and one-go text, in percent")
p.add_argument("--max-wait-ratio", type=float, default=0.5,
               help="while-recording wait after release / one-go wait, on clips of 30 s or more")
args, _ = p.parse_known_args()

import dictate  # noqa: E402  (after sys.path; importing does not load the model)

SR = dictate.SAMPLE_RATE
BASELINE = os.path.join(ROOT, "tests", "baseline.json")
QUICK = os.path.join(ROOT, "tests", "quick_clips.json")
# Share of the quick run's time given to each clip-length band (seconds). Long clips get more
# than their share of the dataset because they are the ones transcribed while recording.
QUICK_BANDS = [(0, 10, 0.10), (10, 20, 0.25), (20, 30, 0.25), (30, 45, 0.15), (45, 1e9, 0.25)]


def words(text: str) -> list[str]:
    text = text.lower().replace("-", " ")
    return re.sub(r"[^a-z0-9' ]", "", text).split()


def errors(ref: list[str], hyp: list[str]) -> int:
    """Word-level edit distance."""
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1]


def rate(pairs) -> float:
    """Pooled word error rate, in percent, over (reference, hypothesis) texts."""
    pairs = [(words(a), words(b)) for a, b in pairs]
    total = sum(len(a) for a, _ in pairs)
    return 100.0 * sum(errors(a, b) for a, b in pairs) / total if total else 0.0


def load_wav(path: str) -> np.ndarray:
    with wave.open(path) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1, path
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32767


def read_clips(folder: str, min_s: float, need_reference: bool) -> list[dict]:
    meta = os.path.join(folder, "metadata.jsonl")
    if not os.path.exists(meta):
        print(f"[skip] no {meta}")
        return []
    clips = {}
    for line in open(meta):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        path = os.path.join(folder, r.get("file_name", ""))
        if r.get("duration_s", 0) < min_s or not os.path.exists(path):
            continue
        if need_reference and not r.get("transcription"):
            continue
        clips[r["id"]] = {"id": r["id"], "path": path, "duration_s": r["duration_s"],
                          "reference": r.get("transcription", "") if need_reference else ""}
    out = sorted(clips.values(), key=lambda c: -c["duration_s"])
    return out[: args.limit] if args.limit else out


def quick_sample(labelled: list[dict]) -> list[dict]:
    """The fixed quick-run clips: read from QUICK, or drawn (and saved) if it does not exist.

    Drawing: each length band gets a share of the time budget and is filled with clips spread
    evenly over the band's range of durations. A clip is estimated to cost 0.45 s per second
    of audio plus 1 s (one-go plus replay, measured on the laptop's GPU)."""
    by_id = {c["id"]: c for c in labelled}
    if os.path.exists(QUICK):
        ids = json.load(open(QUICK))["clips"]
        missing = [i for i in ids if i not in by_id]
        if missing:
            print(f"[warn] {len(missing)} quick clips are no longer in the dataset")
        return [by_id[i] for i in ids if i in by_id]
    cost = lambda c: 0.45 * c["duration_s"] + 1.0
    chosen = []
    for lo, hi, share in QUICK_BANDS:
        band = sorted((c for c in labelled if lo <= c["duration_s"] < hi), key=lambda c: c["duration_s"])
        budget = share * args.quick_budget_s
        n = len(band)
        while n > 1 and sum(cost(band[round(k * (len(band) - 1) / (n - 1))]) for k in range(n)) > budget:
            n -= 1
        picks = {round(k * (len(band) - 1) / (n - 1)) for k in range(n)} if n > 1 else ({len(band) // 2} if band else set())
        chosen += [band[k] for k in sorted(picks)]
    json.dump({"note": "Fixed clips for `test_streaming.py --quick`; delete this file to re-draw.",
               "estimated_s": round(sum(map(cost, chosen))), "clips": [c["id"] for c in chosen]},
              open(QUICK, "w"), indent=1)
    print(f"Drew {len(chosen)} quick clips -> {QUICK}")
    return chosen


def replay(transcribe, audio: np.ndarray) -> dict:
    """Transcribe `audio` as dictate.py would during a recording of it, on a simulated clock.

    Mirrors _Stream.follow(): a pass starts as soon as the previous one is done and
    STREAM_STEP_S of new audio has been recorded, and sees the audio recorded by then."""
    stream = dictate._Stream(transcribe)
    duration = len(audio) / SR
    free_at = 0.0  # clock time at which the background thread is idle again
    while True:
        start = max(free_at, stream.covered / SR + dictate.STREAM_STEP_S)
        if start >= duration:
            break
        t0 = time.perf_counter()
        stream.step(audio[: int(start * SR) // 1024 * 1024])
        free_at = start + time.perf_counter() - t0
    t0 = time.perf_counter()
    text = stream.finish(audio)
    wait = max(free_at - duration, 0.0) + time.perf_counter() - t0
    return {"text": text, "wait_s": wait, "passes": stream.passes}


def check_stream_logic():
    """No model needed: _Stream keeps all but the last STREAM_HOLD tokens and passes them on."""
    calls = []

    def fake(audio, prefix=None, return_ids=False):
        calls.append(list(prefix or []))
        ids = list(range(len(audio) // SR * 3))  # 3 tokens per second of audio
        assert ids[: len(prefix or [])] == list(prefix or []), "prefix must be the start of the answer"
        return " ".join(map(str, ids)), ids

    stream = dictate._Stream(fake)
    one_second = np.zeros(SR, np.float32)
    stream.step(np.tile(one_second, 10))
    assert stream.kept == list(range(30 - dictate.STREAM_HOLD)), stream.kept
    stream.step(np.tile(one_second, 10))  # nothing new: must not lose settled tokens
    assert stream.kept == list(range(30 - dictate.STREAM_HOLD))
    text = stream.finish(np.tile(one_second, 20))
    assert calls[-1] == list(range(30 - dictate.STREAM_HOLD)) and text == " ".join(map(str, range(60)))
    print("[ok] _Stream bookkeeping")


def main() -> int:
    check_stream_logic()
    labelled = read_clips(args.dataset, 0.0, need_reference=True)
    labelled_ids = {c["id"] for c in labelled}
    staged = [c for c in read_clips(args.staged, args.min_staged_s, need_reference=False)
              if c["id"] not in labelled_ids]
    if args.quick:
        labelled, staged = quick_sample(labelled), []
    clips = labelled + staged
    if not clips:
        print("No clips found; nothing to test.")
        return 1
    total = sum(c["duration_s"] for c in clips)
    print(f"{len(labelled)} labelled + {len(staged)} staged clips, {total / 60:.0f} min of audio; "
          f"step {dictate.STREAM_STEP_S:g} s, hold {dictate.STREAM_HOLD} tokens")

    transcribe = dictate._load_model()
    if not getattr(transcribe, "supports_prefix", False):
        print(f"Model {dictate.MODEL} cannot transcribe while recording; nothing to test.")
        return 1
    transcribe(np.zeros(SR * 3, np.float32))  # warm-up
    started = time.perf_counter()

    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rows_path = os.path.join(args.out, f"streaming_{stamp}.jsonl")
    rows = []
    with open(rows_path, "w") as f:
        for i, c in enumerate(clips, 1):
            audio = load_wav(c["path"])
            t0 = time.perf_counter()
            one_go = transcribe(audio)
            one_go_s = time.perf_counter() - t0
            r = replay(transcribe, audio)
            row = {**c, "one_go": one_go, "one_go_wait_s": round(one_go_s, 2), "streamed": r["text"],
                   "streamed_wait_s": round(r["wait_s"], 2), "passes": r["passes"],
                   "drift_words": errors(words(one_go), words(r["text"]))}
            del row["path"]
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"\r{i}/{len(clips)}", end="", flush=True)
    print()

    lab = [r for r in rows if r["reference"]]
    lab_long = [r for r in lab if r["passes"]]
    long_ = [r for r in rows if r["passes"]]
    short = [r for r in rows if not r["passes"]]
    over30 = [r for r in rows if r["duration_s"] >= 30]
    wer_one = rate((r["reference"], r["one_go"]) for r in lab)
    wer_one_long = rate((r["reference"], r["one_go"]) for r in lab_long)
    wer_stream_long = rate((r["reference"], r["streamed"]) for r in lab_long)
    drift = rate((r["one_go"], r["streamed"]) for r in long_)
    short_same = sum(r["one_go"] == r["streamed"] for r in short)
    med = lambda xs: statistics.median(xs) if xs else 0.0
    wait_one, wait_stream = med([r["one_go_wait_s"] for r in over30]), med([r["streamed_wait_s"] for r in over30])

    # The baseline holds each labelled clip's one-go errors, so the comparison is made on the
    # clips both runs share and still works after clips are added to the dataset.
    per_clip = {r["id"]: [errors(words(r["reference"]), words(r["one_go"])), len(words(r["reference"]))] for r in lab}
    baseline = json.load(open(BASELINE)).get("clips", {}) if os.path.exists(BASELINE) else {}
    shared = [i for i in per_clip if i in baseline and per_clip[i][1] == baseline[i][1]]
    n_words = sum(per_clip[i][1] for i in shared) or 1
    wer_now = 100.0 * sum(per_clip[i][0] for i in shared) / n_words
    wer_base = 100.0 * sum(baseline[i][0] for i in shared) / n_words
    checks = [
        ("one-go error rate vs baseline",
         not shared or wer_now <= wer_base + args.wer_slack,
         (f"{wer_now:.2f}% vs baseline {wer_base:.2f}% on the {len(shared)} clips in both" if shared
          else f"{wer_one:.2f}% on {len(lab)} labelled clips (no baseline to compare with)")),
        ("while-recording error rate vs one-go", wer_stream_long < wer_one_long + args.stream_slack,
         f"{wer_stream_long:.2f}% vs {wer_one_long:.2f}% on {len(lab_long)} labelled clips long enough "
         f"(limit +{args.stream_slack:g} points)"),
        ("while-recording text close to one-go text", drift <= args.max_drift,
         f"{drift:.2f}% of words differ on {len(long_)} clips (limit {args.max_drift:g}%)"),
        ("short clips unchanged", short_same == len(short), f"{short_same}/{len(short)} identical"),
        ("wait after release on clips of 30 s or more",
         not over30 or wait_stream <= args.max_wait_ratio * wait_one,
         f"median {wait_stream:.1f} s vs {wait_one:.1f} s in one go, {len(over30)} clips"),
    ]
    lines = [f"{'PASS' if ok else 'FAIL'}  {name}: {detail}" for name, ok, detail in checks]
    worst = sorted(long_, key=lambda r: -r["drift_words"])[:5]
    lines += ["", "Largest differences between while-recording and one-go text:"]
    lines += [f"  {r['id']} ({r['duration_s']:.0f} s): {r['drift_words']} words" for r in worst]
    lines += ["", f"{len(rows)} clips in {(time.perf_counter() - started) / 60:.1f} min. Per-clip results: {rows_path}"]
    print("\n".join(lines))
    with open(os.path.join(args.out, f"streaming_{stamp}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    if args.update_baseline:
        if args.limit or args.quick:
            print("Not writing a baseline from a --limit or --quick run.")
        else:
            json.dump({"model": dictate.MODELS[dictate.MODEL], "date": time.strftime("%Y-%m-%d"),
                       "one_go_wer": round(wer_one, 2), "clips": per_clip}, open(BASELINE, "w"), indent=0)
            print(f"Baseline written: {BASELINE}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
