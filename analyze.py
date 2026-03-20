#!/usr/bin/env python3
"""
Analyze meeting transcripts using Claude API.
Produces: summary, TODOs, next-meeting goals, per-person feedback.
Writes markdown analysis files alongside transcripts in Obsidian vault.
"""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import anthropic

OBSIDIAN_TRANSCRIPTS = Path.home() / "playground/Obsidian/Transcripts"
MODEL = "claude-sonnet-4-6"

ANALYSIS_PROMPT = """You are analyzing a meeting transcript for Glen Berseth, a machine learning professor.
Glen runs a research lab (REAL) at Université de Montréal / Mila focused on RL and robot learning.
These meetings are typically 1:1s or group meetings with PhD students, postdocs, or collaborators.

Transcript filename: {filename}
Date: {date}

TRANSCRIPT:
{transcript}

Produce a structured markdown analysis with exactly these sections:

## Summary
2-4 sentences capturing the main purpose and outcomes of the meeting.

## Key Topics
Bullet list of the main technical/research topics discussed.

## Action Items
Checklist of concrete next steps. Format each as:
- [ ] **[Person]** Task description *(deadline if mentioned)*

If no clear action items, write "None identified."

## Open Questions
Research or technical questions raised but not resolved.

## Next Meeting Goals
Suggested agenda items for the next meeting based on open threads, pending work, and action items.

## Feedback & Suggestions
Per-person observations on research approach, communication, or productivity. Be specific and constructive.
Focus on patterns that would help them improve as researchers. Skip people with minimal speaking time.
Format as:
### [Person Name]
- Observation and suggestion

Keep the tone direct and collegial — Glen and his students are experts."""


def parse_date_from_filename(filename: str) -> str:
    match = re.match(r"(\d{4}-\d{2}-\d{2})", filename)
    return match.group(1) if match else "Unknown date"


def parse_meeting_title(filename: str) -> str:
    # Strip date and ID prefix: YYYY-MM-DD_ID_Title.txt -> Title
    name = Path(filename).stem
    parts = name.split("_", 2)
    if len(parts) >= 3:
        return parts[2].replace("_", " ")
    return name


def analysis_path_for(transcript_path: Path) -> Path:
    return transcript_path.with_name(transcript_path.stem + "_analysis.md")


def already_analyzed(transcript_path: Path) -> bool:
    return analysis_path_for(transcript_path).exists()


def analyze_transcript(transcript_path: Path, client: anthropic.Anthropic, force: bool = False) -> Path | None:
    if not force and already_analyzed(transcript_path):
        print(f"  [skip] {transcript_path.name} (already analyzed)")
        return None

    text = transcript_path.read_text(encoding="utf-8").strip()
    if len(text) < 200:
        print(f"  [skip] {transcript_path.name} (too short)")
        return None

    date = parse_date_from_filename(transcript_path.name)
    title = parse_meeting_title(transcript_path.name)

    print(f"  [analyzing] {transcript_path.name} ...")

    prompt = ANALYSIS_PROMPT.format(
        filename=transcript_path.name,
        date=date,
        transcript=text,
    )

    message = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    analysis_body = message.content[0].text

    # Build the full markdown file
    participants = extract_participants(text)
    participant_str = ", ".join(participants) if participants else "Unknown"

    output = f"""---
date: {date}
meeting: "{title}"
participants: [{participant_str}]
transcript: "[[{transcript_path.name}]]"
analyzed: {datetime.now().strftime("%Y-%m-%d")}
---

# {title} — {date}

{analysis_body}
"""

    out_path = analysis_path_for(transcript_path)
    out_path.write_text(output, encoding="utf-8")
    print(f"  [done] → {out_path.name}")
    return out_path


def extract_participants(transcript: str) -> list[str]:
    """Extract unique speaker names from transcript."""
    names = set()
    for line in transcript.splitlines():
        match = re.match(r"^([^:\[\n]{2,40}):", line)
        if match:
            name = match.group(1).strip()
            if name and name != "Meeting ID":
                names.add(name)
    return sorted(names)


def find_all_transcripts(root: Path) -> list[Path]:
    return sorted(root.rglob("*.txt"))


def main():
    parser = argparse.ArgumentParser(description="Analyze meeting transcripts with Claude")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Transcript file(s) or directory to analyze. Defaults to Obsidian Transcripts folder.",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-analyze even if analysis already exists",
    )
    parser.add_argument(
        "--recent", "-r",
        type=int,
        metavar="N",
        help="Only analyze the N most recently modified transcripts",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Collect transcript paths
    if args.paths:
        transcripts = []
        for p in args.paths:
            path = Path(p)
            if path.is_dir():
                transcripts.extend(find_all_transcripts(path))
            elif path.suffix == ".txt":
                transcripts.append(path)
            else:
                print(f"Warning: skipping {p} (not a .txt file or directory)")
    else:
        transcripts = find_all_transcripts(OBSIDIAN_TRANSCRIPTS)

    if args.recent:
        transcripts = sorted(transcripts, key=lambda p: p.stat().st_mtime, reverse=True)[: args.recent]

    if not transcripts:
        print("No transcripts found.")
        return

    print(f"Found {len(transcripts)} transcript(s).\n")
    analyzed = 0
    for t in transcripts:
        result = analyze_transcript(t, client, force=args.force)
        if result:
            analyzed += 1

    print(f"\nDone. Analyzed {analyzed} transcript(s).")


if __name__ == "__main__":
    main()
