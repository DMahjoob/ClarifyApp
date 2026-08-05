#!/usr/bin/env python3
"""
pdf_to_slide_jsonl.py

Converts a lecture-slide PDF into a retrieval-optimized JSONL dataset
(one JSON object per slide), matching the schema used for the CS102
decks (deck_name, slide_number, title, summary, main_text, notes_text,
keywords, images[], layout, metadata).

SETUP
-----
    pip install pymupdf anthropic
    export ANTHROPIC_API_KEY="sk-ant-..."

USAGE
-----
    python pdf_to_slide_jsonl.py input.pdf --deck-name CS102_Unit2b_Something \
        --course CS102 --unit 2b --out output.jsonl

    # Process every PDF in a folder:
    python pdf_to_slide_jsonl.py ./decks/*.pdf --course CS102 --out-dir ./out/
"""

import argparse
import base64
import glob
import json
import os
import re
import sys
import time

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("Missing dependency. Run: pip install pymupdf")

try:
    import anthropic
except ImportError:
    sys.exit("Missing dependency. Run: pip install anthropic")


SCHEMA_INSTRUCTIONS = """\
You are converting ONE slide from a lecture deck into a single structured JSON object \
for a retrieval-optimized dataset (semantic search / RAG for students and an AI \
classroom assistant).

You are given:
- An image of the rendered slide (look at it carefully -- read any diagrams, \
flowcharts, tables, code screenshots, or illustrations that appear).
- The raw text extracted from the slide's text layer (may include duplicated or \
out-of-order text due to PDF extraction quirks -- use it as a reference, not \
gospel; trust what you see in the image for layout and diagram content).

Return EXACTLY ONE JSON object (no markdown fences, no commentary, no preamble) \
matching this schema:

{
  "chunk_index": <int, 0-based; keep 0 unless this single slide's content is so \
long it genuinely needs to be split into multiple coherent chunks>,
  "title": "<short, human-readable slide title, e.g. 'Making Decisions'>",
  "summary": "<1-2 sentences, 20-40 words, describing the slide's purpose>",
  "main_text": "<all meaningful visible educational content: bullets, code, \
table contents, in your own transcription; 150-300 words typical, 500 max; \
exclude slide numbers, decorative headers/footers, copyright boilerplate>",
  "notes_text": "<optional deeper context, connections to other slides, or caveats; \
empty string if none>",
  "keywords": ["5 to 12 specific technical/topical terms"],
  "images": [
    {
      "description": "<natural-language description of WHAT this diagram/figure \
teaches, based on what you actually see in the image -- not a guess>",
      "labels": ["visible text labels inside the figure"],
      "position": {"x": <0-1>, "y": <0-1>, "width": <0-1>, "height": <0-1>}
    }
  ],
  "layout": {
    "num_text_boxes": <int, rough count of distinct text blocks/bullet groups>,
    "num_images": <int, count of entries in images[]>,
    "dominant_visual_type": "<one of: text-heavy | diagram | flowchart | mixed | comparison>"
  },
  "topic": "<the main conceptual topic this slide belongs to, e.g. 'Page Replacement Algorithms'>",
  "importance_score": <int 1-10, how central this slide is to the unit's core learning \
objectives vs. administrative/divider/backup content>
}

Rules:
- Do not hallucinate content that isn't visible on the slide or in the extracted text.
- If a diagram is illegible or ambiguous, say so briefly in "description" rather than \
inventing detail.
- position coordinates are normalized (0.0-1.0) estimates of where the figure sits on \
the slide canvas (0,0 = top-left).
- Keep main_text a faithful transcription/paraphrase of what's on the slide, not new \
commentary.
"""


def render_page(page, zoom=2.0):
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def extract_text(page):
    return page.get_text("text")


def call_claude_for_slide(client, model, image_bytes, page_text, page_number):
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    user_content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image_b64,
            },
        },
        {
            "type": "text",
            "text": (
                f"Slide / page number: {page_number}\n\n"
                f"Raw extracted text from this page's text layer:\n"
                f"-----\n{page_text}\n-----\n\n"
                "Produce the JSON object for this slide now, per the schema and "
                "rules in the system prompt. Return ONLY the JSON object."
            ),
        },
    ]

    resp = client.messages.create(
        model=model,
        max_tokens=2000,
        system=SCHEMA_INSTRUCTIONS,
        messages=[{"role": "user", "content": user_content}],
    )

    text_out = "".join(
        block.text for block in resp.content if getattr(block, "type", None) == "text"
    )
    text_out = text_out.strip()
    text_out = re.sub(r"^```json\s*|\s*```$", "", text_out).strip()
    return text_out


def build_entry(slide_json_str, deck_name, course, unit, slide_number):
    obj = json.loads(slide_json_str)

    return {
        "deck_name": deck_name,
        "slide_number": slide_number,
        "chunk_index": obj.get("chunk_index", 0),
        "title": obj["title"],
        "summary": obj["summary"],
        "main_text": obj["main_text"],
        "notes_text": obj.get("notes_text", ""),
        "keywords": obj.get("keywords", []),
        "images": obj.get("images", []),
        "layout": obj.get(
            "layout",
            {"num_text_boxes": 1, "num_images": 0, "dominant_visual_type": "text-heavy"},
        ),
        "metadata": {
            "course": course,
            "unit": unit,
            "topic": obj.get("topic", ""),
            "importance_score": obj.get("importance_score", 5),
            "file_hash": "sha256:PLACEHOLDER",
        },
    }


def process_pdf(pdf_path, deck_name, course, unit, out_path, model, max_retries=2):
    client = anthropic.Anthropic()

    doc = fitz.open(pdf_path)
    entries_written = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, page in enumerate(doc):
            slide_number = i + 1
            image_bytes = render_page(page)
            page_text = extract_text(page)

            attempt = 0
            while True:
                attempt += 1
                try:
                    raw = call_claude_for_slide(
                        client, model, image_bytes, page_text, slide_number
                    )
                    entry = build_entry(raw, deck_name, course, unit, slide_number)
                    break
                except Exception as e:
                    if attempt > max_retries:
                        print(
                            f"  [slide {slide_number}] FAILED after {attempt} "
                            f"attempts: {e}",
                            file=sys.stderr,
                        )
                        entry = {
                            "deck_name": deck_name,
                            "slide_number": slide_number,
                            "chunk_index": 0,
                            "title": f"{deck_name}_slide_{slide_number}_ERROR",
                            "summary": "Automatic processing failed for this "
                            "slide; review manually.",
                            "main_text": page_text[:1000],
                            "notes_text": f"Processing error: {e}",
                            "keywords": [],
                            "images": [],
                            "layout": {
                                "num_text_boxes": 0,
                                "num_images": 0,
                                "dominant_visual_type": "text-heavy",
                            },
                            "metadata": {
                                "course": course,
                                "unit": unit,
                                "topic": "",
                                "importance_score": 1,
                                "file_hash": "sha256:PLACEHOLDER",
                            },
                        }
                        break
                    time.sleep(2 * attempt)
                    continue

            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            entries_written += 1
            print(f"  [slide {slide_number}/{len(doc)}] -> {entry['title']}")

    doc.close()
    return entries_written


def main():
    parser = argparse.ArgumentParser(
        description="Convert lecture-slide PDFs to retrieval-optimized JSONL."
    )
    parser.add_argument(
        "pdfs", nargs="+", help="Path(s) to PDF file(s). Supports shell globs."
    )
    parser.add_argument(
        "--deck-name",
        help="Deck name to embed in each entry. Defaults to the PDF filename stem.",
    )
    parser.add_argument("--course", required=True, help="Course code, e.g. CS102")
    parser.add_argument("--unit", required=True, help="Unit label, e.g. 2b")
    parser.add_argument("--out", help="Output .jsonl path (single-PDF mode).")
    parser.add_argument(
        "--out-dir",
        help="Output directory (multi-PDF mode); one .jsonl per input PDF.",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Anthropic model to use (must support vision).",
    )
    args = parser.parse_args()

    pdf_paths = []
    for p in args.pdfs:
        pdf_paths.extend(glob.glob(p) or [p])

    if len(pdf_paths) > 1 and not args.out_dir:
        sys.exit("Multiple PDFs given: please specify --out-dir")
    if len(pdf_paths) == 1 and not args.out and not args.out_dir:
        sys.exit("Single PDF given: please specify --out (or --out-dir)")

    for pdf_path in pdf_paths:
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        deck_name = args.deck_name or stem
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            out_path = os.path.join(args.out_dir, f"{stem}.jsonl")
        else:
            out_path = args.out

        print(f"Processing {pdf_path} -> {out_path}")
        n = process_pdf(pdf_path, deck_name, args.course, args.unit, out_path, args.model)
        print(f"Done: {n} slide entries written to {out_path}\n")


if __name__ == "__main__":
    main()
