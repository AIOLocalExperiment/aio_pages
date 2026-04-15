import os, argparse
import pandas as pd
from openai import OpenAI
import sys

# export OPENAI_API_KEY="YOUR_KEY"
# python 01_format_text.py --sample=5

INPUT_CSV = "full_samples/retrievals.csv"
OUTPUT_CSV = "full_samples/retrievals_formatted.csv"

DEV = """
You are a text formatter. 
Your only job is to add HTML structure to plain text so it visually resembles a Google AI Overview snippet. 
You never rewrite, reorder, add, or delete any words from the input. 
Formatting tags and whitespace are the only permitted additions.

You format text into an HTML fragment to be inserted inside:
<p class="T286Pc">{HERE}</p>

Output ONLY the HTML fragment for {HERE}. Do NOT include <p>, <html>, <body>, <div>, or any commentary.

Allowed tags only: <br>, <b>, <ul>, <ol>, <li>, <h3>, <h4>.
No attributes. No markdown.
"""

# USR = """
# Format the INPUT TEXT below into a structured HTML snippet resembling a Google AI Overview layout.

# ## Allowed HTML Tags
# `<h2>`, `<h3>`, `<p>`, `<ul>`, `<li>`, `<strong>`

# ## Rules

# ### Content Integrity
# - Do not rephrase, reorder, add, or delete any words or punctuation from the input.
# - The original text must appear in full, in its original order, inside the output HTML.

# ### Headings (`<h2>`, `<h3>`)
# Apply a heading tag when ALL of these conditions are true:
# 1. The phrase is short (roughly 8 words or fewer).
# 2. It ends with a colon, question mark, or stands alone on its own line.
# 3. It is immediately followed by a list or a new paragraph.

# Examples of valid headings: "Key Takeaways", "Who is eligible?", "In Summary:"
# Do not wrap full sentences or long clauses in heading tags.

# ### Lists (`<ul>`, `<li>`)
# Convert consecutive sentences into a list when they follow one of these patterns:
# - Each sentence begins with a short label followed by a colon (e.g., "Eligibility: You must be…").
# - A sequence of short, parallel sentences that logically enumerate items.

# Do not force unrelated sentences into a list just because they are adjacent.

# ### Bold (`<strong>`)
# - Inside a `<li>`, if the item starts with a short label (1–4 words) followed by a colon, wrap ONLY that label in `<strong>`. Leave everything after the colon unbolded.
# - Never bold full sentences, full list items, or labels longer than 4 words.
# - Ignore colons that appear mid-sentence or are part of normal grammar (e.g., "users who: registered early").

# ### Spacing and Line Breaks
# - Each paragraph is its own `<p>` tag. Do not insert blank lines between any block-level elements (`<p>`, `<ul>`, `<h2>`, `<h3>`). Output them on consecutive lines with no blank lines between them.
# - Do not add space or `<br>` tags between bullet points (`<li>` elements inside the same `<ul>`).
# - Do not insert line breaks after a heading.
# - Do NOT add line breaks between bullet points or list items. 

# ## INPUT TEXT:
# """

USR = """
Format the INPUT TEXT to resemble a Google AI Overview layout using headings, bullets, bold, spaces, and line breaks.

CRITICAL CONSTRAINT:
- Preserve the input text verbatim.
- Do not add, remove, reorder, or modify any words, characters, or punctuation.
- The output must contain all original characters in the same order.
- Only insert HTML tags (from the allowed list) and whitespace/line breaks to structure the text.
- Headings may only wrap existing text; do not invent new titles.
- Bullet points should be used when there is a list of sentences that follow the structure "lorem ipsum: sentence follows." OR a sequence of short sentences that logically appear to form a list.
- In list items or summary sections, bold ONLY the words BEFORE the colon. 
- Do NOT bold full sentences. Do NOT bold full list items. For example, a sentence like "Use apps like Google Maps and set your route to see toll costs if available." should NOT be bold
- Do NOT add line breaks between bullet points or list items. 
- Only include a single space between paragraphs and do not add a space between heading and bullet points. 
- Treat summary phrases with a colon as headings. For example: "Verdict:", "In Summary:", "Conclusion:"
- Only assign headings to standalone phrases followed by lists (e.g. "Who is eligible for a free card?", 
"Key Information and Statistics", "Key Takeaways").
- Do not add punctuation.

INPUT TEXT:
"""

IN_PER_1M  = 0.40
OUT_PER_1M = 1.60

def est_cost(totals):
    return (totals["in"] * IN_PER_1M + totals["out"] * OUT_PER_1M) / 1_000_000

def fmt(text, client, model, totals):
    text = "" if pd.isna(text) else str(text)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "developer", "content": DEV},
            {"role": "user", "content": f"{USR}\n{text}"},
        ],
        temperature=0.2,
    )

    if getattr(resp, "usage", None):
        totals["in"] += resp.usage.prompt_tokens or 0
        totals["out"] += resp.usage.completion_tokens or 0

    return (resp.choices[0].message.content or "").rstrip("\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--progress-every", type=int, default=1)
    ap.add_argument("--sample", type=int, default=None)

    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    df = pd.read_csv(INPUT_CSV)
    if args.sample:
        df = df.head(args.sample)

    if "aio_text" not in df.columns:
        raise SystemExit(f"Missing column aio_text. Found: {list(df.columns)}")

    client = OpenAI()
    totals = {"in": 0, "out": 0}

    n = len(df)
    out = []

    for i, text in enumerate(df["aio_text"].tolist(), start=1):
        out.append(fmt(text, client=client, model=args.model, totals=totals))

        if args.progress_every > 0 and (i % args.progress_every == 0 or i == n):
            cost = est_cost(totals)
            sys.stderr.write(
                f"\rRow {i}/{n} | in={totals['in']} out={totals['out']} | cumulative_est=${cost:.6f}   "
            )
            sys.stderr.flush()

    sys.stderr.write("\n")

    df["formatted_text"] = out
    df.to_csv(OUTPUT_CSV, index=False)

    final_cost = est_cost(totals)
    print(f"Total estimated cost: ${final_cost:.6f}")
    print(f"Done. Wrote: {OUTPUT_CSV}")
    print(f"Input tokens:  {totals['in']}")
    print(f"Output tokens: {totals['out']}")

if __name__ == "__main__":
    main()