import os, argparse, re, html as html_lib
import pandas as pd
from openai import OpenAI
import sys

# export OPENAI_API_KEY="YOUR_KEY"
# python 01_format_text.py --sample=5 --max-retries=2
# python 01_format_text.py --retry-from full_samples/retrievals_formatted.csv --max-retries 2
# python 01_format_text.py --retrieval-id 8f488f66-9c40-48a2-903e-4136c66d66c6 --max-retries 2

INPUT_CSV = "full_samples/retrievals.csv"
OUTPUT_CSV = "full_samples/retrievals_formatted.csv"
RETRIEVAL_ID_COL = "retrieval_id"

DEV = """
You are a strict text-to-HTML formatter. Your job is to add HTML structure
to a plain-text passage so it visually resembles a Google AI Overview answer
card, without altering the text itself.

# Output contract
Return ONLY the inner HTML to be inserted between <p class="T286Pc"> and </p>.
Do not output <p>, <html>, <body>, <div>, code fences, markdown, or commentary.

Allowed tags (no attributes, no other tags ever):
  <h3>      top-level heading
  <h4>      sub-heading inside an <h3> section
  <ul><li>  bulleted list
  <b>       bold a short label inside <li>, or an inline summary lead-in
  <ol><li>  numbered list — use ONLY when the input itself contains
            an explicit numeric enumeration ("1. ...  2. ...  3. ...",
            or "1) ..., 2) ...").

Disallowed: <p>, <br>, <ol>, <strong>, <em>, <hr>, <a>, attributes, classes.

# Verbatim rule (highest priority — overrides every rule below)
The input passage must appear in the output character-for-character, in the
original order, with the SAME punctuation and capitalization.
- You may insert HTML tags and whitespace between characters of the input.
- You may NOT add, remove, reorder, replace, merge, or split any character
  that is not part of a tag. This includes periods, commas, colons, quotes,
  dashes, parentheses, digits, and letters. Do not "fix" typos or spacing.
- Straight quotes stay straight. Curly quotes stay curly. Do not normalize.
- Whitespace-preservation corollary: any pair of input characters that
  were separated by whitespace (space, tab, or newline) MUST remain
  separated in your output — by either whitespace or a block-level tag
  (which renders as a boundary). You may swap a newline for a space, or
  for a tag boundary, but you may not delete it outright. Concatenating
  two sentences with no separator (e.g. "marriage.Here's") is a verbatim
  violation.
- Tag boundaries as visual separators: where the input lacks whitespace
between two phrases that are clearly distinct (e.g., a parenthetical
fragment butted against a new capitalized sentence, a colon before a new capitalized sentence),
inserting a block-level tag boundary like </h3><h4> or </h3><ul>... is allowed
and preferred over wrapping both phrases together.

# When to wrap text in <h3>
Wrap a phrase in <h3> if EITHER Pattern A or Pattern B holds.

Pattern A — explicit label / question (current rule):
  All three must be true:
  1. The phrase is <= 15 words.
  2. It already exists in the input verbatim as a standalone label, line,
     or sentence fragment ending in ":" or "?".
  3. It is immediately followed by a list OR by explanatory text that
     clearly belongs to it.
  Typical patterns: "Key Takeaways", "Who is eligible?",
  "Key Information and Statistics".

Pattern B — unpunctuated topical fragment (start-of-paragraph OR orphan):
  Rules 1, 2, and 5 must all hold, AND either Rule 3 (start-of-paragraph)
  OR Rule 4 (orphan-at-end) must hold.
  1. The phrase is <= 8 words and is a NOUN PHRASE with no main verb
     (not a complete sentence, not a clause). Typical shapes:
       "<Topic>'s history", "History of <Topic>", "About <Topic>",
       "Overview of <Topic>", "Background on <Topic>",
       "Key facts about <Topic>", "<Topic> explained",
       "Key Considerations for <Topic>".
  2. The fragment is in Title Case — the majority of content words
     (ignoring short function words like "of", "for", "in", "the",
     "a", "an", "and", "to", "on", "by") start with a capital letter.
     Sentence-case fragments ("Key considerations for living in west
     palm beach") do NOT qualify under Pattern B.
  3. START-OF-PARAGRAPH position. The fragment sits at the very start
     of a paragraph and is followed immediately by a new capitalized
     word that begins a self-contained sentence — the fragment is NOT
     the grammatical subject of what follows. DELETION TEST: if you
     delete the fragment, the remaining text must still be a complete,
     coherent paragraph. The sentence that follows is clearly about
     the same topic. If deletion breaks the grammar, do NOT wrap.
  4. ORPHAN-AT-END position. The fragment is preceded by sentence-final
     punctuation (".", "!", "?") and is followed by the end of the
     input (or by a paragraph break with nothing further on its topic).
     This handles the upstream-truncation case where a heading's list
     was dropped before the formatter ran, leaving a stranded title
     at the tail of the passage.
     Example input:
       "...to more affordable areas. Key Considerations for Living
        in West Palm Beach"
     Example output:
       "...to more affordable areas.<h3>Key Considerations for Living
        in West Palm Beach</h3>"
  5. Nothing on the same line precedes the fragment and flows into it.

Pattern C — recognized summary lead-in:
  If a sentence begins with one of the recognized summary lead-in
  phrases ("In Summary:", "Verdict:", "Conclusion:", "Bottom Line:",
  "The Bottom Line:", "Key Takeaway:"), wrap ONLY that lead-in phrase
  (including the colon) in <h3>, then continue the wrap-up sentence
  immediately after.
  Example:
    Input:  "...enjoying the community's charm. In Summary: It's a
            relatively peaceful town..."
    Output: "...enjoying the community's charm.<h3>In Summary:</h3>It's
            a relatively peaceful town..."

Punctuation belongs INSIDE the heading. If the input phrase ends in
":", "?", or "!", that character must appear before </h3>, never after.
  Correct:   <h3>Local Hillsboro Options (Check Hours!):</h3>
  Incorrect: <h3>Local Hillsboro Options (Check Hours!)</h3>:
  Incorrect: <h3>Local Hillsboro Options (Check Hours!)</h3>     ← colon deleted

  Example input:
    "Grant City's history Grant City, Missouri was laid out in 1864
     and named after General Ulysses S. Grant..."
  Correct output:
    "<h3>Grant City's history</h3>Grant City, Missouri was laid out
     in 1864 and named after General Ulysses S. Grant..."

If you are unsure whether something is Pattern B, leave it unwrapped.
False headers are worse than missed headers.

Use <h4> only for a labeled sub-section nested under an <h3>.
Most outputs will need zero or one <h3> and no <h4>.

# When to wrap text in <ul><li>
Convert consecutive sentences into a list only if EITHER pattern holds:
  A) Each sentence opens with a short label followed by ":"
     ("Eligibility: ...", "Cost: ...").
  B) Three or more short, parallel sentences that read as enumeration.
Never list a single sentence. Never split one sentence across multiple <li>.
Never group unrelated sentences just because they are adjacent.

# When to wrap text in <ol><li>
Use <ol> only when the input contains an explicit numeric enumeration
("1. ... 2. ... 3. ..." or "1) ..., 2) ..."). Keep the numeric prefix
inside the <li> verbatim — never strip it, never add one.

If a numbered item's body contains two or more "Label: text." segments
after the item's title, split each segment into its own <li> in a
nested <ul>, with the label bolded per the standard label rule. A new
sub-field starts wherever a fresh "Label:" appears, even when no
whitespace separates it from the previous segment in the input.

Example:
  Input:  "1. Determine Where the Death Occurred Died in NYC
           (5 Boroughs): Contact NYC Health Department. Died in New
           York State (Outside NYC): Contact the NYS Department of
           Health."
  Output: <ol><li>1. Determine Where the Death Occurred<ul><li><b>Died in NYC (5 Boroughs)</b>: Contact NYC Health Department.</li><li><b>Died in New York State (Outside NYC)</b>: Contact the NYS Department of Health.</li></ul></li></ol>

Never use <ol> for plain prose, never split or merge items.


# When to wrap text in <b>
Use <b> in exactly two situations:
  A) Inside an <li>: if the item begins with a label followed by ":", wrap
     ONLY that label in <b>. The label is everything before the first ":"
     of the item. Labels may include parenthetical qualifiers such as
     dates, centuries, or short clarifications.
     Labels are typically 2-10 words including any parenthetical.
     Examples:
       <li><b>Eligibility</b>: You must be 18+.</li>
       <li><b>Early Growth (Late 19th Century)</b>: Became a vital rail center.</li>
       <li><b>The Texas Centennial (1936)</b>: The massive exposition...</li>
    A label is any phrase of 2–10 words preceding the first : of an <li>,
    including proper-noun names, place names, and organization names with
    parenthetical qualifiers. Bold it. The 'mid-sentence colon' exception applies
    only when the text before : is a complete clause with subject and verb
    (e.g., 'The bill was signed in 1872:').
    If the colon falls mid-sentence rather than between a label and its content, do not bold.
  B) Inline summary lead-in: at the start of a sentence or paragraph that
     introduces a wrap-up, wrap ONLY the short lead-in phrase in <b>.
     Leave the colon and the rest unbolded. Recognized lead-ins include
     "In Summary:", "Verdict:", "Conclusion:", "Bottom Line:",
     "Key Takeaway:", "The Bottom Line:", and similar short summary tags
     ending in ":".
     Example: <h3>In Summary:</h3>most low-income adults qualify.
Never bold a full sentence or a full <li>. Never use <b> for emphasis on
arbitrary words mid-sentence.

# Whitespace (CRITICAL — the renderer converts every newline to <br/>)
Output the entire HTML fragment on ONE LINE. Use NO newlines anywhere in
your output.

Tag-to-tag joins: concatenate adjacent tags directly, with no space
between them.
  Right: </h3><ul><li>...</li><li>...</li></ul>
  Wrong: </h3>\n<ul>\n<li>...</li>\n<li>...</li>\n</ul>
  Wrong: </li> <li>

Text-to-text joins: NEVER delete a separator between two words or two
sentences. Wherever the input had whitespace (space, newline, or tab)
between two pieces of text, keep at least one space in your output.
  Input:  "...healthcare.\nWho is eligible..."
  Right:  "...healthcare. Who is eligible..."
  Wrong:  "...healthcare.Who is eligible..."

In short: tags can touch each other; words cannot.

# Self-check before responding
Verify all four:
  1. Stripping every tag from your output yields the input, allowing only
     whitespace differences.
  2. Every <b> is either inside an <li> wrapping the segment before the
     item's first ":", or wraps a recognized summary lead-in phrase at the
     start of a sentence. <b> never wraps a full sentence or full <li>.
  3. Every <h3>/<h4> wraps an existing input phrase of <= 8 words.
  4. No disallowed tag is present.
  5. If the input contains "1. ... 2. ... 3. ..." or "1) ..., 2) ...",
   the output uses <ol><li>, not <ul><li>. Each <li> retains the
   numeric prefix verbatim. Internal "Label:" sub-fields inside an
   item are nested as <ul><li><b>Label</b>: ...</li></ul>.
If any check fails, revise before answering.
"""

USR = "Format the INPUT TEXT per your instructions. Output only the HTML fragment.\n\nINPUT TEXT:\n"

# Few-shot examples, sent as alternating user/assistant turns. This is
# more effective than embedding examples in the system prompt. Two demos:
# (1) flat <ul> with bolded labels; (2) <ol> with a nested <ul> of bolded
# sub-fields inside each numbered item.
FEWSHOT = [
    {
        "role": "user",
        "content": (
            "Format the INPUT TEXT per your instructions. Output only the HTML fragment.\n\n"
            "INPUT TEXT:\n"
            "Who is eligible for a free card? Eligibility: You must be 18 or older. "
            "Residency: You must live in the US. Income: Your household income must be "
            "below $50,000. In Summary: most low-income adults qualify."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "<h3>Who is eligible for a free card?</h3>"
            "<ul>"
            "<li><b>Eligibility</b>: You must be 18 or older.</li>"
            "<li><b>Residency</b>: You must live in the US.</li>"
            "<li><b>Income</b>: Your household income must be below $50,000.</li>"
            "</ul>"
            "<b>In Summary</b>: most low-income adults qualify."
        ),
    },
    {
        "role": "user",
        "content": (
            "Format the INPUT TEXT per your instructions. Output only the HTML fragment.\n\n"
            "INPUT TEXT:\n"
            "1. Determine Where the Death Occurred Died in NYC (5 Boroughs): "
            "Contact NYC Health Department. Died in New York State (Outside NYC): "
            "Contact the NYS Department of Health. "
            "2. Gather Required Documents Proof of Relationship: Birth or marriage "
            "certificate. Photo ID: Government-issued."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "<ol>"
            "<li>1. Determine Where the Death Occurred"
            "<ul>"
            "<li><b>Died in NYC (5 Boroughs)</b>: Contact NYC Health Department.</li>"
            "<li><b>Died in New York State (Outside NYC)</b>: Contact the NYS Department of Health.</li>"
            "</ul>"
            "</li>"
            "<li>2. Gather Required Documents"
            "<ul>"
            "<li><b>Proof of Relationship</b>: Birth or marriage certificate.</li>"
            "<li><b>Photo ID</b>: Government-issued.</li>"
            "</ul>"
            "</li>"
            "</ol>"
        ),
    },
]

IN_PER_1M  = 0.40
OUT_PER_1M = 1.60

# ----- validation helpers ---------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_INLINE_TAG_RE = re.compile(r"</?b\b[^>]*>", re.I)
_WS_RE = re.compile(r"\s+")

def strip_html(s: str) -> str:
    """Remove HTML tags and decode entities. Inline tags (<b>) strip to
    nothing — otherwise '<b>Foo</b>:' becomes 'Foo :' and falsely fails.
    Block-level tags strip to a space so adjacent words don't concatenate."""
    s = _INLINE_TAG_RE.sub("", s)
    s = _TAG_RE.sub(" ", s)
    s = html_lib.unescape(s)
    return s

def normalize_for_compare(s: str) -> str:
    """Collapse whitespace runs to a single space and strip ends. The prompt
    explicitly allows added whitespace, so we ignore whitespace differences."""
    return _WS_RE.sub(" ", s).strip()

def validate(original: str, formatted: str):
    """Return (is_valid, normalized_original, normalized_stripped_output)."""
    a = normalize_for_compare(original)
    b = normalize_for_compare(strip_html(formatted))
    return (a == b, a, b)

def first_diff(a: str, b: str, window: int = 40) -> str:
    """Short human-readable indicator of where two strings diverge."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    start = max(0, i - window)
    return (
        f"@char {i}: "
        f"orig=...{a[start:i+window]!r} "
        f"got=...{b[start:i+window]!r}"
    )

# ----- model call -----------------------------------------------------------

def est_cost(totals):
    return (totals["in"] * IN_PER_1M + totals["out"] * OUT_PER_1M) / 1_000_000

def fmt(text, client, model, totals):
    text = "" if pd.isna(text) else str(text)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "developer", "content": DEV},
            *FEWSHOT,
            {"role": "user", "content": f"{USR}{text}"},
        ],
        temperature=0.2,
    )

    if getattr(resp, "usage", None):
        totals["in"] += resp.usage.prompt_tokens or 0
        totals["out"] += resp.usage.completion_tokens or 0

    return (resp.choices[0].message.content or "").rstrip("\n")

def fmt_with_validation(text, client, model, totals, max_retries):
    """Call fmt, validate, retry up to max_retries times if invalid.
    Returns (output, is_valid, attempts, diff_note)."""
    text = "" if pd.isna(text) else str(text)
    last_output = ""
    last_diff = ""
    for attempt in range(1, max_retries + 2):  # initial try + retries
        last_output = fmt(text, client, model, totals)
        ok, norm_in, norm_out = validate(text, last_output)
        if ok:
            return last_output, True, attempt, ""
        last_diff = first_diff(norm_in, norm_out)
    return last_output, False, max_retries + 1, last_diff

# ----- main -----------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+")
_BRACE_RE = re.compile(r"[{}]")

def strip_aio_citation_markup(s: str) -> str:
    """Remove the '{Name URL}' / '{N1, {N2}, ..., URL}' citation markup that
    occasionally leaks in from the AIO capture pipeline. Strip URLs first
    (since '}' often abuts a URL with no space), then drop the braces, then
    tidy whitespace and stray spaces before punctuation."""
    if not isinstance(s, str) or not s:
        return s
    s = _URL_RE.sub("", s)
    s = _BRACE_RE.sub("", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)   # "Tequila ," -> "Tequila,"
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

_MD_EMPH_RE = [
    re.compile(r'\*\*\s*(.+?)\s*\*\*', re.S),     # **bold**
    re.compile(r'__\s*(.+?)\s*__',     re.S),     # __bold__
    re.compile(r'(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])', re.S),  # *italic*
    re.compile(r'(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])',   re.S),  # _italic_
]

def strip_md_emphasis(s: str) -> str:
    """Drop Markdown emphasis markers (**, __, *, _) but keep the wrapped
    content. AIO captures occasionally leak these, and the formatter's
    verbatim rule otherwise preserves them as literal asterisks/underscores
    in the rendered HTML."""
    if not isinstance(s, str) or not s:
        return s
    for pat in _MD_EMPH_RE:
        s = pat.sub(r'\1', s)
    return s

# Add a single space between clauses
_NO_SPACE_BOUNDARY_RE = re.compile(r'([)\]:])([A-Z])')

def insert_missing_clause_boundaries(s: str) -> str:
    """Insert a newline where a closing paren or bracket is immediately
    followed by a capitalized word with no whitespace between them.

    Example:
      'Figures)These figures adjust...' →
      'Figures)\nThese figures adjust...'

    The newline gives the formatter a clear paragraph boundary to wrap
    on, rather than fusing both clauses into one overlong <h3>.

    The validator collapses whitespace to a single space when comparing,
    so this preprocessing step does not produce false validation
    failures."""
    if not isinstance(s, str) or not s:
        return s
    return _NO_SPACE_BOUNDARY_RE.sub(r'\1\n\2', s)

_SUMMARY_LEADINS = (
    "In Summary", "In Short", "Verdict", "Conclusion",
    "Bottom Line", "The Bottom Line", "Key Takeaway", "Key Takeaways",
    "Overall",
)
# Boundary cases we want to split:
#   "In SummaryCalling..."  → fused to a capital letter
_FUSED_LEADIN_RE = re.compile(
    r'\b(' + '|'.join(map(re.escape, _SUMMARY_LEADINS)) + r')(?=[A-Z])'
)

def split_fused_summary_leadins(s: str) -> str:
    """Insert a newline after a known summary lead-in when it's
    immediately followed by a capital letter with no separator.
    Example: 'In SummaryCalling all of D.C.' →
             'In Summary\nCalling all of D.C.'
    The validator collapses whitespace, so this preprocessing
    step does not produce false validation failures."""
    if not isinstance(s, str) or not s:
        return s
    return _FUSED_LEADIN_RE.sub(r'\1\n', s)

def _filter_to_retrieval_id(df, retrieval_id, source_path):
    """Filter df to rows whose RETRIEVAL_ID_COL equals retrieval_id.
    Compares as strings to be robust to int/str CSV parsing differences."""
    if RETRIEVAL_ID_COL not in df.columns:
        raise SystemExit(
            f"--retrieval-id was set, but {source_path} has no "
            f"'{RETRIEVAL_ID_COL}' column. Found: {list(df.columns)}"
        )
    mask = df[RETRIEVAL_ID_COL].astype(str) == str(retrieval_id)
    matched = df[mask]
    if matched.empty:
        raise SystemExit(
            f"No rows in {source_path} have {RETRIEVAL_ID_COL}={retrieval_id!r}"
        )
    return matched

def run_retry(args):
    """Read a previous output CSV, reprocess only rows where valid=False,
    and write the updated rows back in place. If --retrieval-id is set,
    restrict to that retrieval id (still skipping rows that are valid)."""
    path = args.retry_from
    df = pd.read_csv(path)
    if "valid" not in df.columns or "aio_text" not in df.columns:
        raise SystemExit(
            f"{path} must have both 'valid' and 'aio_text' columns "
            f"(this file's columns: {list(df.columns)})"
        )

    mask = ~df["valid"].astype(bool).fillna(False)
    if args.retrieval_id is not None:
        if RETRIEVAL_ID_COL not in df.columns:
            raise SystemExit(
                f"--retrieval-id was set, but {path} has no "
                f"'{RETRIEVAL_ID_COL}' column. Found: {list(df.columns)}"
            )
        id_mask = df[RETRIEVAL_ID_COL].astype(str) == str(args.retrieval_id)
        if not id_mask.any():
            raise SystemExit(
                f"No rows in {path} have {RETRIEVAL_ID_COL}={args.retrieval_id!r}"
            )
        mask = mask & id_mask

    indices = df.index[mask].tolist()
    if not indices:
        if args.retrieval_id is not None:
            print(
                f"No invalid rows for {RETRIEVAL_ID_COL}={args.retrieval_id!r} "
                f"in {path}. Nothing to retry."
            )
        else:
            print(f"No invalid rows in {path}. Nothing to retry.")
        return

    print(f"Retrying {len(indices)} invalid rows from {path}")
    client = OpenAI()
    totals = {"in": 0, "out": 0}
    n_recovered = 0

    for j, idx in enumerate(indices, start=1):
        text = df.at[idx, "aio_text"]
        output, ok, attempts, diff_note = fmt_with_validation(
            text, client=client, model=args.model,
            totals=totals, max_retries=args.max_retries,
        )
        df.at[idx, "formatted_text"] = output
        df.at[idx, "valid"] = ok
        df.at[idx, "attempts"] = attempts
        df.at[idx, "validation_diff"] = diff_note
        if ok:
            n_recovered += 1

        if args.progress_every > 0 and (j % args.progress_every == 0 or j == len(indices)):
            cost = est_cost(totals)
            sys.stderr.write(
                f"\rRetry {j}/{len(indices)} | recovered={n_recovered} "
                f"| in={totals['in']} out={totals['out']} | cost=${cost:.6f}   "
            )
            sys.stderr.flush()

    sys.stderr.write("\n")
    df.to_csv(path, index=False)

    final_cost = est_cost(totals)
    print(f"Retry complete. Updated in place: {path}")
    print(f"Recovered: {n_recovered}/{len(indices)} previously-invalid rows")
    print(f"Total estimated cost: ${final_cost:.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--progress-every", type=int, default=1)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="If >0, re-call the model up to this many times when the formatted "
             "output doesn't match the input after stripping tags.",
    )
    ap.add_argument(
        "--retry-from",
        default=None,
        help="Path to a previous output CSV. If set, only rows where "
             "valid=False are reprocessed; the file is updated in place.",
    )
    ap.add_argument(
        "--retrieval-id",
        default=None,
        help=f"If set, process only the row whose '{RETRIEVAL_ID_COL}' column "
             f"matches this value. Works with the default flow (single-row "
             f"output written to OUTPUT_CSV) and with --retry-from (only that "
             f"id's row is retried, in place).",
    )

    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    if args.retry_from:
        return run_retry(args)

    df = pd.read_csv(INPUT_CSV)

    # Strip citation markup (which doesn't render anyway)
    df["aio_text"] = df["aio_text"].apply(strip_aio_citation_markup)
    df["aio_text"] = df["aio_text"].apply(strip_md_emphasis)
    df["aio_text"] = df["aio_text"].apply(insert_missing_clause_boundaries)
    df["aio_text"] = df["aio_text"].apply(split_fused_summary_leadins)


    if args.retrieval_id is not None:
        df = _filter_to_retrieval_id(df, args.retrieval_id, INPUT_CSV)
        print(f"Filtered to {len(df)} row(s) matching "
              f"{RETRIEVAL_ID_COL}={args.retrieval_id!r}")

    if args.sample:
        df = df.head(args.sample)

    if "aio_text" not in df.columns:
        raise SystemExit(f"Missing column aio_text. Found: {list(df.columns)}")

    client = OpenAI()
    totals = {"in": 0, "out": 0}

    n = len(df)
    out_texts = []
    out_valid = []
    out_attempts = []
    out_diffs = []

    n_invalid = 0

    for i, text in enumerate(df["aio_text"].tolist(), start=1):
        output, ok, attempts, diff_note = fmt_with_validation(
            text, client=client, model=args.model,
            totals=totals, max_retries=args.max_retries,
        )
        out_texts.append(output)
        out_valid.append(ok)
        out_attempts.append(attempts)
        out_diffs.append(diff_note)
        if not ok:
            n_invalid += 1

        if args.progress_every > 0 and (i % args.progress_every == 0 or i == n):
            cost = est_cost(totals)
            sys.stderr.write(
                f"\rRow {i}/{n} | in={totals['in']} out={totals['out']} "
                f"| invalid={n_invalid} | cumulative_est=${cost:.6f}   "
            )
            sys.stderr.flush()

    sys.stderr.write("\n")

    df["formatted_text"] = out_texts
    df["valid"] = out_valid
    df["attempts"] = out_attempts
    df["validation_diff"] = out_diffs
    df.to_csv(OUTPUT_CSV, index=False)

    final_cost = est_cost(totals)
    print(f"Total estimated cost: ${final_cost:.6f}")
    print(f"Done. Wrote: {OUTPUT_CSV}")
    print(f"Input tokens:  {totals['in']}")
    print(f"Output tokens: {totals['out']}")
    print(f"Validation: {n - n_invalid}/{n} rows passed "
          f"({n_invalid} failed after {args.max_retries} retries)")
    if n_invalid:
        print("Failed row indices:",
              [i for i, v in enumerate(out_valid) if not v])

if __name__ == "__main__":
    main()