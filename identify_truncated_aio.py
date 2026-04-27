"""
Find truncated `aio_text` rows in retrievals.csv.
Regex pre-filter flags suspicious endings.

Only rows the LLM labels 'truncated' are written to the output CSV.

Usage:
    python identify_truncated_aio.py --input full_samples/retrievals.csv --output full_samples/truncated.csv
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Pass 1 — regex heuristics
# ---------------------------------------------------------------------------

TERMINAL_CHARS = set('.!?"\')]”’')

ABBREVIATIONS = {
    'mr', 'mrs', 'ms', 'dr', 'jr', 'sr', 'st', 'mt', 'ft', 'co', 'inc',
    'ltd', 'etc', 'vs', 'e.g', 'i.e', 'u.s', 'u.k', 'no', 'fig', 'pg',
    'pp', 'vol', 'ave', 'blvd', 'rd', 'hwy',
}

DANGLING_MARKDOWN_RE = re.compile(r'(\*\*|__|\*|_)([A-Za-z][\w\-/]*)\s*$')
TRAILING_PUNCT_RE = re.compile(r'[,;\-\u2013\u2014]\s*$')
HEADING_FRAGMENT_RE = re.compile(r'(^|\n)\s*(#{1,6}\s+|\*\*)[^.\n]{1,80}$')
ENDS_WITH_BARE_COLON_RE = re.compile(r':\s*$')
SHORT_TAIL_FRAGMENT_RE = re.compile(r'(?:^|[\s\n])([A-Za-z]{1,4})\s*$')


def regex_truncation_signals(text: str) -> dict:
    """Return per-signal booleans. Any True signal => candidate for LLM review."""
    if not isinstance(text, str) or not text.strip():
        return {'empty': True}

    stripped = text.rstrip()
    last_char = stripped[-1]
    last_line = stripped.rsplit('\n', 1)[-1].strip()

    s = {}
    s['no_terminal_punct'] = last_char not in TERMINAL_CHARS
    s['dangling_markdown'] = bool(DANGLING_MARKDOWN_RE.search(stripped))
    s['trailing_continuator'] = bool(TRAILING_PUNCT_RE.search(stripped))
    s['heading_fragment'] = bool(HEADING_FRAGMENT_RE.search(stripped))
    s['bare_trailing_colon'] = (
        bool(ENDS_WITH_BARE_COLON_RE.search(stripped)) and len(last_line) < 60
    )

    if last_char == '.':
        m = re.search(r'(\S+)\.\s*$', stripped)
        tail_word = (m.group(1) if m else '').lower().strip('.,;:"\')')
        s['ends_in_abbreviation'] = tail_word in ABBREVIATIONS
    else:
        s['ends_in_abbreviation'] = False

    if s['no_terminal_punct']:
        m = SHORT_TAIL_FRAGMENT_RE.search(stripped)
        s['short_tail_fragment'] = bool(m and len(m.group(1)) <= 3)
    else:
        s['short_tail_fragment'] = False

    return s


def regex_is_candidate(text: str) -> bool:
    sig = regex_truncation_signals(text)
    if sig.get('empty'):
        return False
    return any(sig.values())


def fired_signals(text: str) -> str:
    sig = regex_truncation_signals(text)
    return ','.join(k for k, v in sig.items() if v) or ''

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', '-i', default='full_samples/retrievals.csv',
                   help='Path to input CSV (default: retrievals.csv)')
    p.add_argument('--output', '-o', default='truncated_aio_text.csv',
                   help='Path to output CSV (default: truncated_aio_text.csv)')
    p.add_argument('--text-column', default='aio_text',
                   help='Column to check (default: aio_text)')
    p.add_argument('--id-column', default='retrieval_id',
                   help='Identifier column to include in output (default: retrieval_id)')
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Input not found: {in_path}")

    df = pd.read_csv(in_path)
    print(f"Loaded {len(df):,} rows from {in_path}")

    if args.text_column not in df.columns:
        sys.exit(f"Column '{args.text_column}' not in CSV. Columns: {list(df.columns)}")

    # ---- Pass 1: regex ----
    df['regex_candidate'] = df[args.text_column].apply(regex_is_candidate)
    df['regex_signals'] = df[args.text_column].apply(fired_signals)
    n_cand = df['regex_candidate'].sum()
    print(f"Regex flagged {n_cand} / {len(df)} rows ({n_cand/len(df):.1%})")

    out = df[df['regex_candidate']].copy()
    out['tail'] = out[args.text_column].astype(str).str.slice(-120).str.replace(
        '\n', ' \\n ', regex=False)
    cols = [c for c in [args.id_column, 'query', 'tail', 'regex_signals']
            if c in out.columns or c == 'tail']
    out[cols].to_csv(args.output, index=False)
    print(f"Wrote {len(out)} regex candidates -> {args.output}")
    return

    ## Pass 2: manual review 
    # After manual review, truncated aio_text in full_samples/confirmed_truncated.csv


if __name__ == '__main__':
    main()