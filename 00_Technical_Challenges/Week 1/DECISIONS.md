# Decisions

One row one decision. This week is the first one. By Week 5 this file will explain what decisions you made during this bootcamp.

## Retrieval, week 1

**What I changed**

> `load_corpus` now returns the UTF-8 text as a list of lines, and `split_corpus` accepts that list, keeps each nonblank line separate, and splits long lines after the nearest suitable period without overlap.

**Why**

> Passing lines between the loader and splitter preserves the book's line boundaries, so separate lines are not combined. Period-based splitting is intended to avoid breaking sentences mid-thought. Before/after retrieval scores have not been measured yet.

**What would change my mind**

> If evaluation shows worse retrieval than fixed-size overlapping chunks, or period-based splitting creates poor boundaries (for example around abbreviations), I would compare a sentence-aware splitter or adjust the chunking strategy.

