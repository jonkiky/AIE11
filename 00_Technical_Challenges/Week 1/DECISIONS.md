# Decisions

One row one decision. This week is the first one. By Week 5 this file will explain what decisions you made during this bootcamp.

## Retrieval, week 1

**What I changed**

> `split_corpus` now keeps each nonblank input line separate and splits long lines after the nearest suitable period, without overlap.

**Why**

> This is intended to avoid breaking sentences mid-thought and combining separate book lines into one chunk. Before/after retrieval scores have not been measured yet.

**What would change my mind**

> If evaluation shows worse retrieval than fixed-size overlapping chunks, or period-based splitting creates poor boundaries (for example around abbreviations), I would compare a sentence-aware splitter or adjust the chunking strategy.

