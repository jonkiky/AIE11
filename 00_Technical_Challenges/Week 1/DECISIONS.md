# Decisions

One row one decision. This week is the first one. By Week 5 this file will explain what decisions you made during this bootcamp.

## Retrieval, week 1

**What I changed**

> `load_corpus` now returns the UTF-8 text as a list of lines, and `split_corpus` accepts that list, keeps each nonblank line separate, and splits long lines after the nearest suitable period without overlap.

**Why**

> Passing lines between the loader and splitter preserves the book's line boundaries, so separate lines are not combined. Period-based splitting is intended to avoid breaking sentences mid-thought. Before/after retrieval scores have not been measured yet.

**What would change my mind**

> If evaluation shows worse retrieval than fixed-size overlapping chunks, or period-based splitting creates poor boundaries (for example around abbreviations), I would compare a sentence-aware splitter or adjust the chunking strategy.

## Hard token limits and optional wiki retrieval

**What I changed**

> Source chunks are limited to 1,000 Qwen tokens, replacing the splitter's
> 1,000-character target. Numbered wiki input windows are limited to 2,000
> Qwen tokens with up to 500 tokens of overlap, giving the topic extractor
> more surrounding context. Prefer complete
> sentences, but split an oversized sentence at whitespace or a Unicode
> character boundary rather than exceed the limit. The loader still returns
> original nonblank lines; wiki construction rejects files with blank lines.
>
> Added an opt-in wiki retriever with cached topic embeddings, original
> source-line citations, raw-question matching, up to three topics per query,
> and an explicit no-match response. Chunk retrieval remains the default.

**Why**

> Token limits bound model inputs more accurately than character counts.
> Wiki topics are intended to retrieve connected explanations across paragraphs.
> Initial local testing built 46 topics from the CEO corpus with a maximum
> numbered input window of 992 tokens.
>
> A retrieval-only threshold sweep from 0.40 to 0.70, in 0.05 increments, on
> the 10 answerable and 3 unrelated questions selected 0.55: it was the largest
> tested threshold retaining a match for all 10 answerable questions, and
> rejected all 3 unrelated questions. At 0.50 one unrelated question still
> matched; at 0.60 two answerable questions had no match. This measures match
> availability, not answer correctness.
>
> The first full wiki evaluation, using near-tie rescue, scored 33.3/100 on the original 10 questions,
> below the historical chunk baseline of 54.2. The 13-question mean was 37.2.
> All three unrelated queries had no matches and skipped the answer model,
> but the LLM judge inconsistently scored their equivalent refusals. The
> wiki is experimental and must not become the default based on this run.
>
> The top-three rerun scored 50.8/100 on the original 10 questions and
> 50.6/100 on all 13; see [eval/results2.json](eval/results2.json).
> Three answerable questions still received refusals. All three unrelated
> queries had no matches, but the judge scored the same refusal 0, 50, and
> 100, so these scores should be treated cautiously.

**What would change my mind**

> Re-tune the threshold for each corpus and a larger held-out evaluation set.
> Keep chunk retrieval as the default unless the wiki outperforms the
> neighboring-line baseline in the design's evaluation plan. The same small
> model generates and judges answers, so its scores may be biased.

