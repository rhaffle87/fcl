---
name: scholarly-writing-refiner
description: Polishes academic English paragraph by paragraph, reviewing grammar, word choice, voice, coherence, and sentence structure. Output revision suggestions alongside polished text. Triggered by phrases like 'polish this paragraph,' 'check the grammar,' 'rewrite in academic English,' or keywords like manuscript editing, SCI polishing, and journal submission editing.
---

# Scholarly Writing Refiner

## Overview

The `scholarly-writing-refiner` skill systematically refines scientific manuscripts paragraph by paragraph. Grounded in Dr. Kristin Sainani's *Writing in the Sciences* principles, Prof. Peng Sida's AI research writing guidelines, and top-tier computer science / machine learning editorial standards, it eliminates word clutter, restores active voice, tightens logical cohesion, and enforces calibrated academic hedge language.

---

## Trigger Phrases

Invoke this skill when requested to:
- "polish this paragraph" / "rewrite in academic English"
- "check grammar and flow of my paper"
- "manuscript editing" / "SCI polishing" / "journal submission editing"
- "tighten the writing in section X"
- "improve academic tone without fluff"

---

## 1. Five Core Audit Passes

When analyzing any paragraph or section, run these 5 sequential audit passes:

### Pass 1: Clutter & Nominalization Pruning
- **Eliminate zombie nouns (nominalizations)**: Verbs turned into sluggish nouns weaken sentences.
  - *Weak*: "We conducted an investigation into the occurrence of catastrophic forgetting."
  - *Strong*: "We investigated catastrophic forgetting."
  - *Weak*: "The implementation of the EWC algorithm resulted in an improvement of accuracy."
  - *Strong*: "EWC improved accuracy."
- **Cut deadwood phrases**:
  - Replace *"due to the fact that"* $\rightarrow$ *"because"*
  - Replace *"in spite of the fact that"* $\rightarrow$ *"although"*
  - Replace *"a majority of"* $\rightarrow$ *"most"*
  - Replace *"in order to"* $\rightarrow$ *"to"*
  - Replace *"serves to illustrate"* $\rightarrow$ *"illustrates"*

### Pass 2: Active Voice & Clear Agency
- Favor active voice for investigator actions; reserve passive voice only when the recipient of the action is the explicit topic:
  - *Weak passive*: "Ten federated training rounds were executed by the aggregator across two edge clients."
  - *Strong active*: "The aggregator executed ten federated rounds across two edge clients."
  - *Self-reference standard*: Use *"We propose"*, *"We measure"*, *"We observe"*. Avoid awkward third-person constructs like *"The authors argue"* or *"It is posited here"*.

### Pass 3: Sentence Architecture & Rhythm
- **Sentence length variance**: Alternate between short impact sentences (10–15 words) for key takeaways and compound sentences (20–28 words) for technical mechanisms. Disallow run-ons exceeding 35 words.
- **Topic-Stress Position**: Put familiar context (the topic) at the beginning of the sentence and new, high-impact information (the stress) at the end.
- **Parallelism**: Ensure items in comparisons, bullet points, and conjunctions share identical grammatical forms.

### Pass 4: Quantitative Precision & Word Choice
- Remove subjective intensifiers: *very*, *extremely*, *drastically*, *remarkably*, *huge*, *tiny*.
- Replace vague assertions with exact empirical metrics:
  - *Weak*: "The model achieved remarkably better latency with small accuracy drops."
  - *Strong*: "Dynamic INT8 quantization reduced edge inference latency by 57.1% (0.42 ms to 0.18 ms) while maintaining 99.20% classification accuracy."

### Pass 5: Academic Hedge Ladder Calibration
Calibrate claim strength strictly to the level of empirical evidence:

```text
[Highest Certainty]
  ▲  "proves / establishes"      (Reserve strictly for mathematical theorems and formal proofs)
  │  "demonstrates / confirms"   (Supported by reproducible multi-trial empirical benchmarks)
  │  "indicates / shows"         (Clear correlation or direct observational evidence)
  │  "suggests / implies"        (Reasonable deduction with acknowledged confounding factors)
  ▼  "may / hypothesized"        (Speculative future directions or unverified edge cases)
[Lowest Certainty]
```
*Never over-claim: If $p=0.04$ or sample variance is non-zero, use "demonstrates" or "suggests", never "proves".*

---

## 2. Standard Output Format

For every refined paragraph or section, provide both the diagnostic critique and the drop-in ready replacement text:

```markdown
### Paragraph [X] Polish

**Original:**
> [User's original draft text]

**Audit Findings:**
- *Nominalization*: "conducted an evaluation of" -> "evaluated"
- *Voice*: Converted passive "was demonstrated by our experiments" to active "our benchmarks demonstrate".
- *Precision*: Replaced "very large speedup" with exact measured ratio (2.33x).
- *Hedge adjustment*: Downgraded "proves" to "demonstrates".

**Refined Text:**
> [Polished, high-density academic English text]

**Key Trade-offs:**
- Trimmed word count by X% (from 85 words down to 54 words) while increasing quantitative information density.
```
