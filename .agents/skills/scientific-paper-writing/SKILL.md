---
name: scientific-paper-writing
description: Hierarchical skill group that orchestrates five sub-skills (Literature Survey, Paper Structure & Logic, Experiment Design, Academic Figures & Tables, and Peer Review Simulation) through 4 quality gates and an iterative review loop to autonomously produce publication-ready scientific papers.
---

# Scientific Paper Writing: Hierarchical Autonomous Skill Group

## Overview

The `scientific-paper-writing` skill group is an autonomous pipeline orchestrator designed to produce competitive, publication-ready research manuscripts (targeted at 8.5+/10 scoring). It coordinates five specialized sub-skills, enforces strict quality gates, and operates an iterative review loop with automated weakness routing.

---

## 1. Sub-Skills Division of Labor

```text
┌────────────────────────────────────────────────────────────────────────┐
│             scientific-paper-writing Orchestration Layer               │
└──────────────────┬─────────────────────────────────┬───────────────────┘
                   │                                 │
     ┌─────────────┴─────────────┐     ┌─────────────┴─────────────┐
     ▼                           ▼     ▼                           ▼
[01 Literature Survey]  [02 Paper Structure]  [03 Experiment Design] [04 Figures & Tables]
     │                           │                   │                     │
     └───────────────────────────┼───────────────────┴─────────────────────┘
                                 ▼
                     [05 Peer Review Simulation]
                                 │ (Weakness Routing Feedback Loop)
                                 └───► Re-route to Sub-Skills 01-04
```

### Sub-Skill 01: Literature Survey
- **Input**: Research topic + taxonomy keywords
- **Output**: `references.bib` + `citation_plan.jsonl`
- **4-Stage Pipeline**:
  1. *High-Recall Retrieval*: 20–30 keyword queries (`site:arxiv.org`, Google Scholar, Semantic Scholar). 200–500 candidates.
  2. *LQS (Literature Quality Score)*:
     $$\text{LQS} = 0.30 \cdot \text{Recency} + 0.25 \cdot \text{Impact} + 0.20 \cdot \text{Venue} + 0.10 \cdot \text{Institution} + 0.15 \cdot \text{Acceptance}$$
     Thresholds: $\text{LQS} \ge 7.0$ (must-cite), $5.0–7.0$ (conditional), $<5.0$ (drop).
  3. *Citation Depth Classification*:
     - **A-level** (1–3 paragraphs): Section protagonist (3–5 per chapter).
     - **B-level** (2–5 sentences): Crucial insight or comparison (5–10 per chapter).
     - **C-level** (1 sentence): Supporting empirical evidence.
     - **D-level**: Dropped from citation plan.
  4. *Venue Upgrade*: Cross-check DBLP / OpenReview to upgrade arXiv preprints to published proceedings. arXiv-only ratio $\le 60\%$.

### Sub-Skill 02: Paper Structure & Logic
- **Input**: Bibliography + experimental data
- **Output**: Full manuscript sections (`sections/*.tex` or `manuscript.md`)
- **Core Patterns**:
  - *Funnel*: Broad context $\rightarrow$ Narrow barrier $\rightarrow$ Proposed thesis (§1 Introduction).
  - *Claim-Evidence-Implication*: Assertion $\rightarrow$ Measured data $\rightarrow$ Practical implication (§3–§7).
  - *Compare-Contrast*: Approach A vs B $\rightarrow$ Explicit trade-off boundary (§8 Related Work).
  - *Concession-Rebuttal*: Acknowledge baseline strength $\rightarrow$ Demonstrate superiority in target regime.
- **Formal Claims & Hedge Ladder**: Calibrate assertions (Conjecture/Observation rather than unproven Theorems; *demonstrates* > *suggests* > *may*).

### Sub-Skill 03: Experiment Design
- **Input**: Research conjecture or performance gap
- **Output**: `results.json` + `experiment_summary.md`
- **Loop**: Design (pre-registered hypothesis) $\rightarrow$ Execute (reproducible multi-seed trials) $\rightarrow$ Iterate (tune boundary conditions) $\rightarrow$ Report (raw metrics with mean $\pm$ std).

### Sub-Skill 04: Academic Figures & Tables
- **Input**: `results.json` + draft placeholders
- **Output**: `figures/*.pdf` + `tables/*.tex`
- **Standards**:
  - Tables: Strict `booktabs` format (three horizontal lines: `\toprule`, `\midrule`, `\bottomrule`). **Zero vertical lines**. Bold optimal values. Captions state key takeaway findings.
  - Figures: Vector PDF preferred (TikZ or matplotlib $\ge 300$ DPI). Academic color palette: Blue (`#2196F3`), Red (`#F44336`), Green (`#4CAF50`), Orange (`#FF9800`). All axes labeled, light grid ($\alpha=0.3$), self-contained captions.

### Sub-Skill 05: Peer Review Simulation
- **Input**: Compiled manuscript PDF
- **Output**: Multi-persona scorecard (R1 Experimentalist, R2 Theorist, R3 Perfectionist, R4 Synthesizer, R5 Newcomer), prioritized major/minor weakness lists, and automatic routing to sub-skills 01–04.

---

## 2. Five Quality Gates

Every manuscript revision must clear each progressive gate:

| Gate | Focus | Verification Requirements |
|:---:|:---|:---|
| **Gate 1** | **Literature** | $\ge 80$ citations (draft) / $\ge \text{pages} \times 3$ (final). Within 1 year $\ge 40\%$. Accepted venue $\ge 30\%$. arXiv-only $\le 60\%$. Zero hallucinated citations. |
| **Gate 2** | **Experiment** | Pre-registered hypotheses linked to paper claims. $\ge 3$ seeded trials with mean $\pm$ std reported. Statistical significance ($p < 0.05$ or non-overlapping 95% CI). |
| **Gate 3** | **Structure** | Compiles with 0 errors & 0 undefined references. Each modular `.tex` file $\le 300$ lines. Abstract-conclusion alignment verified. Critical assessment subsections included. |
| **Gate 4** | **Figures & Tables** | Full survey: $\ge 10$ tables, $\ge 6$ figures (Short paper: $\ge 5$ tables, $\ge 3$ figures). `booktabs` compliant (no vertical lines). Every figure/table referenced in text. |
| **Gate 5** | **Final Review (Blocking)**| Gates 1–4 passed. Peer review score reaches target ($\ge 8.5/10.0$). No regression on previously resolved weaknesses. |

---

## 3. Workflow & Iteration Phases

```text
Phase 0: Topic Selection (Scope? Angle? Audience?)
   ▼
Phase 1: Draft Creation (Iter 1–6, Target Score: 6.0/10)
   - Iter 1: Skeleton + §1–2
   - Iter 2: Literature Recall & LQS scoring
   - Iter 3: §3–6 Core + Initial Figures
   - Iter 4: §7–8 Experiments & Discussion
   - Iter 5: Verification & First Simulated Review
   - Iter 6: Route initial fixes
   ▼
Phase 2: Deep Improvement (Iter 7–9, Target Score: 7.5–8.0/10)
   - Iter 7: Experiment execution & ablation tightening
   - Iter 8: Academic figure generation & integration
   - Iter 9: Full compilation & Round 2 Review
   ▼
Phase 3: Sprint & Camera-Ready (Iter 10+, Target Score: 8.5+/10)
   - Loop: Review -> Weakness Routing -> Fix -> Compile
   - Termination Condition: Score >= 8.5 OR delta <= 0.3 across 2 consecutive rounds.
```
