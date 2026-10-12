---
name: sci-paper
description: Structured guidance for composing, formatting, and presenting scientific research papers for top-tier venues (CVPR, ICCV, NeurIPS, ICML, ACL, ICLR, IEEE S&P, USENIX Security, ACM CCS). Covers the full workflow: section-by-section drafting, figure design, LaTeX typesetting, narrative logic, and pre-submission polishing. Use for writing, revising, or typesetting manuscripts into camera-ready or preprint PDFs.
---

# Scientific Research Paper Architect (SCI / Top-Tier Conferences)

## Overview

The `sci-paper` skill provides end-to-end architectural, narrative, and LaTeX typesetting guidance for drafting elite scientific research papers targeted at flagship venues (IEEE S&P, USENIX Security, ACM CCS, NeurIPS, ICML, ICLR, CVPR, and IEEE Transactions). It standardizes chapter progression, narrative logic chains, formal claim formulation, taxonomy matrices, and LaTeX typesetting conventions.

---

## Trigger Phrases

Invoke this skill when requested to:
- "draft a research paper" / "structure my scientific manuscript"
- "write paper section by section" / "organize conference paper"
- "typeset camera-ready paper" / "format LaTeX manuscript"
- "design research paper taxonomy and claims"

---

## 1. Section-by-Section Manuscript Architecture

Every competitive research paper follows this rigid 9-section logical progression:

```text
§1 Introduction ────► Hook ──► Core Barrier ──► Insight ──► Contributions ──► Roadmap
§2 System Model  ──► Threat Model ──► Formal Definitions ──► Multi-Axis Taxonomy
§3-§6 Methods    ──► Technical Core: One method family per chapter with Critical Assessment
§7 Evaluation    ──► Claim-Evidence Mapping ──► Baselines ──► Ablations ──► Stress Tests
§8 Discussion    ──► Open Problems ──► Practical Limitations (Barrier + Attack Vector)
§9 Conclusion    ──► Numbered Key Takeaways (Synthesis, NOT an Abstract echo)
```

### §1 Introduction: The 5-Paragraph Funnel Formula
1. **Paragraph 1 (The Hook)**: Establish broad real-world significance and technical momentum.
2. **Paragraph 2 (The Critical Gap)**: Why do existing approaches fail? Identify the specific fundamental limitation or barrier (not just "few works have explored this").
3. **Paragraph 3 (The Core Insight)**: Introduce our central technical hypothesis or conceptual mechanism in plain, rigorous terms.
4. **Paragraph 4 (Bulleted Contributions)**: Explicitly state 3–4 bulleted contributions:
   - *Conceptual / Theoretical*: Novel formulation, taxonomy, or analytical bounds.
   - *Architectural / Algorithmic*: Novel system design, loss function, or coordination protocol.
   - *Empirical / Deployment*: Real-world testbed implementation, benchmark validation, and public release artifacts.
5. **Paragraph 5 (Roadmap)**: Terse navigational overview ("The remainder of this paper is organized as follows...").

### §2 System & Threat Model
- State operational environment, actor roles, communication boundaries, and trust assumptions.
- Provide formal mathematical notation table or glossary.
- Categorize threat actors using clear capability bounds (e.g. passive eavesdropper, Byzantine client with $\le 20\%$ participation, honest-but-curious aggregator).

### §3–§6 Core Technical Methodology
- One method family per chapter.
- Every technical description must present:
  1. Formal mathematical formulation (equations, objectives, parameter updates).
  2. Concrete execution algorithm (pseudo-code or operational sequence diagram).
  3. **Critical Assessment subsection**: Acknowledge computational overheads, communication costs, or edge cases before the reviewers point them out.

### §7 Empirical Evaluation: Claim-Evidence Mapping
- **Pre-registered Hypotheses**: Group experiment subsections directly by the claims they substantiate (e.g., *§7.2 Evaluation of Claim C1: Forgetting Resistance*).
- **Mandatory Triad**:
  - *Baseline comparison*: Compare against established state-of-the-art baselines under identical testbeds.
  - *Ablation study*: Isolate individual algorithmic components to demonstrate each contributes meaningfully.
  - *Sensitivity / Stress testing*: Hyperparameter sweeps, noise multipliers, non-IID skew, or attack rates.

### §8 Related Work Differentiation
- Never rely solely on temporal superiority (*"our work is more recent"*).
- Include a **Summary & Differentiation Table** comparing our work against 5–8 major prior frameworks across operational axes (e.g. encrypted traffic support, zero-DPI, Byzantine resilience, continual learning backward transfer).

---

## 2. Core Paragraph Logic Patterns

Inside every section, construct paragraphs adhering to one of four formal rhetorical structures:

| Pattern | Internal Progression | Primary Usage |
|:---|:---|:---|
| **Claim-Evidence-Implication** | Assert thesis $\rightarrow$ Present numerical data / proof $\rightarrow$ Articulate system implication (*"So what?"*). | Main technical & empirical sections. |
| **Compare-Contrast** | Introduce Approach A $\rightarrow$ Introduce Approach B $\rightarrow$ Highlight operational divergence $\rightarrow$ Define boundary trade-off. | Baseline comparisons & related work. |
| **Concession-Rebuttal** | Acknowledge competitor or baseline advantage $\rightarrow$ State fundamental trade-off $\rightarrow$ Prove our design superior in target regime. | Critical assessment & discussion. |
| **Funnel** | Broad domain context $\rightarrow$ Narrowing problem context $\rightarrow$ Specific unanswered challenge $\rightarrow$ Our proposed remedy. | Introduction & section motivation openings. |

---

## 3. MECE Taxonomy Matrix Design

When proposing a classification or survey taxonomy:
- **Mutually Exclusive, Collectively Exhaustive (MECE)**: Axes must be independent (e.g., *Coordination Topology* [Centralized vs Decentralized] $\times$ *Memory Strategy* [Regularization vs Episodic Buffer vs Parameter Isolation]).
- **Deliberate Empty Cells**: An effective taxonomy intentionally highlights unfilled parameter combinations, naturally establishing the research gap for our paper.

---

## 4. LaTeX Typesetting & Presentation Rules

When producing `.tex` manuscripts:
1. **Table Hygiene**:
   - Strictly employ `booktabs` package (`\toprule`, `\midrule`, `\bottomrule`).
   - **Never use vertical lines** (`|`).
   - Bold optimal metrics in each column (`\textbf{0.9920}`).
   - Add caption explanations that state the primary conclusion, not just a label.
2. **Mathematical Precision**:
   - Enforce consistent notation (vectors lowercase bold $\mathbf{x}$, matrices uppercase bold $\mathbf{W}$, sets calligraphic $\mathcal{D}$).
   - Define all terms immediately preceding or following equations.
3. **Length Budgeting**:
   - Keep individual `.tex` section files modular ($\le 300$ lines per file).
   - Use `\input{sections/sec1_intro.tex}` inside master `manuscript.tex`.
