---
name: academic-paper-reviewer
description: Simulates academic peer review, evaluating papers across Originality, Methodology, Results, and Writing to provide Major/Minor Revision recommendations with actionable feedback. Triggers when a user asks to "review my paper," "simulate peer review," or "give my paper a peer review."
---

# Academic Paper Reviewer

## Overview

The `academic-paper-reviewer` skill simulates high-standard, skeptical academic peer review (matching top-tier venues such as IEEE S&P, USENIX Security, ACM CCS, NeurIPS, ICML, CVPR, and ICLR). It executes multi-persona scoring, applies rigorous anti-inflation calibration, identifies structural and empirical vulnerabilities, and generates prioritized, actionable revisions with an explicit weakness routing mechanism.

---

## Trigger Phrases

Invoke or activate this skill when asked to:
- "review my paper" / "simulate peer review" / "give my paper a peer review"
- "evaluate manuscript for submission"
- "check my paper against reviewer standards"
- "generate peer review report" / "run review scorecard"

---

## 1. Multi-Persona Review Panel

To eliminate single-perspective blind spots, the review simulation executes an independent 5-persona evaluation panel:

| Persona | Primary Focus | Evaluative Stance | Scoring Weight |
|:---|:---|:---|:---:|
| **R1: Experimentalist** | Statistical rigor, baselines, reproducibility, variance, sample sizes, and threat-model fidelity. | "Show me the error bars, seeds, controls, and ablation baselines. Does the empirical setup isolate the actual mechanism?" | 30% |
| **R2: Theorist** | Formal definitions, proofs, mathematical guarantees, bound derivations, and MECE taxonomies. | "Are the conjectures sound? Does the taxonomy leak edge cases? Is the mathematical notation standard and non-tautological?" | 35% |
| **R3: Perfectionist** | Figure aesthetics, booktabs compliance, narrative clarity, citation precision, and typos. | "Are vector figures self-contained? Are captions informative findings rather than generic labels? Is active voice maintained?" | 30% |
| **R4: Synthesizer** | Related work coverage, conceptual novelty, field positioning, and genuine gap identification. | "What fundamentally distinguishes this from prior art? Does the taxonomy provide actionable forward-looking research vectors?" | 25% |
| **R5: Newcomer** | Readability, intuitive accessibility, self-contained notation, and explicit terminology definitions. | "Can a graduate student outside this immediate subfield understand the core intuition without deciphering unexplained jargon?" | 35% |

---

## 2. Scoring Protocol & Anti-Inflation Calibration

Peer review simulation must strictly resist LLM sycophancy or grade inflation. Calibrate scores against top-tier conference standards:

- **9.0 – 10.0 (Oral / Best Paper Nominee)**: Flawless execution, seminal theoretical or empirical breakthrough. Top 1–2%.
- **8.0 – 8.9 (Strong Accept)**: Rigorous methodology, comprehensive baselines, crystal-clear prose, compelling novelty. Top 15–20%.
- **7.0 – 7.9 (Accept / Minor Revision)**: Solid contribution, minor gaps in ablation or literature, technically sound. Main conference acceptance cutoff.
- **6.0 – 6.9 (Borderline / Major Revision)**: Promising direction, but missing critical baselines, unclear threat model, or narrative leaks. Workshop standard.
- **4.0 – 5.9 (Weak Reject)**: Significant methodological omissions, unsubstantiated claims, or severe clarity barriers.
- **1.0 – 3.9 (Strong Reject)**: Broken theory, fabricated or fatally confounded experiments, plagiarized or superficial content.

### Anti-Inflation Calibration Rules:
1. **First-Round Ceiling**: Initial review iterations (`Iter 1–6`) are capped at a maximum composite score of **7.0**. Every early manuscript has substantial room for tightening.
2. **Maximum Step Delta**: Between iterative review cycles, a paper's score may advance by at most **+1.5 points**.
3. **Persistent Weakness Constraint**: A realistic review must maintain at least one unresolved minor limitation or open challenge; no paper is completely bulletproof.
4. **Final Composite Calculation**: The overall paper score is computed as the **median** of all active reviewer personas to prevent outlier distortion.

---

## 3. Weakness Routing Table

When the panel flags weaknesses, route them systematically to the responsible skill or pipeline stage:

| Identified Reviewer Weakness | Target Skill / Module | Actionable Remediation |
|:---|:---|:---|
| *"Citation coverage insufficient / biased"* | `cite-style-converter` / Literature | Run high-recall search, add key seminal works and within-1-year publications. |
| *"Too many arXiv-only citations"* | `cite-style-converter` | Cross-check DBLP / OpenReview to upgrade preprints to peer-reviewed venue entries. |
| *"Claims too strong / unsubstantiated"* | `scholarly-writing-refiner` | Apply hedge ladder: downgrade *proves* to *demonstrates*, or *demonstrates* to *suggests*. |
| *"Structure disjointed / narrative lacks transitions"* | `sci-paper` | Refactor using Funnel pattern in §1, Claim-Evidence-Implication in body sections. |
| *"Taxonomy is a flat list / not novel"* | `sci-paper` | Redesign taxonomy into a multi-axis MECE matrix with documented gap cells. |
| *"Missing error bars / statistical tests"* | `scientific-paper-writing` | Report mean $\pm$ std over $\ge 3$ seeded trials, report $p$-values or 95% bootstrap CIs. |
| *"Tables have vertical rules / low density"* | `scientific-paper-writing` | Reformat to `booktabs` (three horizontal lines only), bold optimal values, add caption conclusions. |
| *"Missing visual architecture / high-level flow"* | `sci-paper` / Figures | Add self-contained vector diagram (TikZ or matplotlib PDF) with explicit caption insight. |

---

## 4. Standard Review Report Output Schema

Every simulated peer review report generated by this skill must follow this structured markdown template:

```markdown
# Academic Peer Review Report: [Manuscript Title]

- **Target Venue**: [e.g. IEEE S&P / USENIX Security / NeurIPS]
- **Review Cycle**: [Round 1 / Round 2 / Pre-Submission Final]
- **Composite Score**: [Score / 10.0] ([Decision: Strong Accept / Accept / Weak Accept / Borderline / Reject])

## Reviewer Scorecard
| Persona | Score (/10) | Confidence (/5) | Key Evaluative Focus |
|:---|:---:|:---:|:---|
| R1: Experimentalist | X.X | 4/5 | Statistical significance, ablation controls |
| R2: Theorist | X.X | 5/5 | Formal bounds, MECE taxonomy |
| R3: Perfectionist | X.X | 4/5 | Presentation, figure quality, booktabs |
| R4: Synthesizer | X.X | 4/5 | Related work differentiation |
| R5: Newcomer | X.X | 5/5 | Notation clarity, self-contained explanations |
| **Composite (Median)** | **X.X** | **4.4/5** | **[Recommendation Verdict]** |

## 1. Summary of Contribution
[2-3 paragraphs synthesizing the core problem, proposed solution, and principal findings.]

## 2. Key Strengths
- **S1 (Novelty/Impact)**: ...
- **S2 (Empirical Rigor)**: ...
- **S3 (Clarity/Presentation)**: ...

## 3. Major Weaknesses (Required for Acceptance)
- **W1 [Methodology]**: ...
- **W2 [Baselines/Ablations]**: ...
- **W3 [Theoretical Grounding]**: ...

## 4. Minor Weaknesses & Typographical Corrections
- **M1**: ...
- **M2**: ...

## 5. Prioritized Action Plan & Routing
1. **Priority 1 (Blocking)**: ... -> Route to [Skill Name]
2. **Priority 2 (Strengthening)**: ... -> Route to [Skill Name]
3. **Priority 3 (Polishing)**: ... -> Route to [Skill Name]
```
