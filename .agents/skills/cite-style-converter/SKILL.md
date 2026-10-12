---
name: cite-style-converter
description: Convert academic citations between APA, MLA, IEEE, and Harvard styles with batch processing and format validation. Trigger when users ask to convert, check, or fix their references, mention specific citation styles, or talk about their bibliography.
---

# Citation Style Converter & Bibliography Auditor

## Overview

The `cite-style-converter` skill handles bidirectional format conversions across academic citation conventions (IEEE, APA 7th, MLA 9th, Harvard, Chicago), cleans BibTeX files, audits citation distributions, and upgrades preprints (`@misc` arXiv entries) to peer-reviewed venue entries (`@inproceedings`, `@article`) using standard academic bibliographic databases.

---

## Trigger Phrases

Invoke this skill when requested to:
- "convert citation style to IEEE / APA / Harvard / MLA"
- "check my references" / "fix bibliography" / "audit citations"
- "convert bibtex to inline references"
- "upgrade arxiv citations to peer-reviewed conference entries"
- "validate bibliography keys and formatting"

---

## 1. Supported Citation Formats

### 1. IEEE (Numbered, Standard for Computer Science, Electrical Engineering & Security)
- **In-text**: `[1]`, `[1, 2]`, `[1]-[3]`
- **Bibliography Reference**:
  `[1] A. Author and B. Author, "Title of paper," in *Proc. IEEE Symp. Security and Privacy (S&P)*, San Francisco, CA, USA, 2024, pp. 112–128.`

### 2. APA 7th Edition (Author-Date, Standard for Social Sciences & Machine Learning)
- **In-text**: `(Author & Author, 2024)` or `Author and Author (2024)`
- **Bibliography Reference**:
  `Author, A., & Author, B. (2024). Title of paper. *Proceedings of the 41st International Conference on Machine Learning*, 112–128. https://doi.org/...`

### 3. Harvard Style (Author-Date)
- **In-text**: `(Author and Author 2024)`
- **Bibliography Reference**:
  `Author, A. and Author, B., 2024. Title of paper. In: *Proceedings of the 41st International Conference on Machine Learning*. Vienna, Austria, pp. 112–128.`

### 4. MLA 9th Edition
- **In-text**: `(Author and Author 112)`
- **Bibliography Reference**:
  `Author, Author, and Author Author. "Title of Paper." *Proceedings of the 41st International Conference on Machine Learning*, 2024, pp. 112–28.`

---

## 2. Bibliographic Quality Gates & Rules

A publication-grade bibliography must satisfy the following automated verification checks:

1. **Zero Hallucination Guarantee**: Every cited title, author list, year, and DOI must correspond to a real indexable paper.
2. **Venue Upgrade Pipeline (Preprint -> Peer-Reviewed)**:
   - Search DBLP and OpenReview for all `@misc{...}` or arXiv entries.
   - If an arXiv preprint has been accepted at a peer-reviewed venue (e.g. NeurIPS, ICML, IEEE S&P, USENIX, ACM CCS, CVPR), upgrade the BibTeX record to `@inproceedings` or `@article`.
   - **Target**: arXiv-only citations must not exceed **60%** of total references in the final manuscript.
3. **Temporal Freshness Balance**:
   - **Within 1 Year**: $\ge 40\%$ of citations should be recent (to prove mastery of current state-of-the-art).
   - **Peer-Reviewed Accepted Ratio**: $\ge 30\%$ of total references must be formal venue publications.
4. **BibTeX Structural Hygiene**:
   - All entry keys must follow consistent kebab-case or `author_year_keyword` syntax (e.g. `mcmahan2017communication`, `kirkpatrick2017overcoming`).
   - Title capitalization protected by braces: `{Federated Learning}`, `{Elastic Weight Consolidation}`, `{TCP/IP}`.
   - No duplicate citation keys or unclosed braces.
   - Strip unnecessary automated export junk (`file = {...}`, `abstract = {...}`, `keywords = {...}`).

---

## 3. Workflow Steps

### Step 1: Ingestion & Syntax Verification
Scan the `.bib` or markdown reference section:
- Detect syntax errors, missing fields (`author`, `title`, `booktitle`/`journal`, `year`).
- Count total unique entries and identify duplicate DOIs or titles.

### Step 2: Venue Upgrade Audit
Flag all arXiv-only records:
```bibtex
% BEFORE (Unverified arXiv preprint)
@misc{smith2023fl,
  title={Continual Federated Learning at the Edge},
  author={Smith, John and Doe, Jane},
  archivePrefix={arXiv},
  eprint={2305.12345},
  year={2023}
}

% AFTER (Upgraded to published conference entry)
@inproceedings{smith2024continual,
  author    = {John Smith and Jane Doe},
  title     = {Continual Federated Learning at the Edge},
  booktitle = {Proceedings of the IEEE Conference on Computer Communications (INFOCOM)},
  year      = {2024},
  pages     = {1042--1051},
  doi       = {10.1109/INFOCOM.2024.1234567}
}
```

### Step 3: Batch Style Conversion
Output clean, validated references according to the requested target style (IEEE, APA, Harvard, MLA, or canonical BibTeX).
