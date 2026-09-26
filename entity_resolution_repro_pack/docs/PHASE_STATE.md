# Entity Resolution Challenge — Persistent Phase State

This document records the project state reconstructed from the completed work.
It is intentionally stored separately from code so losing a compute machine
does not erase the decisions/results.

## Problem

Three independent sources:
- S1 reference source
- S2
- S3

Each S1 can map to zero, one, or multiple S2/S3 records.
`entity_id` is only a record/source identifier and is not matching evidence.

Primary fields:
- entity_id
- business_name
- business_address
- country

TSV files must be read with `sep="\t"`.

## Phase 1–6: measured data/ground-truth state

### S1
- 2,206,821 rows
- ~6.33 s read time
- ~671.78 MB pandas RAM
- no missing/empty names
- no missing/empty addresses
- no missing/empty country
- 2 countries: US 1,323,633; India 883,188
- 2,206,821 unique IDs
- all IDs use S1- prefix
- exact duplicate business_name rows: 667,592
- exact duplicate business_address rows: 76,215

### S2
- 5,034,616 rows
- ~1,580.64 MB pandas RAM
- business_name missing/empty: 2
- business_address missing/empty: 168,967 (~3.36%)
- 2 countries: US 3,016,817; India 2,017,799
- 5,034,616 unique IDs
- all IDs use S2- prefix
- multilingual/Devanagari records were observed
- Unicode must be preserved

### S3
- 5,285,603 rows
- ~1,646.70 MB pandas RAM
- business_name missing/empty: 13
- business_address missing/empty: 175,916 (~3.33%)
- 2 countries: US 3,170,056; India 2,115,547
- 5,285,603 unique IDs
- all IDs use S3- prefix
- exact duplicate business_name rows: 633,994
- exact duplicate business_address rows: 652,838

### Ground truth
- 2,206,821 S1 rows
- 2,206,821 unique S1 IDs; no duplicates
- zero matches: 123,247 (5.58%)
- exactly one: 119,157 (5.40%)
- multiple: 1,964,417 (89.02%)
- maximum matches for one S1: 11
- total positive links: 7,638,365
- S2 links: 3,693,619
- S3 links: 3,944,746
- S2-only S1: 143,029
- S3-only S1: 164,498
- both S2/S3: 1,776,047
- duplicate IDs inside GT match lists: 0
- invalid prefixes: none

### Referential integrity
- S1 dangling references: 0
- S2 dangling references: 0
- S2 referenced unique IDs: 3,693,619 (73.36%)
- S2 unreferenced records: 1,340,997 (26.64%)
- S3 dangling references: 0
- S3 referenced unique IDs: 3,944,746 (74.63%)
- S3 unreferenced records: 1,340,857 (25.37%)

Important reverse-cardinality observation:
each S2/S3 record appears in ground truth at most once, while one S1 may map to many S2/S3 records.

### Positive-pair variation sample
Sample size: 18,331 true positive pairs from ~5,000 S1 entities.

- country exact agreement: 100.00%
- raw name exact: 4.61%
- case-folded name exact: 10.45%
- name differs / needs ML: 89.55%
- raw address exact: 2.07%
- case-folded address exact: 6.93%
- target address missing: 4.52%
- non-ASCII/multilingual: 14.37%

Examples observed included typos, punctuation variation, reordered addresses,
legal suffix variation, accents, and cross-script/Devanagari variants.

The 100% country agreement is an observed sample statistic, not a reason to
hard-block all future retrieval by country. Country remains a soft signal and
the pipeline must remain open-set because the test set adds France.

## Phase 7 — normalization

Final canonical normalization is in `src/normalization.py`.

Rules:
- preserve raw values
- Unicode NFKC
- casefold
- punctuation/symbols -> spaces
- collapse whitespace
- preserve Unicode scripts
- do not ASCII-strip
- do not blindly remove stopwords/legal suffixes

A prior regex using `[^\\w\\s]` was rejected because it corrupted Devanagari.
The current implementation uses Unicode character categories.

Verified examples:
- `MS [Consultancy]` -> `ms consultancy`
- `Clairvoyant Récord Private Ltd` -> `clairvoyant récord private ltd`
- `सुप्रीम आईटी प्राइवेट लिमिटेड` -> unchanged script-preserving normalized text

## Phase 8 — official metric

Canonical implementation is in `src/metrics.py`.

Metric:
- entity-level macro F0.5
- beta = 0.5
- singleton actual=[] and predicted=[] -> 1.0
- singleton actual=[] and any prediction -> 0.0
- false negative -> 0.0

Unit tests passed for:
- perfect match
- true singleton
- false singleton
- false negative
- precision penalty
- recall penalty

## Phase 9 — grouped validation

Validation is grouped by S1 before pair generation.
No random pair split.

Original split:
- total S1: 2,206,821
- development: 1,765,456 (80%)
- holdout: 441,365 (20%)
- fast-dev slice: 50,000 S1

The current local machine rebuilt `output/s1_splits.tsv` and this file is now
the canonical split for future experiments. Do not regenerate it unless
intentionally changing the validation design.

## Phase 10 Step 1

Fast-dev preparation:
- 50,000 S1 records
- 173,414 total true links across S2+S3 in the original recorded run
- 2,792 singleton S1 entities
- average 3.47 true matches per S1
- average 3.67 among matched S1s

The current standalone Step 2 script reconstructs its required S1/GT state
from disk rather than relying on notebook RAM.

## Phase 10 Step 2 — S2 cheap blocking result

Current local canonical fast-dev slice:
- 50,000 S1
- 43,509 S1 with >=1 S2 match
- 83,737 true S2 links

Key A: `(country, exact normalized name)`
- link recall: 21.45%
- all-match entity recall: 10.90%
- at-least-one recall: 34.52%
- mean candidates/S1: 4.69
- median: 1
- P90: 6
- P99: 91
- max: 223
- reduction ratio: 99.9999%

Key B: `(country, first token)`
- link recall: 73.06%
- all-match entity recall: 59.19%
- at-least-one recall: 83.39%
- mean candidates/S1: 2,567.76
- median: 860
- P90: 7,395
- P99: 13,330
- max: 40,352
- reduction ratio: 99.9490%

Union A ∪ B:
- same recall metrics as Key B on this slice
- same candidate-size metrics as Key B

Interpretation:
- exact blocking is selective but misses too many true links
- first-token blocking recovers substantially more but is too broad
- a lexical retrieval method is needed between these extremes

## Phase 10 Step 3 — Character n-gram TF-IDF Retriever result

Configuration:
- analyzer: `char_wb`
- ngram_range: `(3, 4)`
- max_features: 50,000 (fitted on S1 fast-dev slice)
- vocabulary size: 25,912
- country partitioned (US vs US, India vs India)
- similarity lower bound: 0.40
- runtime: 3.76 minutes across all 5,034,616 S2 records in 250k chunks

Recall & Candidate Size Metrics:
- **Top-10**:
  - Link Recall: 65.00%
  - All-Match Entity Recall: 53.95%
  - At-Least-1 Recall: 74.83%
  - Mean Candidates/S1: 10.00 | Median: 10 | Max: 10
- **Top-25**:
  - Link Recall: 71.08%
  - All-Match Entity Recall: 60.38%
  - At-Least-1 Recall: 79.70%
  - Mean Candidates/S1: 24.98 | Median: 25 | Max: 25
- **Top-50**:
  - Link Recall: 74.58%
  - All-Match Entity Recall: 64.10%
  - At-Least-1 Recall: 82.76%
  - Mean Candidates/S1: 49.82 | Median: 50 | Max: 50
  - Reduction Ratio: 99.9990%

Artifacts Saved:
- Results: `output/phase10_step3_char_tfidf_results.tsv`
- Candidates: `output/phase10_step3_char_tfidf_top50.tsv`
- Vectorizer: `models/phase10_step3_char_tfidf_vectorizer.joblib`
- Config: `output/phase10_step3_char_tfidf_config.json`

Key Takeaways:
- Top-50 Character TF-IDF achieves **74.58% Link Recall** (beating First-Token Cheap Blocking at 73.06%) while slashing average candidate count by **51x** (from 2,567.76 down to 49.82).
- Zero OOM issues with `sparse_dot_topn` (elapsed time under 4 minutes on 5M records).
- All-match entity recall jumped from 59.19% to 64.10%.

## Phase 10 Step 4 — Word / Token TF-IDF Retriever result

Configuration:
- analyzer: `word`
- ngram_range: `(1, 2)` (unigrams + bigrams)
- max_features: 50,000 (fitted on S1 fast-dev slice)
- vocabulary size: 19,023
- country partitioned (US vs US, India vs India)
- similarity lower bound: 0.35
- runtime: 1.76 minutes across all 5,034,616 S2 records in 250k chunks

Recall & Candidate Size Metrics (Standalone Word TF-IDF):
- **Top-10**:
  - Link Recall: 41.18%
  - All-Match Entity Recall: 29.11%
  - At-Least-1 Recall: 53.21%
  - Mean Candidates/S1: 9.96 | Median: 10 | Max: 10
- **Top-25**:
  - Link Recall: 47.53%
  - All-Match Entity Recall: 34.53%
  - At-Least-1 Recall: 59.64%
  - Mean Candidates/S1: 24.89 | Median: 25 | Max: 25
- **Top-50**:
  - Link Recall: 52.76%
  - All-Match Entity Recall: 39.31%
  - At-Least-1 Recall: 64.73%
  - Mean Candidates/S1: 49.76 | Median: 50 | Max: 50

### Cumulative Multi-Channel Union (Char Top-50 ∪ Word Top-50)
- **Link Recall: 76.7857%** (up from 74.5764%, a **+2.21% absolute recall lift**)
- **All-Match Entity Recall: 66.8712%** (up from 64.1040%, a **+2.77% absolute lift**)
- **At-Least-1 Recall: 84.2998%** (up from 82.7622%, a **+1.54% absolute lift**)
- **Mean Candidates / S1: 84.74** | Median: 88 | P90: 98 | P99: 100 | Max: 100
- **Reduction Ratio: 99.9983%** (collapsing 5,034,616 S2 down to ~85 candidates/S1)

Artifacts Saved:
- Results: `output/phase10_step4_word_tfidf_results.tsv`
- Candidates: `output/phase10_step4_word_tfidf_top50.tsv`
- Cumulative Union: `output/phase10_step4_cumulative_union_results.tsv`
- Vectorizer: `models/phase10_step4_word_tfidf_vectorizer.joblib`
- Config: `output/phase10_step4_word_tfidf_config.json`

Key Takeaways:
- Word TF-IDF alone has lower recall (52.76%) than Char TF-IDF (74.58%) due to spelling typos and abbreviations, but it captures **orthogonal token permutations and distinctive word associations** that character n-grams miss.
- Combining both channels pushes Link Recall to **76.79%** and All-Match Recall to **66.87%**, while keeping candidate sets compact (mean 84.74 candidates per query).

## Phase 10 Step 5 — Address TF-IDF Retriever & 3-Channel Union

Configuration:
- analyzer: `word`
- ngram_range: `(1, 2)` (unigrams + bigrams)
- max_features: 50,000 (fitted on S1 fast-dev slice addresses)
- vocabulary size: 47,199
- country partitioned (US vs US, India vs India)
- similarity lower bound: 0.40
- runtime: 2.09 minutes across all 5,034,616 S2 records in 250k chunks

Recall & Candidate Size Metrics (Standalone Address TF-IDF):
- **Top-10**:
  - Link Recall: 70.10%
  - All-Match Entity Recall: 60.24%
  - At-Least-1 Recall: 79.02%
  - Mean Candidates/S1: 9.98 | Median: 10 | Max: 10
- **Top-20**:
  - Link Recall: 74.82%
  - All-Match Entity Recall: 65.80%
  - At-Least-1 Recall: 82.70%
  - Mean Candidates/S1: 19.85 | Median: 20 | Max: 20
- **Top-30**:
  - Link Recall: 77.26%
  - All-Match Entity Recall: 68.68%
  - At-Least-1 Recall: 84.61%
  - Mean Candidates/S1: 29.58 | Median: 30 | Max: 30

### Complete 3-Channel Cumulative Union (Char50 ∪ Word50 ∪ Addr30)
- **Link Recall: 94.8505%** (Astronomical **+18.06% absolute lift** over Name Char+Word Union)
- **All-Match Entity Recall: 91.5879%** (Massive **+24.72% absolute lift** over Name Union)
- **At-Least-1 Recall: 97.1293%** (Up from 84.30%)
- **Mean Candidates / S1: 113.09** | Median: 116 | P90: 127 | P99: 130 | Max: 130
- **Reduction Ratio: 99.9978%** (From 5,034,616 down to ~113 candidates per S1)

Artifacts Saved:
- Results: `output/phase10_step5_address_tfidf_results.tsv`
- Candidates: `output/phase10_step5_address_tfidf_top30.tsv`
- 3-Channel Union: `output/phase10_step5_cumulative_union_results.tsv`
- Vectorizer: `models/phase10_step5_address_tfidf_vectorizer.joblib`
- Config: `output/phase10_step5_address_tfidf_config.json`

Key Takeaways:
- Empirically validated the core hypothesis: Address TF-IDF acts as an indispensable candidate bridge for transliterated (English <-> Hindi) pairs, brand reorganizations, and legal subsidiary variations where company names differ but physical locations, building names, and numbers match.
- Achieved a **94.85% true-match link recall ceiling** while preserving extreme compactness (only ~113 candidates per S1 query).

## Architecture principles to preserve

- candidate generation sets the recall ceiling
- prefer compact candidate sets if recall stays high
- F0.5 is precision-heavy
- never force top-1
- multiple matches are valid
- country must remain open-set
- preserve Unicode
- missing==missing is not positive evidence
- preserve address numbers
- explicit contradictions matter
- Levenshtein is a feature, not a final rule
- character TF-IDF is a major lexical retrieval tool
- word TF-IDF complements character TF-IDF
- XGBoost is the first serious scoring baseline
- hard negatives matter
- split/group by S1
- test stays untouched until final inference
- no external business-data lookup/enrichment
- every final prediction must be present in the final candidate set
