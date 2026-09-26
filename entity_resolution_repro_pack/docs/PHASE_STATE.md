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
