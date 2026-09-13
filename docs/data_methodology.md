# Data Methodology — English, Tamil, Tanglish, Sinhala, Singlish

This document records how each of the five language categories was sourced, cleaned, and sampled for the multilingual toxic-comment detection project. Sections 1–4 (English, Tamil, Tanglish, Sinhala) were sourced and processed as pre-existing labelled datasets; Section 5 (Singlish) was collected and labelled from scratch by the team. Covers: `english_clean.csv`, `tamil_pure_clean.csv`, `tanglish_clean.csv`, `sinhala_clean.csv`, and the Singlish subset derived from `cleaned_singlish_toxic_dataset.xlsx`.

## Common schema

All five files share the same structure:

| Column | Description |
|---|---|
| `text` | The cleaned comment text |
| `label` | Binary — `1` = toxic/offensive, `0` = non-toxic/not-offensive |
| `language_tag` | One of `english`, `tamil`, `tanglish`, `sinhala`, `singlish` |

Text was intentionally **not** lowercased and stopwords were **not** removed. The project uses a multilingual transformer (MuRIL/XLM-R) for classification, and aggressive classical-NLP preprocessing can discard signal these models otherwise use (e.g. casing, punctuation patterns like repeated "!!!", and code-switching patterns are all potentially informative).

All random sampling used `random_state=42` for reproducibility.

---

## 1. English — `english_clean.csv`

**Source:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge) (Kaggle), Wikipedia talk-page comments.

**Original data:** 159,571 rows, with six sub-labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.

**Label mapping:** Collapsed the six sub-labels into a single binary `label` — a comment is `1` (toxic) if *any* of the six sub-labels is `1`, else `0`. Raw class balance after collapsing: 143,346 non-toxic / 16,225 toxic (~90/10 split).

**Sampling:** The raw ~90/10 imbalance was reduced by stratified sampling: 3,500 toxic rows (nearly all available) + 6,500 non-toxic rows (random sample), giving 10,000 rows at a ~65/35 non-toxic/toxic ratio — closer to balanced without discarding the minority class entirely.

**Cleaning:** Stripped URLs, wiki-markup artifacts (`[[User:...]]` links, `==Header==` sections — leftovers from the Wikipedia source), newline characters, and collapsed excess whitespace.

**Final size:** 10,000 rows (6,500 non-toxic / 3,500 toxic).

---

## 2 & 3. Tamil and Tanglish — `tamil_pure_clean.csv`, `tanglish_clean.csv`

**Source:** [DravidianCodeMix Dataset](https://github.com/bharathichezhiyan/DravidianCodeMix-Dataset), Tamil-English offensive-language subset — Tamil YouTube comments (`tamil_offensive_full_train.tsv`, tab-separated despite the `.csv` name; no header row in the raw file).

**Original data:** 35,139 rows with a six-category label scheme: `Not_offensive`, `Offensive_Untargetede`, `Offensive_Targeted_Insult_Group`, `Offensive_Targeted_Insult_Individual`, `Offensive_Targeted_Insult_Other`, `not-Tamil`.

**Pre-processing decisions:**
- Dropped 1,454 rows labelled `not-Tamil` — these were flagged by the original annotators as not actually being Tamil/Tamil-English content, so keeping them would have contaminated the language category.
- Mapped all four `Offensive_*` categories to `label = 1`; `Not_offensive` to `label = 0`. Remaining data: 33,685 rows (25,425 non-offensive / 8,260 offensive).

**Language script split — a key methodological decision:** the source dataset mixes two distinct writing styles within the same "Tamil-English" label — some comments are written entirely in Tamil script, others are Romanized (Tamil words spelled in Latin letters, colloquial "Tanglish" style). Since the project treats these as two separate target languages, the pooled data was split using a simple heuristic:

```python
def is_majority_tamil_script(text):
    tamil_chars = count of characters in Unicode range U+0B80–U+0BFF
    latin_chars = count of a-z, A-Z characters
    return tamil_chars > latin_chars
```

Comments with more Tamil-script characters than Latin characters were classified as **pure Tamil**; the rest as **Tanglish**. Rows with no identifiable script characters at all (pure numbers/emojis) were dropped.

**Validation:** A manual spot-check of 15 randomly sampled rows from each resulting subset (seed 1) found no misclassifications — the pure-Tamil sample was consistently Tamil-script throughout, and the Tanglish sample was consistently Latin-script (with one example containing a couple of embedded Tamil-script words within an otherwise Romanized sentence — a genuine code-mixing example, correctly bucketed since Latin characters still dominated). This heuristic has not been validated beyond this 30-row spot-check; a larger-scale manual audit would strengthen confidence further.

**Pool sizes after split (before final sampling):**
- Pure Tamil pool: 6,406 rows (5,137 non-offensive / 1,269 offensive)
- Tanglish pool: 27,277 rows (20,286 non-offensive / 6,991 offensive)

**Sampling:**
- Pure Tamil: 1,200 toxic (near-exhaustive, given only 1,269 were available) + 2,400 non-toxic → 3,600 rows (~67/33 split)
- Tanglish: 3,000 toxic + 5,000 non-toxic → 8,000 rows (~63/37 split)

**Cleaning:** Stripped URLs, @mentions, newlines, collapsed whitespace (applied before the script split).

**Final sizes:** Tamil — 3,600 rows (2,400 non-toxic / 1,200 toxic). Tanglish — 8,000 rows (5,000 non-toxic / 3,000 toxic).

**Known limitation:** the pure-Tamil subset is notably smaller than the other three language categories (3,600 vs. 7,000–10,000), because the source pool only contained 1,269 offensive pure-Tamil-script comments. This means Tamil is underrepresented in the pooled multilingual training set unless addressed at merge/training time (e.g. upsampling, or accounting for it via per-language evaluation).

---

## 4. Sinhala — `sinhala_clean.csv`

**Source:** [SOLD (Sinhala Offensive Language Dataset)](https://huggingface.co/datasets/sinhala-nlp/SOLD) — Ranasinghe, T., Anuradha, I., Premasiri, D., Silva, K., Hettiarachchi, H., Uyangodage, L., & Zampieri, M. (2022). Manually annotated Sinhala tweets.

**Original data:** 7,500 rows with columns `post_id`, `text`, `tokens`, `rationales`, `label` (`OFF` / `NOT`). The `tokens` (word-tokenized text) and `rationales` (token-level annotations marking which words triggered an offensive judgment) columns were not used, since this project only needs sentence-level binary classification.

**Label mapping:** `OFF` → `1`, `NOT` → `0`. Raw balance: 4,324 `NOT` / 3,176 `OFF` (~58/42, close to balanced already).

**Sampling:** 3,000 toxic + 4,000 non-toxic → 7,000 rows (~57/43 split) — a light downsample from the near-balanced original, no aggressive rebalancing needed.

**Cleaning:** Stripped `@USER` anonymization placeholders (a Twitter-anonymization artifact present in the original dataset), URLs, newlines, collapsed whitespace.

**Final size:** 7,000 rows (4,000 non-toxic / 3,000 toxic).

---

## 5. Singlish — derived from `cleaned_singlish_toxic_dataset.xlsx`

**Source:** Manually collected by a team member (Singlish comments — Sinhala content written in Romanized/Latin letters — scraped from public social media). Unlike the other four languages, no pre-existing labelled dataset was available for Singlish, so this data was collected and labelled from scratch.

**Original collected data:** 40,524 raw records. A cleaning pass (documented in the collector's own `cleaning_report.md`) removed subtitle noise (4,165 rows, 10.28%) and duplicates (134 rows, 0.33%), leaving 28,567 cleaned records, of which 28,522 were confirmed Singlish (the rest were a small number of mixed-language, English, or Sinhala-script outliers not relevant to this category).

**Labelling:** Each of the 28,567 cleaned records was labelled `toxic` or `non-toxic`. Raw label distribution: 27,111 non-toxic / 1,456 toxic (~5.1% toxic) — a much lower toxic rate than the other four languages' raw source data, reflecting the nature of the collected source material rather than a labelling scope difference (worth re-confirming with the collector).

**Label mapping:** `toxic` → `1`, `non-toxic` → `0`.

**Sampling:** Given only 1,456 toxic rows were available, took 1,400 toxic (near-exhaustive) + 2,800 non-toxic → 4,200 rows (~67/33 non-toxic/toxic split) — matching the sampling approach used for the similarly toxic-scarce pure-Tamil subset.

**Cleaning:** Text was already lowercased and cleaned by the collector's own pipeline (`clean_text` column used as the source field); no additional cleaning applied on top.

**Final size:** 4,200 rows (2,800 non-toxic / 1,400 toxic).

**Known limitation:** the raw collected Singlish data has a substantially lower natural toxic rate (~5%) than the other four languages (33–43%), meaning the usable toxic-example pool is small (1,456 rows total). If more toxic Singlish examples are needed later (e.g. for a larger final training set), further manual collection/labelling focused specifically on toxic content would be needed, since simply collecting more general comments would mostly add further non-toxic examples.

---

## Cross-language summary (EDA)

| Language | Total | Non-toxic | Toxic | % Toxic |
|---|---|---|---|---|
| English | 10,000 | 6,500 | 3,500 | 35.0% |
| Tamil | 3,600 | 2,400 | 1,200 | 33.3% |
| Tanglish | 8,000 | 5,000 | 3,000 | 37.5% |
| Sinhala | 7,000 | 4,000 | 3,000 | 42.9% |
| Singlish | 4,200 | 2,800 | 1,400 | 33.3% |

Toxic proportions are reasonably consistent across all five languages (33–43%) *at the sampled level*, which was a deliberate goal of the sampling approach — this avoids the model learning a spurious association between language and toxicity likelihood. Note this consistency was achieved through sampling; the raw collected/source toxic rates varied far more widely (e.g. Singlish's raw source data was only ~5% toxic before sampling).

**Average comment length by label (characters):**

| Language | Non-toxic avg | Toxic avg |
|---|---|---|
| English | 391.5 | 284.6 |
| Tamil | 108.6 | 135.3 |
| Tanglish | 59.9 | 77.6 |
| Sinhala | 102.0 | 124.2 |
| Singlish | 76.1 | 74.7 |

Three distinct patterns emerge across the five languages:
- **English:** toxic comments are **shorter** than non-toxic ones (consistent with short insults vs. longer collaborative Wikipedia discussion text).
- **Tamil, Tanglish, Sinhala:** the pattern **reverses** — toxic comments tend to be **longer** than non-toxic ones.
- **Singlish:** essentially **no length difference** between toxic and non-toxic comments (74.7 vs. 76.1 characters).

This is a genuine cross-linguistic finding worth highlighting in the report: comment length is not a reliable universal signal for toxicity, and a model could learn to over-rely on length as a shortcut rather than actual content if not careful — this risk is more acute once all languages are pooled into a single multilingual model. The fact that the length-toxicity relationship differs (and in Singlish's case, disappears entirely) across languages is itself evidence that a genuinely content-aware multilingual model is needed, rather than one that could get away with shallow heuristics.

---

## Open items for the team

- **Tamil underrepresentation**: Tamil (3,600 rows) and Singlish (4,200 rows) are both notably smaller than English/Tanglish/Sinhala (7,000–10,000 rows), because both had a limited pool of toxic examples to sample from. Decide whether to upsample these two, downsample the larger three to match, or leave as-is and report per-language performance to check for degradation.
- **Script-split heuristic** (Tamil/Tanglish): validated on a 30-row manual spot-check only; a larger audit (e.g. 100+ rows) would give higher confidence before final training.
- **Length imbalance**: worth checking post-training whether the model's errors correlate with comment length, particularly for English (opposite pattern from Tamil/Tanglish/Sinhala) and Singlish (no pattern at all).
- **Singlish raw toxic rate**: the ~5% raw toxic rate in the collected Singlish source data is worth confirming with the collector — whether it reflects the actual source material or a more conservative labelling approach than used elsewhere.
