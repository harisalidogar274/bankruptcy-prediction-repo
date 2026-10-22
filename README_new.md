# Bankruptcy Prediction Project

This project predicts whether a company will go bankrupt, using financial ratios from Polish companies. It uses the public Polish bankruptcy dataset (Zięba, Tomczak & Tomczak, 2016), which has five yearly files, each listing companies' financial ratios and whether they went bankrupt.

This README is built up section by section as the project grows.

**Contents**
1. [Data preparation](#1-data-preparation)
2. [Feature set evaluation](#2-feature-set-evaluation)
3. [Models and results](#3-models-and-results)
4. Conclusions *(to be added)*

---

## 1. Data preparation

**Files:** `data_prep_CHAID.py` (extreme values handled with CHAID) and `data_prep_winsor.py` (extreme values handled by winsorizing)

### 1.1 What it does, in one paragraph

Raw financial data is messy: it has missing entries, repeated rows, extreme values and columns we do not use. Before any model sees the data, the data preparation code cleans it into a consistent, fair format. The same steps run on every yearly file, and each file comes out as a **training set** (used to teach the models) and a **test set** (kept aside to check how well they work on companies they have never seen). There are two versions of the code that differ in **one step only**, how extreme values are handled, so the two approaches could be compared (section 1.10).

### 1.2 The financial ratios we use

In the starting set, every company is described by six ratios, plus the outcome we want to predict (`class`: 0 = did not go bankrupt, 1 = went bankrupt).

| Column | What it measures (plain English) | Formula |
|---|---|---|
| Attr1 | How profitable the company is relative to its size | net profit / total assets |
| Attr3 | Cash cushion for day-to-day operations | working capital / total assets |
| Attr6 | How much profit the company has kept over time | retained earnings / total assets |
| Attr8 | How well owners' equity covers what the company owes | book value of equity / total liabilities |
| Attr9 | How much revenue the company generates from its assets | sales / total assets |
| Attr46 | Ability to pay short-term debts without selling stock | (current assets − inventory) / short-term liabilities |
| class | **Outcome:** bankrupt (1) or not (0) | n/a |

These six are the starting set. Section 2 tests alternatives, and the winsorizing preparation code now reads its list of columns from `feature_configs.py`; every step below works the same for any list of columns.

### 1.3 The steps, from raw file to ready-to-use data

```
Raw CSV file
   │
   ├─ 1. Keep only the outcome + the decided ratios (FEATURES)
   ├─ 2. Convert everything to numbers; treat "?" and infinity as missing
   ├─ 3. Remove rows with missing values, then remove repeated rows
   ├─ 4. Split into training set and test set
   ├─ 5. Handle extreme values, in one of two ways:
   │       • data_prep_CHAID.py:  group each ratio into a few ranges (CHAID, backed by research)
   │       • data_prep_winsor.py: cap each ratio at its 1st and 99th percentile (winsorizing)
   ├─ 6. Put all ratios on a common scale (standardise)
   │
   ▼
Prepared training set + prepared test set
```

Steps 1 to 4 and 6 are identical in both versions, and both use the same random seed, so **both produce exactly the same training and test companies**. Only the ratio values differ.

### 1.4 How each part of the code meets the project requirements

The requirements below are the ones set for the data preparation. The step numbers match the numbered comments inside `prep()` in both files.

| # | Requirement | What the code does | Where in the code |
|---|---|---|---|
| R1 | **Work on any file with the same layout**, even if the number of rows differs | One function, `prep(path)`, holds all the logic and takes a file name. It relies on column names, never on row counts, so any file with the same columns works. A list of file names (`PATHS`) is looped through. | `prep()`, `PATHS` loop at the bottom |
| R2 | **Keep the code simple and commented; split each file into train and test and return both for every file** | Each step carries a plain comment. The split is 80% train / 20% test, with the same share of bankrupt companies in both. Results are stored per file as `prepped["file name"] = (train, test)`. | Comments throughout; step 4; `prepped` dictionary |
| R3 | **Keep only the outcome and the chosen ratios** | Only the outcome and the listed ratio columns are read (seven columns for the starting set), and everything else (including the company `id`) is ignored. | Step 1 (`usecols`), `FEATURES` list |
| R4 | **Remove the first row of column names so it is not treated as data** | The first row is used as column labels and never enters the data, so no company is lost. | Step 1 (`read_csv` header handling) |
| R5 | **Convert to decimal numbers, treat `?` and infinity as missing, then drop rows with anything missing** | All values become decimal numbers. Anything unreadable (`?`) or infinite becomes "missing", and any row with a missing value is dropped. | Steps 2 and 3 (`dropna`) |
| R6 | **Remove duplicate rows** | Rows that are identical across the kept columns (seven in the starting set) are dropped, keeping one copy. This happens before the split, so a duplicate cannot end up in both training and test data (which would flatter the results). | Step 3 (`drop_duplicates`) |
| R7 | **Standardise the data** | Each ratio is rescaled to an average of 0 and a typical spread of 1, so no ratio dominates just because its numbers are larger. The average and spread come from the training set only and are then applied to the test set. | Step 6 (`StandardScaler`) |
| R8 | **Use CHAID-based categorisation to handle outliers** | Each ratio is sorted into a few ranges ("bins"). Ranges that look equally risky are merged, using a statistical test (see 1.5). Extreme values fall into the first or last range, so they cannot distort the results. | `data_prep_CHAID.py`: step 5, `chaid_edges()` |
| R8 (alternative) | **Handle outliers by winsorizing at the 1st and 99th percentile**, so the two methods can be compared | Each ratio is capped at its 1st and 99th percentile (see 1.5). Exact values are kept. This is the only step that differs from R8 above. | `data_prep_winsor.py`: step 5 |

One small addition not in the original list: at the end (step 7) the outcome column is converted back to whole numbers 0/1, which is what the modelling tools expect.

### 1.5 The two ways of handling extreme values, in plain English

**CHAID (`data_prep_CHAID.py`)** groups a number into ranges by looking at how bankruptcy rates differ between them.

1. Start by cutting each ratio into 10 equally sized groups of companies.
2. Look at neighbouring groups. If companies in two neighbouring groups go bankrupt at about the same rate, merge them.
3. Repeat until every remaining neighbouring pair differs clearly (statistically significant at the 5% level).

The result is a handful of ranges per ratio. An extremely high or low value simply lands in the top or bottom range, which is how outliers are handled.

**Winsorizing (`data_prep_winsor.py`)** caps extreme values instead of grouping them.

1. For each ratio, find the value that 1% of training companies fall below and the value that 99% fall below.
2. Any company beyond those limits is given the limit value instead. Everything between the limits keeps its exact value.

Both methods learn their ranges or limits on the training data only, then apply them unchanged to the test data.

### 1.6 Safeguards against unfair results ("data leakage")

Anything the code *learns* from the data (the ranges or caps, and the averages used for scaling) is learned on the **training set only**, then applied to the test set. The test set is therefore a genuine stand-in for companies the models have never seen. Duplicates are also removed before the split for the same reason.

### 1.7 What the preparation produced on the 1-year file (starting six-ratio set)

| Stage | Rows |
|---|---|
| Rows in the raw file | 7,027 |
| Removed for missing values | 33 |
| Removed as duplicates | 116 |
| **Rows kept** | **6,878** |
| Training set | 5,502 (216 bankrupt) |
| Test set | 1,376 (54 bankrupt) |

These numbers are the same for both versions. Preparing more columns together (section 2) changes them slightly, because more columns mean more rows with a missing value but fewer rows that count as duplicates. About 4% of companies are bankrupt in this file, so the data is heavily imbalanced. That matters when choosing how to judge the models (see section 2, to be added).

For CHAID, the number of ranges settled on per ratio was: Attr1: 4, Attr3: 5, Attr6: 3, Attr8: 3, Attr9: 2, Attr46: 3. Winsorizing keeps thousands of distinct values per ratio.

### 1.8 How to run it

1. Put the CSV files in the same folder as the code (names follow `csv_result-1year.csv` … `csv_result-5year.csv`; edit `PATHS` if yours differ).
2. Run `python data_prep_CHAID.py` (or `python data_prep_winsor.py`). It prints a one-line summary per file.
3. In another script: `from data_prep_CHAID import prepped, FEATURES` (or `from data_prep_winsor import prepped, FEATURES`), then `train, test = prepped["csv_result-1year.csv"]`. Use `train[FEATURES]` as the model inputs and `train["class"]` as the outcome.

Requires Python with `pandas`, `numpy` and `scikit-learn`, plus `scipy` for the CHAID version only. The winsorizing version also needs `feature_configs.py` in the same folder, because it reads its list of columns from there.

### 1.9 Limitations and open items

- **CHAID trades detail for robustness.** Each ratio ends up with only 2 to 5 distinct values, so models see a coarser picture than the raw numbers. Winsorizing, which keeps the exact numbers, is provided in `data_prep_winsor.py` and compared in 1.10.
- **The CHAID ranges can shift.** Only 216 bankrupt companies are in the training set, so the statistical tests are modest and a different random split could give slightly different ranges.
- **Duplicates are judged on the kept columns only.** Two companies that differ in other ratios we do not use, but match on the kept ones, count as duplicates. With the starting six ratios this removed 116 rows, more than the missing values did (33).
- **Missing data was first checked for the starting six ratios.** Missing rates were under 1% for each of them in all five files, so dropping those rows loses very little. Bankrupt companies in year 5 had slightly higher missing rates, but on very small counts. Section 2 shows that other candidate ratios can behave very differently.
- **File naming:** the files `csv_result-1year.csv` to `csv_result-5year.csv` are named for convenience and match the source paper's 1st to 5th year datasets exactly. "1year" is the longest horizon (the label is bankruptcy five years after the data) and "5year" is the shortest (one year after).

### 1.10 Testing: CHAID versus winsorizing

**Setup.** Data prepared with each version (using the starting six ratios) was run through the same model, once for each of the five yearly files, and scored on three measures (explained in the glossary): **PR-AUC**, **recall** and **precision**. Because both versions use identical training and test companies, any difference comes from how extreme values were handled.

**Which version scored better** (✔ = winsorizing better, ✘ = CHAID better):

| File | PR-AUC | Recall | Precision |
|---|---|---|---|
| 1 | ✔ | ✔ | ✔ |
| 2 | ✔ | ✔ | ✘ |
| 3 | ✔ | ✔ | ✔ |
| 4 | ✔ | ✔ | ✔ |
| 5 | ✔ | ✔ | ✘ |
| **Winsorizing wins** | **5 of 5** | **5 of 5** | **3 of 5** |

**Note on PR-AUC:** the model outputs yes/no predictions, not probabilities, so our "PR-AUC" is a pseudo-PR-AUC that works out to roughly the average of precision and recall. It can't be compared with published PR-AUC figures, but it rises whenever either component improves, so it still ranks approaches fairly on the same data.

**Reading the result.** Winsorizing gave better PR-AUC and better recall in every file, and was better on all three measures in three of the five files (1, 3 and 4). In files 2 and 5, CHAID had the higher precision, meaning a larger share of its bankruptcy warnings turned out to be correct there. This table records which version won, not by how much.

**Caution.** Each file was tested once, on a single train/test split with relatively few bankrupt companies in each test set, so any single result could be partly down to chance. The consistency across all five files is what makes the overall direction credible.

**Decision.** Winsorizing (`data_prep_winsor.py`) is the version used for all modelling going forward, because it gave better PR-AUC and recall in every file tested. `data_prep_CHAID.py` (CHAID) is kept for reference and for reproducing the comparison.

**What threshold was used for this evaluation?**  
When called through the testing script, the model dynamically optimizes its decision threshold during training using Youden's J Index (Balanced Accuracy, maximizing True Positive Rate minus False Positive Rate) calculated via an ROC curve on out-of-fold cross-validation probability predictions from the training set.

### 1.11 Glossary

| Term | Meaning |
|---|---|
| Training set | The part of the data models learn from |
| Test set | The part held back to check how well the models work on unseen companies |
| Standardise | Rescale numbers to a common scale (average 0, typical spread 1) |
| Outlier | An unusually extreme value that can distort results |
| Winsorizing | Capping extreme values at set limits (here the 1st and 99th percentile) instead of removing them |
| CHAID | A method that groups a number into ranges based on how bankruptcy rates differ between them |
| Duplicate | A row identical to another row |
| Data leakage | When information from the test data accidentally influences training, making results look better than they really are |
| Imbalanced data | When one outcome (here, bankruptcy) is far rarer than the other |
| Recall (sensitivity) | Of the companies that really went bankrupt, the share the model caught |
| Precision | Of the companies the model flagged as bankrupt, the share that really were |
| PR-AUC | One overall score of how well the model finds bankrupt companies while keeping false alarms low; higher is better. Ours is a pseudo-PR-AUC (the model gives yes/no predictions, not probabilities), roughly the average of precision and recall, so it ranks our approaches but isn't comparable to published figures |
| Feature set | A chosen group of ratios given to the model as its inputs |

---

## 2. Feature set evaluation

**Files:** `feature_configs.py` (the candidate sets), `evaluating_Feature_sets.py` (runs the comparison), `data_prep_winsor.py` (prepares the data)

### 2.1 Purpose

Section 1 started from six ratios chosen up front. Before building further, we tested whether a different group of ratios (a **feature set**) lets the model catch more bankrupt companies. The candidates come from the source paper, which reports the ratios its own models relied on most in each year.

### 2.2 The seven candidate sets

| Set | Ratios (column numbers) | Size | Idea |
|---|---|---|---|
| A. Baseline | 1, 3, 6, 8, 9, 46 | 6 | Our starting set |
| B. Core | 25, 40, 52 | 3 | The only ratios in the paper's top 20 in all five years |
| C. Family | 22, 42, 25, 15, 40, 52 | 6 | A spread across financial families (profitability, equity, leverage, liquidity, turnover) from the paper's consistent ratios |
| D. Per-year top 6 | The paper's six best ratios for the matching year | 6 | Differs for each file |
| E. Baseline + core | A and B combined | 9 | Do the paper's core ratios add to ours? |
| F. Baseline + family | A and C combined | 12 | The same question with a broader group |
| G. Consistent | 13 ratios in the paper's top 20 in at least 4 of 5 years | 13 | Many ratios at once |

One ratio the paper ranks highly, **Attr27** (operating profit / financial expenses), was left out on purpose. In the 1-year file it is missing for 44% of bankrupt companies but only 3% of healthy ones, so dropping incomplete rows would have removed nearly half of the bankrupt companies.

### 2.3 How the test works

| File | Role |
|---|---|
| `feature_configs.py` | Lists the seven sets and the combined list of all their columns |
| `data_prep_winsor.py` | Cleans, splits and scales all candidate columns **once**, so every set is scored on the same companies and the same train/test split. Preparing each set separately would drop different rows and make the scores incomparable |
| `evaluating_Feature_sets.py` | For each set, trains the model (KNN with gradient boosting) on the training companies using only that set's ratios, scores it on the test companies, prints one comparison table, and names the winning set, weighing PR-AUC and sensitivity |

The scores are the three from section 1.10: PR-AUC (the pseudo version described there), sensitivity (the same as recall) and precision. The test runs on one file at a time (two lines name the file and its paper year, which also selects set D) and was repeated for all five files. On the 1-year file, preparing the 23 candidate columns together kept 6,841 of 7,027 rows (5,472 train, 1,369 test).

### 2.4 Results

**Best set in each file** (named by the script, which weighs PR-AUC and sensitivity; it was also the top PR-AUC set every time. Scores and runners-up shown are PR-AUC):

| File | Winner | Runner-up |
|---|---|---|
| 1 | F (0.443) | E (0.431) |
| 2 | A (0.440) | E (0.411) |
| 3 | F (0.504) | A (0.475) |
| 4 | G (0.441) | A (0.437) |
| 5 | F (0.481) | G (0.473) |

**Averages over the five files:**

| Set | PR-AUC | Sensitivity | Precision | Files won |
|---|---|---|---|---|
| **F. Baseline + family** | **0.453** | **0.792** | 0.104 | 3 |
| A. Baseline | 0.439 | 0.780 | 0.087 | 1 |
| G. Consistent | 0.434 | 0.750 | 0.107 | 1 |
| E. Baseline + core | 0.430 | 0.755 | 0.094 | 0 |
| C. Family | 0.386 | 0.655 | 0.100 | 0 |
| D. Per-year top 6 | 0.385 | 0.662 | 0.091 | 0 |
| B. Core | 0.371 | 0.652 | 0.074 | 0 |

The paper's ratios on their own (B, C, D) scored clearly below our starting set. Adding the family ratios to it (F) helped; adding the core ratios (E) did not. Every set catches most bankrupt companies (sensitivity 47% to 92%) at the cost of many false alarms (precision 5% to 19%), against a bankruptcy rate of about 4%.

### 2.5 Decision: set F (baseline + family) is carried forward

F has the best average PR-AUC and sensitivity, won three of the five files, and never ranked lower than third. Its warnings were also more often correct than the starting set's in all five files. The six ratios it adds to the starting set are:

| Column | What it measures (plain English) |
|---|---|
| Attr22 | Operating profit relative to company size |
| Attr42 | Operating profit relative to sales |
| Attr25 | Equity (excluding share capital) as a share of assets |
| Attr15 | Days of profit (before depreciation) needed to repay all liabilities |
| Attr40 | Ability to cover short-term debts without relying on inventory or receivables |
| Attr52 | Short-term debts relative to the cost of products sold, in days |

**How strong is the evidence?** F is the best-supported choice, not a proven one. Its lead over the starting set is small (about 1.5 points of PR-AUC on average), the starting set beat it in files 2 and 4, and the starting set is simpler (6 ratios against 12). Each test set holds only roughly 50 to 100 bankrupt companies, so a difference of one or two companies moves the scores. The winner was also picked using the same test data that scored it, so F's scores are probably somewhat optimistic.

**Next step in the code.** `FEATURES` in `data_prep_winsor.py` is set to set F's 12 columns instead of the combined list of all candidates, so the data are prepared on those 12 columns alone. This changes which companies are kept and the train/test split slightly, so later results will not match the comparison tables exactly. To rerun the comparison, set `FEATURES` back to `ALL_COLUMNS`.

**What threshold was used for this evaluation?**  
When called through the feature evaluation script, each feature set trains using the hybrid model, which automatically tunes its decision threshold during training using Youden's J Index (Balanced Accuracy) derived from out-of-fold cross-validation probability predictions on the training set.

## 3. Models and results

**Files:** `clean_KNN_model.py` (the hybrid classifier), `final_testing_file.py` (model evaluation script), `Threshold Comparisons.xlsx` (threshold analysis and evaluation graphs)

### 3.1 What the model is (`clean_KNN_model.py` / `KNNBoost`)

Our final model, `KNNBoost`, is a **hybrid ensemble classifier** combining non-parametric spatial density estimation with gradient-boosted decision trees. Rather than relying purely on global feature splits or a standalone distance classifier, it integrates both approaches:

1. **Spatial Feature Engineering via KNN ($k=50$):** 
   For every company, the model calculates the proportion of bankrupt (class 1) firms among its $k=50$ nearest neighbors using Euclidean distance over the financial ratios. To prevent data leakage during training, this uses a **leave-one-out** approach so a company never counts as its own neighbor. Chunked matrix operations keep memory usage minimal and execution fast.
2. **Gradient-Boosted Decision Trees:** 
   This local neighbor proportion is concatenated as an auxiliary feature alongside the standard financial ratios (e.g., the 12 ratios in Feature Set F). This combined feature matrix is then fed into a `HistGradientBoostingClassifier` (scikit-learn’s efficient gradient boosting implementation configured with class balancing and regularization). 
3. **Prediction & Inference:** 
   The booster evaluates the combined feature set to output a predicted bankruptcy probability for test companies, which is then compared against an optimized decision threshold to yield a final binary prediction (0 = healthy, 1 = bankrupt).

### 3.2 Threshold selection: Youden’s J vs. $F_\beta$ Score

In highly imbalanced datasets where only ~4% of companies are bankrupt, default decision thresholds (such as 0.50 or a static 10% cutoff) are sub-optimal. We evaluated threshold optimization strategies on out-of-fold cross-validation probabilities:

* **Youden’s J Index (Balanced Accuracy):** 
  Maximizes $\text{Sensitivity} + \text{Specificity} - 1$. While robust overall, Youden’s J treats True Positives and True Negatives equally. Because healthy companies vastly outnumber bankrupt ones, this tends to favor higher thresholds that lead to excessive **False Negatives** (missed bankruptcies).
* **$F_\beta$ Score ($\beta \in \{2, 3, 4, 5\}$):** 
  Weights Recall ($\beta$ times as important as Precision). For instance, setting $\beta = 3$ values catching bankruptcies **9 times as heavily** as avoiding false alarms ($\beta^2 = 9$). 

**Business Alignment:** In bankruptcy prediction, missing a failing company carries severe capital loss, whereas a false alarm merely warrants a secondary manual review. Thus, optimizing via $F_\beta$ systematically lowers decision thresholds to capture more bankruptcies.

### 3.3 Threshold comparison and performance analysis

The data and visualizations in `Threshold Comparisons.xlsx` evaluate how different threshold strategies ($Threshold = 0.10$, Balanced Accuracy / Youden's J, and $F_\beta$ variants with $\beta$ from 2 to 5) perform across all five yearly files:

* **Sensitivity (Recall) Gains:** Across the 5 yearly files, increasing $\beta$ aggressively drives up sensitivity. While a static 10% threshold achieves an average sensitivity of **83.4%**, moving to $F_\beta(3)$ raises average recall to **91.4%**, and $F_\beta(5)$ reaches **94.8%**. 
* **PR-AUC and Precision Trade-offs:** As thresholds shift lower to maximize recall, precision naturally decreases (averaging ~7% to 9% against a 4% baseline rate), while pseudo PR-AUC remains stable around ~47% to 51%. 
* **Robustness:** By tuning the threshold strictly on out-of-fold cross-validation predictions during training, the model avoids threshold overfitting, ensuring reliable generalization across unseen test years.

---

## 4. Conclusions *(to be added)*