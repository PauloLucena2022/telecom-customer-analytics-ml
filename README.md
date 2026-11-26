# Project Revenue Shield — Protecting High-Value Telecom Revenue with ML

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML%20Pipelines-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Champion%20Model-006400)
![Course](https://img.shields.io/badge/UTokyo%20GCI%202026-Final%20Assignment-1B365D)

> **Data-driven retention strategy and machine learning business proposal** developed for the **GCI World 2026 Final Assignment** (Matsuo-Iwasawa Lab, The University of Tokyo).

Instead of building a generic churn classifier across all 100,000 subscribers, **Project Revenue Shield** isolates the **High-Value Q4 cohort** (top 25% by monthly revenue, generating **48.54%** of total billing), pairs calibrated **XGBoost** churn probabilities with an **Expected Revenue-at-Risk (RaR)** prioritization layer, and stress-tests real-world financial ROI under Bayes base-rate recalibration.

---

## Executive Summary

* **Core Business Bottleneck:** Among 100,000 wireless subscribers (`Client.csv` + `Record.csv`), the top revenue quartile (**Q4**: `rev_Mean >= $70.75`, $N = 24,913$) drives **48.54% of monthly revenue** (mean ARPU **$114.04**, nearly 4.7x higher than Q1), putting **$1,389,943/month** at risk in the dataset.
* **Primary Churn Triggers in Q4:**
  * **Bill Shock & Plan Misalignment:** Q4 churners sit on **$10 cheaper base plans** (median `$59.99` vs. `$69.99`) yet pay significantly more in overage penalties (**34.0%** of total billing). Accounts with an Account Spending Limit (`asl_flag = Y`) churn **8.82 p.p. less** (`41.44%` vs. `50.26%`).
  * **Equipment Aging:** Handsets older than one year (`eqpdays > 365`) drive a **+9.59 p.p. jump in churn** (`55.21%` vs. `45.62%`).
  * **Early Usage Contraction:** Churning accounts exhibit nearly **3x steeper voice usage drops** (`change_mou` mean `-27.94` vs. `-9.73`) 31–60 days before exit.
* **Commercial Prioritization Lift:** Ranking Q4 subscribers by $\text{Expected RaR}_i = P(\text{churn}_i \mid X_i) \times \text{rev\_Mean}_i$ captures **37.90% of all Q4 monthly at-risk revenue** within just the **Top 20% of Q4** (only **5%** of the full subscriber base — a **1.89x lift** over random targeting and **2.36x lift** in the Top 10%).
* **Holdout Profit vs. Real-World Cannibalization:** On the holdout test set, Expected RaR ranking delivers **$33,529 in net contribution profit (106.64% ROI)**—more than **2x** probability-only targeting (**$16,320 | 50.87% ROI**) and nearly **5x** random targeting (**$7,020 | 24.81% ROI**). However, recalibrating the 50/50 balanced sample to realistic mature postpaid churn (**1.5%–2.0%/month**) exposes severe subsidy cannibalization (**-72.4% to -17.0% ROI**), motivating a **digital-first, low-cost offer redesign**, a **90-day A/B pilot**, and evolution toward **causal uplift modeling**.

---

## Key Results

### 1. Model Benchmarking (5-Fold Stratified CV on Q4 Cohort, $N = 19,930$)

All preprocessing steps (median imputation, missingness indicators, encoding, and scaling) were wrapped inside scikit-learn `Pipeline` objects to ensure **zero data leakage**.

| Model | CV ROC-AUC (mean ± std) | CV PR-AUC | CV Log-Loss | CV Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Champion)** | **0.6647 ± 0.0091** | **0.6438** | **0.6497** | **0.6200** |
| Random Forest | 0.6518 ± 0.0043 | 0.6255 | 0.6593 | 0.6115 |
| Logistic Regression | 0.6331 ± 0.0084 | 0.6060 | 0.6664 | 0.5973 |

* **Holdout Test Performance ($N = 4,983$):** ROC-AUC = **`0.6660`** | PR-AUC = **`0.6410`** | Log-Loss = **`0.6480`** | Accuracy@0.50 = **`0.6113`**.
* **Top 5 XGBoost Gain Features:** `eqpdays` (#1), `months` (#2), `ovr_mou_ratio` (#3, *engineered*), `mrc_rev_ratio` (#4, *engineered*), and `mou_trend_ratio` (#5, *engineered*). Five engineered domain features entered the Top 15.

---

### 2. Actionable Breakdown of the Top 20% Priority Tier (Test Set, $N = 996$)

| Intervention Track | Share | Observed Churn | Mean ARPU | Mean Overage | Actual Monthly RaR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Bill Shock Only** *(Plan Right-Sizing + ASL)* | 57.03% | 55.63% | $198.25 | $122.80 | $61,506.81 |
| **Dual Risk** *(Bill Shock + Old Handset)* | 26.71% | 61.28% | $171.37 | $108.62 | $26,236.39 |
| **Usage / General Risk** *(Proactive Concierge)* | 11.75% | 66.67% | $184.61 | $17.21 | $14,372.03 |
| **Old Handset Only** *(Device Upgrade)* | 4.52% | 64.44% | $166.96 | $9.23 | $4,668.62 |

> **83.74%** of the highest-priority subscribers pay over 60% of their bill in overage fees (`$108–$123/month`), while **31.23%** hold handsets older than 365 days.

---

### 3. Economic Verification (Holdout Set — Top 20% of Q4, 25% Save Rate, 6-Month Horizon)

| Targeting Strategy | Targeted | Actual Churners Hit | Gross Rev Saved (6m) | Total Program Cost | Net Profit | ROI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Expected RaR Ranking (Proposed)** | 996 | 586 | **$144,382.00** | $31,442.65 | **$33,529.25** | **106.64%** |
| **2. Churn Probability Only** | 996 | 679 | $107,551.75 | $32,078.66 | $16,319.63 | 50.87% |
| **3. Random Targeting (No ML)** | 996 | 486 | $78,475.17 | $28,293.84 | $7,019.99 | 24.81% |

* **Bayes Base-Rate Stress Test:** Adjusting from the 48.61% sampled churn rate to realistic mature postpaid churn (**1.5%–2.0%/month**) shows that expensive outbound call centers and upfront cash subsidies yield **-72.4% to -17.0% ROI** (turning positive only at **3.0% churn & 35% save rate → +21.3% ROI, $102,906 net profit**).
* **Strategic Recommendation:** Replace upfront cash handouts with **digital-first, low-friction interventions** (in-app/SMS `asl_flag` alerts, value-neutral plan right-sizing, and 12-month lock-in handset upgrades via trade-in / 0% financing), validated through a **90-day 80/20 controlled A/B pilot** before scaling into **uplift modeling**.


