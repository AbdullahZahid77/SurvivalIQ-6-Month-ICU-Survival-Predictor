# 🫀 SurvivalIQ — ML-Based 6-Month ICU Survival Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Model-HistGradientBoosting-4CAF50?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/ROC--AUC-0.945-success?style=for-the-badge"/>
</p>

<p align="center">
  An end-to-end machine learning project predicting 6-month survival probability for critically ill ICU patients,<br/>
  trained on the SUPPORT2 dataset and deployed as an interactive clinical decision-support prototype on Streamlit.
</p>

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Live Demo](#-live-demo)
3. [The Problem](#-the-problem)
4. [Dataset](#-dataset)
5. [ML Pipeline](#-ml-pipeline)
   - [Stage 1 — Data Preprocessing & EDA](#stage-1--data-preprocessing--eda)
   - [Stage 2 — Modelling & Tuning](#stage-2--modelling--tuning)
6. [Results](#-results)
7. [Interpretability](#-interpretability)
8. [App Features](#-app-features)
9. [Project Structure](#-project-structure)
10. [Local Setup](#-local-setup)
11. [Key Findings & Takeaways](#-key-findings--takeaways)
12. [Limitations & Future Work](#-limitations--future-work)
13. [Tech Stack](#-tech-stack)
14. [Clinical Disclaimer](#-clinical-disclaimer)

---

## 🎯 Project Overview

This project asks a single focused question:

> **Can modern machine learning meaningfully outperform a 30-year-old hand-engineered logistic regression for predicting 6-month ICU survival — while remaining clinically interpretable?**

The original SUPPORT prognostic model (Knaus et al., 1995) was built on the same patient cohort using 11 physiologic markers and achieved a ROC-AUC of ~0.78. This project benchmarks 7 model families, tunes the top 3, validates improvements statistically, and deploys the winning model in a production-style Streamlit app.

**Final answer: yes** — the tuned HistGradientBoosting regressor achieves ROC-AUC of **0.945**, with statistically validated improvements over the XGBoost baseline (bootstrap p = 0.018, Wilcoxon p = 0.0011).

---

## 🌐 Live Demo

👉 **[Launch SurvivalIQ on Streamlit](https://survivaliq-6-month-icu-survival-predictor.streamlit.app/)**

---

## 🏥 The Problem

Every day, ICU clinicians make high-stakes resource allocation decisions — often with incomplete information and cognitive load. Traditional severity scores (APACHE, SOFA, SAPS) are linear models that were designed decades ago. The core challenge:

- **The original SUPPORT model** is still widely cited as a benchmark despite being from 1995
- **Modern tree ensembles** can capture non-linear interactions between physiologic variables that linear models miss
- **Interpretability** is non-negotiable in healthcare — a black-box model won't earn clinical trust

This project closes the loop: train better models, prove the improvement is statistically real, explain _why_ predictions are made, and deploy an interface clinicians could actually interact with.

---

## 📊 Dataset

| Property               | Detail                                                                                 |
| ---------------------- | -------------------------------------------------------------------------------------- |
| **Name**               | SUPPORT2 (Study to Understand Prognoses, Preferences, Outcomes and Risks of Treatment) |
| **Source**             | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/880/support2)    |
| **Cohort**             | 9,105 seriously ill hospitalised adults across 5 U.S. medical centres, 1989–1994       |
| **Raw dimensions**     | 9,105 rows × 45 columns                                                                |
| **Target variable**    | `surv6m` — continuous 6-month survival probability ∈ [0, 0.948]                        |
| **Final feature set**  | 35 features (after leakage removal and redundancy diagnostics)                         |
| **Train / Test split** | 7,283 / 1,821 rows — stratified by quantile bins of `surv6m`                           |

**Notable data challenges:**

- `glucose`: 49.4% missing | `adlp`: 61.9% missing | `income`: 32.8% missing | `ph`: 25.1% missing
- Future-leaking columns (`death`, `hospdead`, `prg6m`, `surv2m`, `sfdm2`) had to be identified and removed before any modelling
- Mixed missingness mechanisms required different imputation strategies per variable

---

## 🔬 ML Pipeline

### Stage 1 — Data Preprocessing & EDA

All preprocessing decisions were fit **exclusively on training data** to prevent leakage.

| Step                              | Decision                                                                  | Rationale                                                                 |
| --------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Leakage removal**               | Dropped 6 future-information columns                                      | `surv2m`, `death`, `hospdead` are derived from the target                 |
| **Skewed variables**              | Median imputation (`charges`, `totmcst`, `glucose`, `scoma`)              | Median is robust to long tails; mean would inflate imputed values         |
| **Symmetric / bimodal variables** | MICE via `IterativeImputer` (`ph`, `sod`, `meanbp`, `hrt`, `temp`, `aps`) | Preserves multivariate structure between correlated physiologic variables |
| **Categorical variables**         | Mode imputation (`income`, `race`, `dnr`)                                 | Consistent with low missingness (<1–33%) and nominal nature               |
| **Expert constants**              | Fixed fills for physiologically bounded variables                         | Avoids distorting clinical ranges                                         |
| **Encoding**                      | Ordinal for `income` and `ca`; one-hot for `dzclass`, `race`, `dnr`       | Respects natural ordering where applicable                                |
| **Redundancy removal**            | 4 features removed via VIF and correlation diagnostics                    | Reduces multicollinearity for linear baselines                            |
| **Train/test split**              | Stratified by `surv6m` quantile bins                                      | Preserves target distribution across splits given bimodal shape           |

### Stage 2 — Modelling & Tuning

**Three-phase tuning framework:**

```
Phase A — Benchmark  →  Phase B — Randomised Search  →  Phase C — Grid Refinement
7 model families          Top 3 models, 40 iterations,       Learning rate ±50% grid
scored with 5-fold CV     5-fold CV (neg. RMSE)              search on the winner
```

**Model families benchmarked:**

| Family               | Models                                                     |
| -------------------- | ---------------------------------------------------------- |
| Linear / regularised | Ridge, ElasticNet, Ridge with logit-transform              |
| Tree ensembles       | Random Forest, XGBoost, HistGradientBoosting, LightGBM     |
| Neural network       | MLP (128→64 hidden units, early stopping, logit-transform) |

**Pipeline design:** Every model was wrapped in a scikit-learn `Pipeline` combining a `ColumnTransformer` and estimator. Linear and neural models received scaled continuous and ordinal features; tree models received raw features (invariant to scale). All predictions were post-clipped to `[0, 0.948]`.

---

## 📈 Results

### Benchmark Comparison

| Model                              | CV RMSE    | Test RMSE  | Test R²    | Fit Time (s) |
| ---------------------------------- | ---------- | ---------- | ---------- | ------------ |
| **HistGradientBoosting ★ (Final)** | **0.1081** | **0.1068** | **0.8226** | 13.3         |
| LightGBM                           | 0.1085     | 0.1065     | 0.8235     | 44.4         |
| LightGBM (tuned)                   | 0.1073     | 0.1069     | 0.8221     | 420.7        |
| XGBoost (tuned)                    | 0.1092     | 0.1073     | 0.8208     | 63.8         |
| XGBoost (baseline)                 | 0.1092     | 0.1086     | 0.8197     | 8.6          |
| Random Forest                      | 0.1196     | 0.1167     | 0.7879     | 274.3        |
| MLP (logit)                        | 0.1289     | 0.1241     | 0.7601     | 10.7         |
| Ridge                              | 0.1333     | 0.1337     | 0.7216     | 5.5          |
| ElasticNet                         | 0.1365     | 0.1362     | 0.7113     | 0.5          |

> **Why HistGradientBoosting over LightGBM?** LightGBM achieved marginally lower raw test RMSE, but required 5× more compute (420s vs 83s) with no improvement after tuning. HGB provided the optimal accuracy-to-efficiency ratio — the right call for real-world deployment.

### Statistical Validation

The 1.66% RMSE improvement was confirmed as genuine across three independent tests:

| Test                              | Result             | Interpretation                                                             |
| --------------------------------- | ------------------ | -------------------------------------------------------------------------- |
| **1,000-sample bootstrap 95% CI** | [−0.0035, −0.0003] | Entirely below zero — improvement is consistent across resampled test sets |
| **Paired bootstrap (two-sided)**  | p = 0.018          | Reject H₀ at α = 0.05; models are not equivalent                           |
| **Wilcoxon signed-rank test**     | p = 0.0011         | Distribution-free confirmation of consistently lower sample-wise errors    |

### Clinical Threshold Performance (surv6m < 0.5 = high risk)

| Metric                                     | Value     |
| ------------------------------------------ | --------- |
| Sensitivity (recall of high-risk patients) | **83.7%** |
| Specificity                                | **88.1%** |
| Positive Predictive Value (PPV)            | 82.6%     |
| Negative Predictive Value (NPV)            | 88.9%     |
| **ROC-AUC**                                | **0.945** |
| **PR-AUC**                                 | **0.924** |

The model's ROC-AUC of 0.945 substantially exceeds the original SUPPORT logistic regression benchmark of 0.78–0.79. Sensitivity of 83.7% means the model correctly identifies approximately 5 in 6 high-risk patients — clinically meaningful for triage.

---

## 🧠 Interpretability

Interpretability was treated as a first-class deliverable, not an afterthought. Two parallel importance methods were computed and consolidated by **average rank** to guard against single-method artefacts.

### Top 10 Features by Average Rank

| Feature                        | Perm. Rank | SHAP Rank | Avg Rank | Clinical Meaning                                                           |
| ------------------------------ | ---------- | --------- | -------- | -------------------------------------------------------------------------- |
| `age`                          | 4          | 4         | **4.0**  | Universal mortality predictor                                              |
| `aps` (APACHE III Score)       | 1          | 10        | **5.5**  | Composite ICU severity — directly captures acute physiologic deterioration |
| `dzclass_COPD/CHF/Cirrhosis`   | 5          | 6         | **5.5**  | Chronic disease burden strongly worsens prognosis                          |
| `avtisst` (TISS-28 Score)      | 10         | 5         | **7.5**  | Nursing workload proxy for treatment intensity                             |
| `ca` (cancer status)           | 2          | 13        | **7.5**  | Metastatic cancer sharply reduces 6-month survival                         |
| `meanbp` (mean blood pressure) | 11         | 8         | **9.5**  | Haemodynamic instability → poor prognosis                                  |
| `charges` (hospital charges)   | 13         | 9         | **11.0** | Proxy for care intensity                                                   |
| `hday` (days in hospital)      | 7          | 16        | **11.5** | Longer stays correlate with severity                                       |
| `glucose`                      | 22         | 2         | **12.0** | Metabolic derangement — strong physiologic signal                          |
| `temp` (temperature)           | 14         | 12        | **13.0** | Fever or hypothermia indicate systemic stress                              |

> **Note on `totmcst`:** This feature ranked #1 in raw SHAP magnitude but poorly in permutation importance — indicating it is a proxy for treatment intensity that correlates with severity without being causally independent. It was excluded from clinical conclusions. This disagreement analysis is exactly why dual-method ranking matters.

All top predictors align with established clinical literature, which provides confidence that the model is learning real medical signals rather than spurious correlations.

---

## 🎛️ App Features

The Streamlit app (`app.py`) exposes the full model in a clinical-facing interface:

| Feature                     | Description                                                                                                                                           |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Live prediction**         | Survival probability updates instantly as sidebar inputs are adjusted                                                                                 |
| **Survival gauge**          | Colour-coded Plotly dial (green → red) with 4-tier risk classification                                                                                |
| **Clinical flags**          | Auto-detects and highlights abnormal values: hypoalbuminaemia, hypotension, metastatic cancer, ARDS, renal impairment, elevated bilirubin, DNR status |
| **Clinical interpretation** | Risk-tier-specific narrative guidance contextualising the prediction                                                                                  |
| **Patient snapshot**        | Expandable full input summary table across all 33 variables                                                                                           |
| **Sensitivity analysis**    | "What-if" chart: vary one parameter across its full clinical range, see real-time survival impact with risk-zone shading                              |
| **Model info panel**        | Algorithm details, dataset provenance, performance benchmarks                                                                                         |

**Input coverage:**

- Demographics (age, sex, education, income, race/ethnicity)
- Diagnosis & comorbidities (cancer status, diabetes, dementia, DNR, disease class)
- Severity scores (APACHE III, coma score, TISS-28, ADL)
- Vital signs (mean BP, heart rate, respiratory rate, temperature)
- Laboratory values (albumin, bilirubin, creatinine, sodium, pH, glucose, BUN, WBC, PaO₂/FiO₂, urine output)
- Resource utilisation (hospital charges, total micro-costs, days in hospital)

---

## 📁 Project Structure

```
SurvivalIQ/
│
├── 📓 dataset_pre_processing_and_EDA.ipynb   ← Stage 1: full EDA, missingness analysis,
│                                                imputation strategy, feature engineering,
│                                                leakage removal, stratified split
│
├── 📓 model_training_and_tuning.ipynb        ← Stage 2: 7-model benchmark, 3-phase tuning,
│                                                bootstrap validation, SHAP + permutation
│                                                importance, clinical threshold evaluation
│
├── 🐍 app.py                                 ← Streamlit web application
│
├── 🤖 model_HistGradientBoosting_untuned.joblib  ← Serialised final model (add to repo or
│                                                     load from cloud storage)
│
├── 📄 requirements.txt                       ← Pinned dependencies
├── 📄 runtime.txt                            ← Python 3.10.13
└── 📄 README.md
```

---

## 🚀 Local Setup

### Prerequisites

- Python 3.10 or 3.11
- `scikit-learn==1.6.1` (**must match training environment** — joblib models are version-sensitive)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/AbdullahZahid77/SurvivalIQ-6-Month-ICU-Survival-Predictor.git
cd SurvivalIQ-6-Month-ICU-Survival-Predictor

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the model file in the project root
#    model_HistGradientBoosting_untuned.joblib

# 5. Run the app
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

### Reproducing the Model

Open and run the notebooks in order:

```
1. dataset_pre_processing_and_EDA.ipynb   → produces the clean train/test splits
2. model_training_and_tuning.ipynb        → trains, tunes, evaluates, saves the model
```

All random states are fixed at `random_state=42` throughout every estimator, splitter, and search for full reproducibility.

---

## 💡 Key Findings & Takeaways

**1. Data quality > model choice**
The most impactful decisions were made in Stage 1: careful leakage removal, imputation strategies matched to each variable's distribution, and stratified splitting. Switching between algorithms in Stage 2 moved RMSE by ~0.002; bad preprocessing could have invalidated the entire experiment.

**2. Bounded targets are quietly dangerous for standard regressors**
Unbounded regressors generate out-of-range predictions for `surv6m ∈ [0, 0.948]`. Logit-transform wrappers helped but still underperformed gradient boosting, which handles bounded targets implicitly through its tree structure. Residual plots and QQ plots revealed this — it wouldn't have been visible in summary metrics alone.

**3. Statistical significance ≠ clinical significance**
The RMSE improvement was 1.66% — real and statistically validated, but modest. The clinical threshold metrics (ROC-AUC 0.945, sensitivity 83.7%) are more meaningful for the actual use case. Both dimensions matter; reporting only one tells half the story.

**4. Dual-method feature importance is non-negotiable**
`totmcst` ranked #1 in SHAP but near the bottom in permutation importance — a clear signal of correlated-proxy leakage rather than a genuine predictor. Using only one method would have led to a misleading clinical conclusion.

**5. Efficiency is a valid model selection criterion**
LightGBM (tuned) required 5× more compute than HistGradientBoosting with no test performance gain. In real deployments with retraining pipelines, this matters.

---

## ⚠️ Limitations & Future Work

### Known Limitations

- **No external validation:** All results are from a held-out split of the same dataset. Generalisability to modern ICUs, non-U.S. hospitals, or different patient populations is unknown.
- **Dataset age:** SUPPORT2 was collected in 1989–1994. Treatments, equipment, and patient demographics have changed substantially.
- **Subgroup performance:** Patients with metastatic cancer (ca = 2) show a residual standard deviation of 0.143 vs 0.093 for non-cancer patients — the model is less reliable for this clinically important subgroup.
- **Bounded regression:** Post-hoc clipping to [0, 0.948] is practical but not theoretically principled.
- **Cost-proxy features:** `charges` and `totmcst` may introduce subtle circular bias (treatment intensity ↔ outcome).

### Future Directions

- **Survival analysis methods:** Cox proportional hazards, AFT-XGBoost, or DeepSurv to directly model right-censored time-to-event outcomes
- **Beta regression:** Naturally respects (0, 1) bounds without requiring post-hoc clipping
- **External validation:** Test on MIMIC-IV or eICU datasets to assess generalisability to modern clinical environments
- **Calibration improvement:** Isotonic regression or Platt scaling to tighten the calibration curve at the extremes
- **Real-time integration:** Explore FHIR-compatible APIs for pulling live lab values into the prediction pipeline

---

## 🛠 Tech Stack

| Layer                   | Tool                                                       |
| ----------------------- | ---------------------------------------------------------- |
| **Language**            | Python 3.10                                                |
| **ML framework**        | scikit-learn 1.6.1                                         |
| **Gradient boosting**   | HistGradientBoostingRegressor (sklearn), XGBoost, LightGBM |
| **Imputation**          | sklearn IterativeImputer (MICE)                            |
| **Interpretability**    | SHAP (TreeExplainer), sklearn permutation importance       |
| **Statistical testing** | Non-parametric bootstrap, Wilcoxon signed-rank test        |
| **Visualisation**       | Plotly, Matplotlib, Seaborn                                |
| **App framework**       | Streamlit ≥ 1.35                                           |
| **Model serialisation** | joblib                                                     |
| **Data handling**       | pandas ≥ 2.0, NumPy ≥ 1.24                                 |
| **Experiment tracking** | Notebook-native (random_state=42 throughout)               |

---

## 📚 References

1. Knaus WA et al. The SUPPORT prognostic model. _Ann Intern Med._ 1995;122(3):191–203.
2. Chen T, Guestrin C. XGBoost: A scalable tree boosting system. _KDD 2016._
3. Lundberg SM, Lee SI. A unified approach to interpreting model predictions. _NeurIPS 2017._
4. Accuracy of AI vs traditional scoring systems (APACHE, SOFA, SAPS) for ICU mortality prediction. _medRxiv._ 2026.01.14.
5. Azur MJ et al. Multiple imputation by chained equations. _Int J Methods Psychiatr Res._ 2011;20(1):40–49.
6. SUPPORT2 Dataset. UCI ML Repository. https://archive.ics.uci.edu/dataset/880/support2

---

## ⚕️ Clinical Disclaimer

This tool is for **educational and research purposes only**. It is not a validated clinical decision-support tool and must not be used to guide treatment decisions. Predictions are derived from a retrospective dataset collected in the early 1990s and have not been externally validated. Always defer to qualified clinical judgment.

---

<p align="center">
  Created by <strong>Abdullah Zahid</strong> · University of Technology Sydney
</p>
