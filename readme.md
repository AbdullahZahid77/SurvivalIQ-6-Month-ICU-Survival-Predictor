# 🫀 SurvivalIQ — 6-Month ICU Survival Predictor

A Streamlit web app that lets anyone interact with a **HistGradientBoosting** regression model
trained on the SUPPORT2 dataset to predict **6-month survival probability** for seriously ill ICU patients.

---

## 📁 Project Structure

```
survival_app/
├── app.py
├── requirements.txt
├── model_HistGradientBoosting_untuned.joblib   ← copy your model here
└── README.md
```

---

## 🚀 Setup (Windows — step by step)

### Step 1 — Prerequisites

Make sure you have **Python 3.10 or 3.11** installed.
Download from: https://www.python.org/downloads/

> ✅ During install, **check "Add Python to PATH"**

Verify in a new terminal (PowerShell or CMD):

```
python --version
```

---

### Step 2 — Create the project folder

Open PowerShell and run:

```powershell
mkdir C:\survival_app
cd C:\survival_app
```

Copy these files into `C:\survival_app\`:

- `app.py`
- `requirements.txt`
- `model_HistGradientBoosting_untuned.joblib`

---

### Step 3 — Create a virtual environment

```powershell
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the start of your prompt.

---

### Step 4 — Install dependencies

```powershell
pip install -r requirements.txt
```

> ⚠️ **Important:** `scikit-learn==1.6.1` is pinned because the model was trained with that version on Colab.
> Installing a different version (e.g. 1.5.x) will cause a version mismatch warning and may produce wrong predictions.

---

### Step 5 — Run the app

```powershell
streamlit run app.py
```

Your browser will automatically open at **http://localhost:8501** 🎉

---

## 🌐 Deploy to Streamlit Cloud (make it public/online)

### Step 1 — Push to GitHub

1. Create a new **public** GitHub repo (e.g. `survival-iq`)
2. Push your project folder:

```powershell
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/survival-iq.git
git push -u origin main
```

> ⚠️ **Model file size warning:** If `model_HistGradientBoosting_untuned.joblib` is > 100 MB,
> GitHub will reject it. In that case, use [Git LFS](https://git-lfs.com/):
>
> ```
> git lfs install
> git lfs track "*.joblib"
> git add .gitattributes
> git add model_HistGradientBoosting_untuned.joblib
> git commit -m "Add model via LFS"
> git push
> ```

### Step 2 — Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repo → branch: `main` → file: `app.py`
5. Click **Deploy** 🚀

Your app will be live at:

```
https://YOUR_USERNAME-survival-iq-app-XXXX.streamlit.app
```

---

## 🎛️ App Features

| Feature                     | Description                                                        |
| --------------------------- | ------------------------------------------------------------------ |
| **Live prediction**         | Survival probability updates instantly as you adjust inputs        |
| **Survival gauge**          | Colour-coded dial (green → red) with risk tier                     |
| **Clinical flags**          | Auto-detects abnormal values (hypoalbuminaemia, hypotension, etc.) |
| **Clinical interpretation** | Risk-tier-specific narrative guidance                              |
| **Patient snapshot**        | Full input summary table                                           |
| **Sensitivity analysis**    | "What-if" chart: vary one parameter, see impact on survival        |
| **Model info panel**        | Algorithm details, dataset info, performance benchmarks            |

---

## ⚠️ Dependencies & Compatibility Notes

| Package        | Version  | Why pinned                                                              |
| -------------- | -------- | ----------------------------------------------------------------------- |
| `scikit-learn` | `1.6.1`  | **Must match Colab training env** — joblib models are version-sensitive |
| `streamlit`    | `>=1.35` | Required for `st.columns` API used in the app                           |
| `plotly`       | `>=5.18` | Gauge chart + sensitivity line chart                                    |
| `pandas`       | `>=2.0`  | DataFrame handling                                                      |
| `numpy`        | `>=1.24` | Numerical ops & clipping                                                |

---

## 📌 Clinical Disclaimer

This tool is for **educational and research purposes only**.
It is not a validated clinical decision-support tool and must not be used to guide treatment decisions.
Always defer to qualified clinical judgment.
