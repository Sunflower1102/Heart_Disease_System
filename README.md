# ❤️ Heart Disease Diagnosis System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python"/>
  <img src="https://img.shields.io/badge/Streamlit-Framework-red?logo=streamlit"/>
  <img src="https://img.shields.io/badge/ML-Scikit--learn-orange?logo=scikitlearn"/>
  <img src="https://img.shields.io/badge/XAI-SHAP-purple"/>
  <img src="https://img.shields.io/badge/AI-Google%20Gemini-green?logo=google"/>
</p>

A two-app Streamlit system for clinical heart disease diagnosis, combining machine learning, explainable AI (SHAP), and Google Gemini for data-driven decision support.

---

## 🗂️ Project Structure

```
Heart_Disease_System/
├── admin_app.py          # Admin: data analysis, model training & deployment
├── dudoan.py             # Doctor: patient diagnosis interface
├── active_model.pkl      # Currently deployed model (auto-generated)
├── data/
│   ├── data3.csv         # Main clinical dataset
│   └── datadudoan.csv    # Sample prediction data
├── model/                # Pre-trained model files (.pkl)
│   ├── model_K-Nearest Neighbors (KNN).pkl
│   ├── model_Logistic Regression (Hồi quy Logistic).pkl
│   ├── model_Random Forest (Rừng ngẫu nhiên).pkl
│   ├── model_Support Vector Machine (SVM).pkl
│   └── model_XGBoost (Gradient Boosting).pkl
└── requirements.txt
```

---

## 🚀 Two-App Architecture

### 🔧 `admin_app.py` — Admin & Data Science Console

The admin panel covers the full ML pipeline in 4 tabs:

**Tab 1 — Data & AI Analysis**
- Upload any CSV clinical dataset
- Auto-detects and handles missing values (drop or KNN Imputer)
- Label Encoding for categorical variables
- **Google Gemini AI** explains column meanings and recommends the most important features (up to 12)

**Tab 2 — EDA Dashboard**
- KPI metrics: total patients, positive/negative rate, average age
- Distribution pie chart and correlation bar chart per target variable
- Per-feature analysis: histogram, violin plot, stacked bar (categorical)
- Interactive 3D scatter plot
- Statistical significance testing (T-test report)

**Tab 3 — Training & Optimization**
- Supports 5 algorithms: **SVM, Random Forest, Logistic Regression, KNN, XGBoost**
- Two training modes:
  - **Manual** — user-defined hyperparameters
  - **Auto-Tune** — RandomizedSearchCV with configurable iterations
- Imbalance handling: **SMOTE, ADASYN, BorderlineSMOTE**
- Full medical metrics: Sensitivity, Specificity, Precision, F1, **F2-Score**, AUC
- Advanced charts: Confusion Matrix, ROC Curve, **Calibration Plot**, **Decision Curve Analysis (DCA)**
- **Learning Curve** for overfitting analysis
- Real-time threshold slider (adjustable decision boundary)
- **Leaderboard** with two ranking modes:
  - Medical standard (F2-Score — prioritizes Recall)
  - Custom weighted score (user-defined Recall / Precision / F1 / Accuracy weights)
- Radar chart for multi-model comparison
- Download trained model as `.pkl`

**Tab 4 — Deployment Center**
- View active model metrics (Accuracy, F2, Recall, Precision, AUC)
- Compare candidate models against the live model with delta indicators
- One-click deployment: writes `active_model.pkl` to disk

---

### 🩺 `dudoan.py` — Doctor Diagnosis Interface

Loads `active_model.pkl` and provides two diagnosis modes:

**Manual input** — Enter each clinical indicator individually; instant risk prediction with gauge chart.

**Batch upload** — Upload a CSV of multiple patients; get risk scores for all and drill into any individual case.

**Result output for each patient:**
- Risk score (%) with color-coded verdict (Positive / Negative)
- Clinical recommendations based on risk level
- Feature comparison table (patient value vs. training median)
- Feature importance bar chart (Permutation Importance)
- Optional **SHAP Waterfall chart** for deep explainability

---

## 📊 Dataset

The system ships with a real cardiac clinical dataset (`data3.csv`) containing 55 columns including:

| Category | Features |
|---|---|
| Demographics | AGE, GENDER, RURAL |
| Vitals & Labs | HB, TLC, PLATELETS, GLUCOSE, UREA, CREATININE, BNP |
| Cardiac indicators | EF, RAISED CARDIAC ENZYMES, CAD, HTN, DM, CKD |
| Diagnoses | HEART FAILURE, ACS, STEMI, HFREF, HFNEF, VALVULAR, AF, ... |
| Outcome | OUTCOME (DISCHARGE / DEATH) |

---

## ⚙️ Setup

### Install dependencies
```bash
pip install -r requirements.txt
pip install xgboost
```

### Run Admin App
```bash
streamlit run admin_app.py
```

### Run Doctor App
```bash
streamlit run dudoan.py
```

> **Workflow:** Run `admin_app.py` first → upload data → train a model → deploy it (Tab 4) → then use `dudoan.py` for patient diagnosis.

---

## 🛠️ Tech Stack

| Component | Library |
|---|---|
| Web framework | Streamlit |
| ML models | scikit-learn, XGBoost |
| Imbalance handling | imbalanced-learn (SMOTE, ADASYN, BorderlineSMOTE) |
| Visualization | Plotly Express, Plotly Graph Objects, Matplotlib |
| Explainable AI | SHAP, Permutation Importance |
| Generative AI | Google Gemini API (`gemini-2.0-flash`) |
| Data processing | Pandas, NumPy, SciPy |
| Model persistence | Joblib |

---

## 📋 Requirements

```
streamlit
pandas
numpy
scikit-learn
plotly
joblib
matplotlib
scipy
shap
imbalanced-learn
google-generativeai
xgboost
```

---

## ⚠️ Disclaimer

This system is intended for **research and educational purposes only**. All predictions are model-based estimates and must not replace clinical judgment or professional medical diagnosis.
