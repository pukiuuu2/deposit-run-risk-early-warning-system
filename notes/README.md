# 🏦 Banking Deposit Run Risk Early Warning System

## 🌐 Live Dashboard

👉 https://deposit-run-risk-early-warning-system-d2faso5emwemre3cxvfvn6.streamlit.app/

An end-to-end Machine Learning project that predicts customer deposit runoff risk and estimates **Deposit at Risk** to support liquidity risk monitoring, customer prioritisation, and proactive retention strategies.

---

# 📖 Project Overview

Customer deposits are one of the most important funding sources for commercial banks.

Unexpected customer withdrawals may increase liquidity pressure and funding costs. Traditional customer churn models estimate the probability that a customer leaves the bank, but Treasury teams also need to understand the **financial impact** of potential deposit withdrawals.

This project develops a **Deposit Run Risk Early Warning System** that combines machine learning predictions with customer balances to estimate the expected deposits at risk.

The final solution includes:

- Customer runoff probability prediction using Machine Learning
- Deposit at Risk estimation
- SHAP explainability
- Interactive Streamlit dashboard
- End-to-end GitHub portfolio project

---

# 🎯 Business Objective

The project aims to answer the following business questions:

- Which customers are most likely to withdraw their deposits?
- How much funding is potentially at risk?
- Which customers should Relationship Managers prioritise?
- How can Treasury monitor portfolio-level liquidity risk?

---

# ⚙️ Project Workflow

```

Kaggle Dataset

↓

Feature Engineering

↓

Model Comparison

(Logistic Regression / Random Forest / XGBoost)

↓

Best Model Selection

↓

SHAP Explainability

↓

Deposit at Risk Calculation

↓

Interactive Streamlit Dashboard

```

---

# 📊 Model Performance

| Model | ROC-AUC |
|------|---------:|
| Logistic Regression | 0.8095 |
| Random Forest | 0.8645 |
| **XGBoost** | **0.8652** |

XGBoost achieved the highest predictive performance and was selected as the final production model.

---

# 💡 Feature Engineering

Several business-driven features were created to improve predictive performance, including:

- Balance / Salary Ratio
- Balance per Product
- Age / Tenure Ratio
- High Balance + Inactive Customer Flag
- Customer Value

These engineered variables significantly improved model performance over the baseline Logistic Regression model.

---

# 🔍 Model Explainability

SHAP (SHapley Additive exPlanations) was used to improve model transparency.

The explainability analysis allows business users to understand:

- Why a customer is classified as high risk
- Which features contribute most to the prediction
- Whether the prediction is driven by behavioural or demographic factors

---

# 💰 Deposit at Risk

Rather than using churn probability alone, the project estimates:

Deposit at Risk = Customer Balance × Runoff Probability

This metric allows Treasury teams to prioritise customers based on expected funding exposure rather than probability alone.

---

# 📈 Dashboard

The project includes an interactive Streamlit dashboard providing:

- Portfolio overview
- Risk segmentation
- Deposit at Risk monitoring
- Customer explorer
- Customer drill-down
- Executive insights

> *(Dashboard screenshots will be added here.)*

---

# 📂 Repository Structure

```

deposit-run-risk-early-warning-system/

├── app/
│ └── Streamlit dashboard
│
├── data/
│ └── Dataset (excluded from Git)
│
├── models/
│ └── Trained XGBoost model
│
├── notebooks/
│ └── Model development
│
├── requirements.txt
│
└── README.md

```

---

# 📥 Dataset

This project uses the public **Bank Customer Churn Modelling** dataset from Kaggle.

Download the dataset here:

https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling

After downloading, place the dataset in:

```

data/
└── Churn_Modelling.csv

```

The dataset is intentionally excluded from this repository to keep the repository lightweight and to respect the original dataset distribution.

---

# 🚀 Quick Start

## 1. Clone the repository

```bash
git clone https://github.com/pukiuuu2/deposit-run-risk-early-warning-system.git