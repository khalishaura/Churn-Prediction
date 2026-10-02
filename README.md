# Churn-Prediction

## Overview 
This project develops a classification machine learning model to predict customer churn for an e-commerce business, covering **exploratory data analysis, data preprocessing, model comparison, model explainability with SHAP (SHapley Additive exPlanations), and deployment as an interactive Streamlit application**. 

## Data Source 
The raw dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)

---

##  Project Workflow

1. **Exploratory Data Analysis (EDA)**: Analyzed feature distributions, churn rate, and customer behavior 
2. **Data Preprocessing**: Median imputation, IQR-based outlier handling, One-Hot Encoding, and StandardScaler. 
3. **Modeling**: Trained baseline models and optimized hyperparameters using GridSearchCV with PR-AUC as the scoring metric 
4. **Evaluation**: PR-AUC, Recall, Precision, F1-Score 
5. **Model Interpretability**: SHAP to explain global feature importance and individual predictions
6. **Deployment**: Interactive web app built with Streamlit

## Key Results

Several models were trained and compared. The best-performing model is a **Multi-Layer Perceptron (MLP)**:

| Metric  | Score  |
|---------|--------|
| PR-AUC   | **0.9931** |
| Recall   | **0.9472** | 
| Precision   | **0.9818** |
| F1-Score | **0.9642** |

> Models compared: Logistic Regression, MLP, Random Forest, XGBoost, LightGBM, Catboost

## Model Interpretability (SHAP)

SHAP values are used to show which features push a customer's churn probability up or down, both across the whole dataset and for a single customer. This makes the model's predictions easier to trust and act on.

**Top 3 drivers of churn:**  Tenure, customer complaints, and preferred order category (based on whole dataset/global) 

## Streamlit App

The app lets users input customer information and get:
* The predicted churn outcome and probability
* The top 3 factors contributing to the customer's churn or retention prediction based on SHAP

> 🔗 [**Live demo:**] (https://churn-prediction-ecommerce.streamlit.app/)
