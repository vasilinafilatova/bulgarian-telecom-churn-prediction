# Bulgarian Telecom Customer Churn Prediction

Customer churn analysis and machine learning prediction using Python, SMOTE and classification models.

## Project Overview

This project analyzes customer churn for a Bulgarian telecommunications company using exploratory data analysis, feature engineering and machine learning.

The goal was to identify customer characteristics associated with churn, compare multiple classification algorithms and translate model findings into actionable customer-retention recommendations.

The project also explores how machine learning predictions could be operationalized through a customer churn prediction interface.

## Business Problem

Customer churn directly affects revenue, customer acquisition costs and long-term profitability in the telecommunications industry.

The analysis addresses several business questions:

- Which customer characteristics are associated with churn?
- Can machine learning identify customers at higher risk of leaving?
- How does severe class imbalance affect model performance?
- Which customer segments should be prioritized for retention?
- How can model outputs be translated into practical business actions?

## Dataset

The project uses the Bulgarian Telecom Customer Churn Dataset.

- **8,453 customer records**
- **14 original variables**
- Target variable: `CHURN`

### Main Feature Groups

**Customer Profile**
- CRM value segment
- Effective segment
- Billing ZIP
- Account / geographic information

**Subscription Activity**
- Active subscribers
- Inactive subscribers
- Suspended subscribers
- Total subscriptions

**Revenue**
- Average Mobile Revenue
- Average Fixed Revenue
- Total Revenue
- ARPU

## Tools & Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- XGBoost
- Random Forest
- Logistic Regression
- Support Vector Machine
- SMOTE
- Matplotlib
- Seaborn
- Gradio
- Joblib

## Analysis Workflow

### 1. Data Cleaning

The dataset was inspected for:

- duplicates
- missing values
- inconsistent data types
- invalid text-based missing values
- outliers

Missing numerical values were handled using mean imputation, while categorical values were imputed using the mode.

## 2. Feature Engineering

Three additional business-oriented features were created:

### Activity Ratio

Measures customer activity relative to total subscriptions.

`Active_subscribers / Total_SUBs`

### Mobile Revenue Share

Measures the share of customer revenue generated from mobile services.

`AvgMobileRevenue / TotalRevenue`

### Revenue per Subscriber

Measures average revenue generated per subscription.

`TotalRevenue / Total_SUBs`

## 3. Outlier Treatment

The IQR method was used to identify extreme numerical values.

Rather than deleting potentially valuable high-value customers, extreme observations were capped to reduce their influence while preserving valid business cases.

## 4. Class Imbalance

Customer churn was highly imbalanced:

- **93.5% non-churners**
- **6.5% churners**

This created an important modelling challenge because a model could achieve high accuracy while failing to identify actual churn cases.

SMOTE was therefore applied to the training dataset to improve minority-class representation.

## 5. Machine Learning Models

Five supervised classification models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- Support Vector Machine

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

## Initial Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.577 | 0.078 | 0.509 | 0.135 | 0.545 |
| Random Forest | 0.933 | 0.000 | 0.000 | 0.000 | 0.499 |
| XGBoost | 0.888 | 0.065 | 0.055 | 0.059 | 0.500 |
| **LightGBM** | **0.826** | **0.113** | **0.245** | **0.155** | **0.556** |
| SVM | 0.672 | 0.075 | 0.355 | 0.123 | 0.525 |

The results demonstrate why accuracy alone can be misleading for highly imbalanced datasets.

Random Forest achieved very high overall accuracy but failed to detect churners effectively.

LightGBM provided the strongest balance between F1 score and ROC-AUC in the initial evaluation.

## SMOTE Experiment

SMOTE was applied to the training data to increase minority-class representation.

The training distribution was changed from:

- Non-churn: 6,323
- Churn: 439

to:

- Non-churn: 6,323
- Churn: 6,323

This experiment highlighted that balancing the training data alone does not guarantee strong minority-class prediction and that model threshold selection remains important.

## Model Selection

LightGBM was selected as the main model because it provided the strongest overall balance in the original evaluation, achieving:

- **F1 Score: 0.155**
- **ROC-AUC: 0.556**

The analysis also demonstrates the importance of threshold optimization and cost-sensitive evaluation when modelling rare events such as customer churn.

## Feature Importance

The strongest LightGBM predictors included:

- Billing ZIP
- Active subscribers
- Total subscriptions
- Inactive subscribers
- Revenue per subscriber
- ARPU
- Average mobile revenue
- Total revenue
- Activity ratio

These results suggest that churn risk is associated with a combination of:

- geographic segmentation
- customer activity
- subscription portfolio
- inactivity
- revenue characteristics

Feature importance is interpreted as directional evidence rather than causal proof.

## Key Business Insights

### Customer Activity

Inactive and suspended customers represent an important early warning signal for potential churn.

### High-Value Customers

High-ARPU customers deserve special attention because losing these customers can create disproportionate revenue impact.

### Geographic Segmentation

Location-related variables appear among important model features, suggesting that regional customer behaviour or service conditions may differ.

### Churn Is a Rare Event

Because only around 6.5% of customers churn, traditional accuracy should not be used as the primary decision metric.

Recall, F1, ROC-AUC and business cost should be considered together.

## Business Recommendations

### 1. Early Customer Retention

Develop personalized onboarding and loyalty campaigns for customers showing early disengagement.

### 2. High-ARPU Customer Monitoring

Prioritize valuable customers with elevated churn risk and provide:

- flexible plans
- personalized discounts
- retention offers

### 3. Re-engage Inactive Customers

Develop reactivation campaigns targeting inactive or suspended accounts through:

- SMS
- email
- calls
- personalized offers

### 4. Predictive Churn Monitoring

Integrate churn probability scores into CRM or business dashboards to allow retention teams to identify at-risk customers proactively.

### 5. Automated Risk Alerts

Trigger alerts when customer activity metrics fall below predefined thresholds.

## Deployment

A prototype churn prediction interface was developed using **Gradio**.

The application accepts customer profile and subscription information, applies the same preprocessing pipeline used during model training and returns churn-risk probabilities.

The deployment demonstrates how a machine learning model can be transformed from an analytical experiment into a business-facing decision-support tool.

## Business Value

The project demonstrates how customer data can support:

- churn prediction
- customer retention
- customer segmentation
- revenue protection
- proactive CRM strategies
- risk prioritization
- data-driven marketing decisions

## Limitations

Several limitations should be considered:

- The dataset represents a single time period.
- External factors such as competitor activity and service outages are unavailable.
- Churn represents a very small minority class.
- Geographic variables may act as proxies for unobserved regional factors.
- Feature importance does not establish causality.

## Future Improvements

Potential extensions include:

- threshold optimization
- cost-sensitive classification
- SHAP explainability
- probability calibration
- temporal churn modelling
- customer lifetime value integration
- retention ROI optimization

## Repository Structure

```text
bulgarian-telecom-churn-prediction/
│
├── README.md
│
├── notebooks/
│   └── telecom_churn_analysis.ipynb
│
├── data/
│   └── bulgarian_telecom_churn.csv
│
├── report/
│   └── Bulgarian_Telecom_Churn_Analysis_Report.pdf
│
└── images/
    ├── churn_distribution.png
    ├── model_comparison.png
    └── feature_importance.png