# Customer Churn Prediction & Retention Analytics

## Project Overview

This project analyzes customer churn using the IBM Telco Customer Churn dataset, which contains 7,043 customers. The goal was to identify factors associated with customers leaving, develop machine learning models to predict churn, and identify opportunities to improve customer retention.

I used Python to clean and explore the data, trained Logistic Regression and Random Forest models, and analyzed a high-risk customer segment to estimate the potential financial impact of a retention campaign.

## Tools & Technologies

- **Python:** pandas, NumPy, scikit-learn
- **Data Analysis:** Data cleaning, exploratory data analysis, customer segmentation
- **Machine Learning:** Logistic Regression, Random Forest, one-hot encoding, feature scaling
- **Model Evaluation:** Accuracy, precision, recall, F1-score, validation-based threshold selection

## Data Cleaning & Exploratory Analysis

The dataset contained 11 missing values in `TotalCharges`, all associated with customers who had zero months of tenure. I converted this column to numeric and replaced those missing values with zero.

I then analyzed customer churn across tenure, contract type, internet service, technical support, payment method, and monthly charges.

### Key Findings

| Customer Segment | Churn Rate |
|---|---:|
| Overall customer base | 26.54% |
| Customers with 0–12 months tenure | 47.44% |
| Month-to-month contracts | 42.71% |
| Two-year contracts | 2.83% |
| Fiber optic customers | 41.89% |
| Customers without technical support | 41.64% |
| Electronic check customers | 45.29% |

Customers with shorter tenure and month-to-month contracts had substantially higher churn rates. Technical support and payment method were also associated with meaningful differences in churn.

These relationships are observational and do not establish causation.

## Machine Learning Models

I developed two classification models to predict whether customers would churn:

1. **Logistic Regression:** A simpler, interpretable model that estimates churn probabilities.
2. **Random Forest:** An ensemble model that combines multiple decision trees.

The dataset was split into training and testing sets using an 80/20 split. I then created a validation set from the training data to compare prediction thresholds of 30%, 40%, 50%, and 60%.

Both models achieved their highest validation F1-score among the tested thresholds at **40%**.

### Model Performance

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 77.7% | 77.6% |
| Precision | 56.8% | 56.9% |
| Recall | 66.8% | 65.0% |
| F1-score | **0.614** | 0.607 |

Logistic Regression achieved slightly better test recall and F1-score, although the difference between models was small. Its interpretability also makes it useful for explaining churn-related patterns.

The test set was examined during model development, so these comparisons should be considered exploratory rather than an entirely untouched final evaluation.

## Model Interpretation

Random Forest identified the following features as particularly important:

- Customer tenure
- Total charges
- Monthly charges
- Month-to-month contracts
- Online security
- Fiber optic internet service
- Electronic check payments
- Technical support

Logistic Regression also highlighted contract type, tenure, and fiber optic service.

These findings were broadly consistent with the exploratory analysis. However, feature importance and model coefficients should not be interpreted as proof that a feature causes churn.

## Customer Retention Analysis

I identified a high-risk customer segment consisting of customers with **month-to-month contracts and no technical support**.

| Metric | Result |
|---|---:|
| Customers in segment | 2,680 |
| Percentage of total customers | 38.05% |
| Segment churn rate | 50.37% |
| Overall churn rate | 26.54% |
| Relative churn rate | 1.90x |
| Churned customers in segment | 1,350 |
| Share of all churned customers | **72.23%** |

Although this segment represented only 38.05% of customers, it accounted for 72.23% of all observed churn.

This makes it a strong candidate for targeted retention efforts.

### Potential Revenue Impact

I modeled a hypothetical scenario in which a retention campaign reduces churn within this segment by 10% relative to its observed level.

| Metric | Estimate |
|---|---:|
| Hypothetical churn reduction | 10% |
| Potential customers retained | 135 |
| Average monthly charges among churned customers | $75.84 |
| Estimated annualized gross revenue retained | **$122,858.70** |

This estimate assumes retained customers continue paying approximately the same monthly charges for another 12 months. It does not account for campaign expenses, discounts, or subsequent churn, and it is not a prediction of actual campaign performance.

## Business Recommendations

**1. Prioritize high-risk customers.**

Focus initial retention efforts on month-to-month customers without technical support. This group accounts for a large share of observed customer churn.

**2. Test proactive technical support.**

Offer targeted technical assistance or support trials to determine whether additional support improves retention.

**3. Test longer-contract incentives.**

Evaluate discounts or other incentives for customers willing to switch from month-to-month contracts to longer agreements.

**4. Measure campaign effectiveness.**

Use a controlled experiment to compare churn rates between customers receiving retention offers and a similar group that does not. Track customer retention, campaign costs, and net financial impact before expanding the program.

## How to Run the Project

1. Download or clone this repository.
2. Install Python and the required libraries:

   ```bash
   pip install pandas numpy scikit-learn
   ```

3. Keep `customer_churn_cleaned.py` and `IBM_Customer_dataset.csv` in the same directory.
4. Run:

   ```bash
   python customer_churn_cleaned.py
   ```

The script prints exploratory analysis results, model validation and test metrics, feature importance, and retention opportunity estimates.

## Dataset

IBM Telco Customer Churn dataset, containing 7,043 customer records.

## Project Takeaway

This project demonstrates how exploratory data analysis and machine learning can be combined to identify customer retention opportunities. Although the two predictive models performed similarly, the customer segmentation analysis revealed a concentrated area of churn that could be prioritized for business testing.

The main takeaway is that predictive modeling becomes more useful when its results are connected to actionable business decisions and measurable financial outcomes.
