"""Customer Churn Prediction & Retention Analytics.

Dependencies: pandas, numpy, scikit-learn
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

RANDOM_STATE = 42
THRESHOLDS = (0.30, 0.40, 0.50, 0.60)

# Hypothetical 10% relative reduction in churn
CHURN_REDUCTION_SCENARIO = 0.10


# ============================================================
# 2. DATA LOADING AND CLEANING
# ============================================================

def load_data():
    """Load the IBM Telco dataset and clean TotalCharges."""

    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        candidates = [
            Path(__file__).resolve().parent / 'IBM_Customer_dataset.csv',
            Path.cwd() / 'IBM_Customer_dataset.csv',
            Path.cwd() / 'Customer Churn Prediction & Retention Analytics' / 'IBM_Customer_dataset.csv'
        ]

        csv_path = next(
            (path for path in candidates if path.is_file()),
            candidates[0]
        )

    if not csv_path.is_file():
        raise FileNotFoundError(
            f'Dataset not found: {csv_path}\n'
            'Place IBM_Customer_dataset.csv next to this script.'
        )

    df = pd.read_csv(csv_path)

    # Blank TotalCharges values belong to customers with tenure = 0
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'],
        errors='coerce'
    ).fillna(0)

    return df


# ============================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================

def churn_rates(df, column):
    """Calculate churn percentage for each category."""

    rates = (
        df['Churn']
        .eq('Yes')
        .groupby(df[column], observed=True)
        .mean() * 100
    )

    return rates.round(2)


def exploratory_analysis(df):
    """Summarize customer churn patterns."""

    print('\n=== EXPLORATORY DATA ANALYSIS ===')

    print(f'Customers: {len(df):,}')
    print(f'Overall churn: {df["Churn"].eq("Yes").mean():.2%}')

    # Churn by tenure range
    tenure_groups = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 72],
        labels=['0-12', '13-24', '25-48', '49-72'],
        include_lowest=True
    )

    tenure_rates = (
        df['Churn']
        .eq('Yes')
        .groupby(tenure_groups, observed=True)
        .mean() * 100
    )

    print('\nChurn by tenure range (%):')
    print(tenure_rates.round(2).to_string())

    # Churn by important customer characteristics
    columns = [
        'Contract',
        'InternetService',
        'SeniorCitizen',
        'TechSupport',
        'PaymentMethod'
    ]

    for column in columns:
        print(f'\nChurn by {column} (%):')
        print(churn_rates(df, column).to_string())

    # Monthly charges
    print('\nMean monthly charges by churn status ($):')
    print(
        df.groupby('Churn')['MonthlyCharges']
        .mean()
        .round(2)
        .to_string()
    )

    print('\nMean monthly charges by contract and churn status ($):')
    print(
        df.groupby(['Contract', 'Churn'])['MonthlyCharges']
        .mean()
        .round(2)
        .to_string()
    )

    # Fiber optic customers
    fiber = df[df['InternetService'].eq('Fiber optic')]

    print('\nFiber optic churn by contract (%):')
    print(churn_rates(fiber, 'Contract').to_string())

    # Technical support among month-to-month customers
    monthly = df[df['Contract'].eq('Month-to-month')]

    print('\nMonth-to-month churn by technical support (%):')
    print(churn_rates(monthly, 'TechSupport').to_string())


# ============================================================
# 4. FEATURE PREPROCESSING
# ============================================================

def prepare_features(train, other):
    """
    Fit encoding and scaling on training data only.
    Apply the same transformations to the other dataset.
    """

    categorical = train.select_dtypes(
        include=['object', 'string', 'category']
    ).columns

    numerical = train.select_dtypes(
        include='number'
    ).columns

    # Encode categorical features
    encoder = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False
    )

    train_encoded = encoder.fit_transform(train[categorical])
    other_encoded = encoder.transform(other[categorical])

    # Scale numerical features
    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(train[numerical])
    other_scaled = scaler.transform(other[numerical])

    # Combine encoded and scaled features
    train_final = np.hstack([
        train_encoded,
        train_scaled
    ])

    other_final = np.hstack([
        other_encoded,
        other_scaled
    ])

    # Preserve feature names for model interpretation
    feature_names = np.concatenate([
        encoder.get_feature_names_out(categorical),
        numerical.to_numpy()
    ])

    return train_final, other_final, feature_names


# ============================================================
# 5. MODEL EVALUATION HELPERS
# ============================================================

def model_metrics(actual, predicted):
    """Calculate classification performance metrics."""

    return {
        'Accuracy': accuracy_score(actual, predicted),

        'Precision': precision_score(
            actual, predicted,
            pos_label='Yes',
            zero_division=0
        ),

        'Recall': recall_score(
            actual, predicted,
            pos_label='Yes',
            zero_division=0
        ),

        'F1': f1_score(
            actual, predicted,
            pos_label='Yes',
            zero_division=0
        )
    }


def predict_at_threshold(model, features, threshold):
    """Convert predicted churn probabilities into Yes/No labels."""

    yes_column = list(model.classes_).index('Yes')

    probabilities = model.predict_proba(features)[:, yes_column]

    return np.where(
        probabilities >= threshold,
        'Yes',
        'No'
    )


def make_models():
    """Create the two classification models."""

    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000
        ),

        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=RANDOM_STATE
        )
    }


# ============================================================
# 6. MODEL TRAINING, VALIDATION, AND TESTING
# ============================================================

def evaluate_models(df):
    """Compare models using validation-selected thresholds."""

    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn']

    # Initial 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Split training data again for threshold validation
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    X_fit_ready, X_val_ready, _ = prepare_features(
        X_fit,
        X_val
    )

    print('\n=== VALIDATION: THRESHOLD SELECTION ===')

    selected_thresholds = {}

    # Evaluate candidate thresholds on validation data
    for name, model in make_models().items():

        model.fit(X_fit_ready, y_fit)

        rows = []

        for threshold in THRESHOLDS:

            predictions = predict_at_threshold(
                model,
                X_val_ready,
                threshold
            )

            metrics = model_metrics(
                y_val,
                predictions
            )

            rows.append({
                'Threshold': threshold,
                **metrics
            })

        results = pd.DataFrame(rows)

        # Choose the threshold with the highest validation F1
        best_row = results.loc[
            results['F1'].idxmax()
        ]

        selected_thresholds[name] = float(
            best_row['Threshold']
        )

        print(f'\n{name}:')

        print(
            results.to_string(
                index=False,
                float_format=lambda value: f'{value:.3f}'
            )
        )

        print(
            f'Selected threshold: '
            f'{selected_thresholds[name]:.2f}'
        )

    # Refit models on the full training set
    X_train_ready, X_test_ready, feature_names = prepare_features(
        X_train,
        X_test
    )

    print('\n=== TEST PERFORMANCE ===')

    for name, model in make_models().items():

        model.fit(X_train_ready, y_train)

        threshold = selected_thresholds[name]

        predictions = predict_at_threshold(
            model,
            X_test_ready,
            threshold
        )

        metrics = model_metrics(
            y_test,
            predictions
        )

        print(f'\n{name} (threshold {threshold:.2f}):')

        for metric, value in metrics.items():
            print(f'  {metric}: {value:.3f}')

        # Logistic Regression coefficients
        if name == 'Logistic Regression':

            coefficients = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': model.coef_[0]
            })

            print('\nLargest positive coefficients:')
            print(
                coefficients.nlargest(
                    10, 'Coefficient'
                ).to_string(index=False)
            )

            print('\nLargest negative coefficients:')
            print(
                coefficients.nsmallest(
                    10, 'Coefficient'
                ).to_string(index=False)
            )

        # Random Forest feature importance
        else:

            importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            })

            print('\nTop Random Forest feature importances:')
            print(
                importance.nlargest(
                    10, 'Importance'
                ).to_string(index=False)
            )

    print(
        '\nNote: Model comparisons are exploratory because '
        'the test set was examined during development.'
    )

    print(
        'Feature associations and importance scores '
        'do not establish causation.'
    )


# ============================================================
# 7. CUSTOMER RETENTION AND BUSINESS IMPACT
# ============================================================

def retention_analysis(df):
    """Analyze a high-churn segment and potential retention value."""

    # Target segment: month-to-month customers without support
    high_risk = df[
        (df['Contract'] == 'Month-to-month') &
        (df['TechSupport'] == 'No')
    ]

    churned_high_risk = high_risk[
        high_risk['Churn'] == 'Yes'
    ]

    total_churned = (df['Churn'] == 'Yes').sum()

    # Churn rates
    segment_churn_rate = (
        high_risk['Churn'] == 'Yes'
    ).mean()

    overall_churn_rate = (
        df['Churn'] == 'Yes'
    ).mean()

    relative_risk = (
        segment_churn_rate / overall_churn_rate
    )

    # Share of company churn
    churn_share = (
        len(churned_high_risk) / total_churned
    )

    # Hypothetical retention campaign
    potential_saved = (
        len(churned_high_risk) * CHURN_REDUCTION_SCENARIO
    )

    # Average charges of churned customers in this segment
    avg_monthly_charges = (
        churned_high_risk['MonthlyCharges'].mean()
    )

    # Estimate annualized gross revenue retained
    annual_revenue = (
        potential_saved * avg_monthly_charges * 12
    )

    print('\n=== RETENTION OPPORTUNITY ===')

    print(
        'Segment: Month-to-month customers '
        'without technical support'
    )

    print(
        f'Segment size: {len(high_risk):,} '
        f'({len(high_risk) / len(df):.2%} of customers)'
    )

    print(
        f'Segment churn rate: {segment_churn_rate:.2%}'
    )

    print(
        f'Overall churn rate: {overall_churn_rate:.2%}'
    )

    print(
        f'Relative churn rate: {relative_risk:.2f}x'
    )

    print(
        f'Churned customers in segment: '
        f'{len(churned_high_risk):,} / {total_churned:,}'
    )

    print(
        f'Share of all churn: {churn_share:.2%}'
    )

    print(
        f'Hypothetical churn reduction: '
        f'{CHURN_REDUCTION_SCENARIO:.0%}'
    )

    print(
        f'Potential retained customers: '
        f'{potential_saved:,.0f}'
    )

    print(
        f'Average monthly charges among churned customers: '
        f'${avg_monthly_charges:,.2f}'
    )

    print(
        f'Hypothetical annualized gross revenue retained: '
        f'${annual_revenue:,.2f}'
    )

    print('\nBusiness recommendation:')
    print(
        'Test proactive technical support and longer-contract '
        'incentives for this customer segment.'
    )

    print(
        'Revenue estimates assume 12 additional months '
        'at unchanged charges.'
    )

    print(
        'Campaign costs, discounts, and subsequent churn '
        'are not included.'
    )


# ============================================================
# 8. RUN PROJECT
# ============================================================

def main():
    """Run the complete customer churn analysis."""

    df = load_data()

    exploratory_analysis(df)

    evaluate_models(df)

    retention_analysis(df)


if __name__ == '__main__':
    main()