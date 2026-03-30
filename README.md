# Diamond Price Prediction — Linear Regression

A supervised machine learning project to predict diamond prices using physical and quality attributes, covering the full pipeline from EDA to Streamlit deployment.

---

## Problem Statement

Predict the price of a diamond (in USD) based on its physical and quality attributes. This is a regression problem.

---

## Dataset

53,940 records with 10 features — 7 numerical, 3 ordinal categorical.

| Feature | Type | Notes |
|---|---|---|
| carat | Continuous | Weight of the diamond |
| cut | Ordinal | Fair < Good < Very Good < Premium < Ideal |
| color | Ordinal | J (worst) to D (best) |
| clarity | Ordinal | I1 (worst) to IF (best) |
| depth, table | Continuous | Proportional geometry measurements |
| x, y, z | Continuous | Physical dimensions in mm |
| price | Continuous | Target variable (USD) |

---

## Project Structure

```
diamond-price-prediction/
├── lr_model_diamond_dataset.ipynb   <- Full pipeline notebook
├── app.py                           <- Streamlit deployment app
├── lr_diamond_model.pkl             <- Saved model (includes scaler and encoder)
├── diamonds.csv                     <- Dataset
└── README.md
```

---

## Process Overview

**1. Data Loading and Exploration**
- Inspected shape, dtypes, and descriptive statistics
- No missing values found across all 53,940 records

**2. Data Cleaning**
- Removed 35 rows where x, y, or z dimensions were zero (invalid diamonds)
- Removed 145 duplicate rows — final clean dataset: 53,775 rows

**3. Exploratory Data Analysis**
- Histogram of `price` and `carat` — both are right-skewed
- Count plots for cut, color, clarity
- Correlation heatmap — carat, x, y, z, volume are strongly correlated with price
- Scatter plot of carat vs price — strong positive relationship
- Box plots of price by cut, color, clarity — revealed counterintuitive patterns (see Key Insights)
- Pair plot across size-related and target features

**4. Feature Engineering**
- `volume = x * y * z` — consolidates dimension features into one
- `log_carat = log1p(carat)` — linearises the carat-price relationship

**5. Ordinal Encoding**
- Used `OrdinalEncoder` inside a `ColumnTransformer` with explicit category orders for cut, color, and clarity
- `remainder="passthrough"` retained all other columns

**6. Log Transformation of Target**
- Applied `log1p(price)` to normalise the right-skewed price distribution
- Satisfies the linear regression assumption of normally distributed residuals
- Predictions are reversed with `expm1()` to recover dollar values

**7. Train-Test Split**
- 80% train / 20% test with `random_state=42`
- Features: cut, color, clarity, carat, depth, table, x, y, z, volume, log_carat
- Target: `log_price`

**8. Feature Scaling**
- `StandardScaler` fit on training set only, applied to both train and test
- Prevents data leakage and ensures features contribute equally

**9. Model Training**
- `LinearRegression` from scikit-learn trained on scaled training data

**10. Evaluation**
- Metrics computed on both train and test sets
- Residual histogram plotted to verify normality
- Predicted vs actual scatter plot produced

**11. Model Saving**
- `sc` (StandardScaler) and `ct` (ColumnTransformer) attached to the model object before saving
- Entire pipeline saved as a single file: `lr_diamond_model.pkl`

---

## Key Insights

- **Carat dominates price** — strongest predictor; x, y, z, and volume are highly correlated with it
- **Cut quality does not straightforwardly drive price** — Fair cut diamonds sometimes show higher median prices than Ideal due to larger carat weights (confounding variable)
- **Same pattern in color and clarity** — lower grades can appear more expensive because of carat size
- **Log transformation on price is necessary** — raw price is heavily right-skewed; transformation significantly improves residual normality
- **Depth and table have weak price influence** — low correlation compared to size-based features
- **Volume reduces multicollinearity** — consolidates x, y, z into one meaningful feature

---

## Model Performance

| Metric | Train | Test |
|---|---|---|
| R-squared | 0.9813 | 0.9802 |
| RMSE (log scale) | 0.1387 | 0.1421 |
| MAE (log scale) | 0.1076 | 0.1091 |
| MSE (log scale) | 0.0192 | 0.0202 |

R-squared of 0.98 on both sets — model explains 98% of the variance in log price with no significant overfitting.

---

## Deployment

Built with Streamlit. The app loads `lr_diamond_model.pkl`, reconstructs `ct` and `sc` from it, accepts user inputs, applies the full preprocessing pipeline, and displays the predicted price in USD.

Run locally:

```bash
pip install streamlit
streamlit run app.py
```

For public hosting, connect the repository to Streamlit Cloud and set `app.py` as the entry point.

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
streamlit
```

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib streamlit
```

---

## How to Run

```bash
# Notebook
jupyter notebook lr_model_diamond_dataset.ipynb

# Streamlit app
streamlit run app.py
```

Ensure `diamonds.csv` and `lr_diamond_model.pkl` are in the same directory as their respective files.
