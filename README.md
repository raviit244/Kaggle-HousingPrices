# House Prices Prediction – Advanced Regression Techniques (Kaggle)

An end-to-end regression project on Kaggle’s **House Prices: Advanced Regression Techniques** dataset, focused on **EDA-driven feature engineering**, **robust preprocessing**, and **ensemble regression models** for high-dimensional tabular data.

---

## Objective
Predict residential house sale prices while demonstrating a disciplined regression workflow on noisy, mixed-type real-world data.

---

## Exploratory Data Analysis (EDA)
EDA was used to uncover structure, outliers, and feature–target relationships that guided preprocessing and feature engineering.

- **Target analysis**
  - Distribution and right-skew of `SalePrice`
  - Log transformation applied to stabilise variance
- **Scatterplots (used heavily)**
  - Visualised `SalePrice` vs. key continuous predictors (e.g., living area, basement area, garage area, overall quality proxies)
  - Used these plots extensively to **detect and remove outliers / influential points** that would otherwise distort regression fits
- **Boxplots**
  - Compared `SalePrice` distributions across categorical/ordinal features (e.g., quality/condition categories)
  - Helped quantify the impact of categories on price, and supported decisions on **whether a feature was worth keeping vs. dropping**
- **Correlation matrix (visualisation)**
  - Identified strong numeric relationships and redundancy among size/quality-related features
  - Helped prioritise informative predictors and avoid unnecessary duplicates

---

## Data Cleaning & Preprocessing

### Missing Values (Semantics-Driven Imputation)
Missingness was handled based on what it *likely meant* in context (not blindly imputed).

- **Structural absence encoded explicitly**
  - When missing values plausibly implied the feature **does not exist** for that house (e.g., no basement, no garage, no fireplace), missing entries were imputed using meaningful “absence” values:
    - Categorical: `None` / `No<Feature>`
    - Numeric: `0` where “absence” logically corresponds to zero quantity (e.g., area/square-footage style fields)
  - Example rationale: if many rows are missing a “basement square feet” field and the dataset indicates a large fraction of homes have **no basement**, replacing missing values with `0` is more faithful than using a median imputation
- **Standard imputations**
  - Numeric features imputed with medians where absence semantics did not apply
  - Categorical features imputed with mode or domain-consistent defaults

### Transformations
- Log-transform applied to skewed numeric predictors where appropriate
- Log-transform applied to the target (`SalePrice`)
- Rare categorical levels consolidated to reduce noise

---

## Feature Engineering
Domain-informed features were constructed to capture size, age, usability, and overall property quality beyond raw columns.

- **Total square footage**
  - Combined basement, first-floor, and second-floor areas into a single “total area” style feature
- **House age features**
  - Age of the house at sale
  - Time since last renovation
- **Garage features**
  - Garage age and capacity/area style indicators
- **Bathroom features**
  - Total bathrooms computed from full + half baths
- **Porch & outdoor space**
  - Aggregated porch/deck square footage into a single outdoor-living feature
- **Quality interactions**
  - Combined indicators leveraging `OverallQual` / related quality or condition signals

---

## Preprocessing Pipeline
A unified preprocessing pipeline was built to make modeling consistent and leakage-safe across experiments.

- `ColumnTransformer` used to apply transformations by feature type
- **One-hot encoding** for nominal categorical variables
- **Ordinal encoding** for ordinal categorical variables (where categories have a natural order)
- Scaling applied where required for linear models
- A single pipeline design allowed fair comparisons across models and ensembles

---

## Modeling & Evaluation

All models were evaluated using:
- Train/validation split
- K-fold cross-validation
- **RMSE** as the primary evaluation metric
- Pipelines combining preprocessing and model fitting

### Models Used
- Linear Regression (baseline)
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting-style ensemble models

### Ensemble Methods
- **Voting Regressor**
  - Combined predictions from multiple strong base regressors
- **Stacking Regressor**
  - Meta-model trained on out-of-fold predictions
  - Delivered the strongest overall performance among approaches tested

---

## Key Findings
- Log-transforming `SalePrice` improved stability and regression fit quality
- Scatterplot-driven outlier removal noticeably improved model behaviour
- Semantics-driven imputation (encoding “absence” correctly) was crucial on this dataset
- Feature engineering contributed more than model choice alone
- Ensemble methods (voting & stacking) consistently outperformed individual models

---

## Takeaways
This project reinforced the importance of:
- Visual EDA (scatterplots, boxplots, correlation matrices) for feature decisions and outlier handling
- Semantics-aware missing value treatment in messy real-world datasets
- Feature engineering for tabular regression performance
- Ensemble learning + rigorous evaluation for reliable improvements
