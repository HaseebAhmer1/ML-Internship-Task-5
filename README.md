# 🏥 Medical Insurance Charges Prediction

## 🎯 Objective
Build and evaluate regression models that predict individual medical insurance charges from demographic and health-related attributes (age, BMI, number of children, sex, smoking status, and region), while validating the statistical assumptions behind the models and comparing multiple regression techniques to identify the best-performing approach.

## 📊 Dataset Information
- **Source file:** `insurance.csv`
- **Target variable:** `charges` — the individual medical costs billed by health insurance
- **Numerical features:** `age`, `bmi`, `children`
- **Categorical features:** `sex`, `smoker`, `region`
- **Preprocessing checks performed:** missing value inspection, duplicate detection and removal, and descriptive statistics for both numerical and categorical columns

## ✅ Tasks & Outcomes

**1. Exploratory Data Analysis (EDA)**
- Inspected data shape, types, null values, and duplicates
- Visualized the distribution of `charges` (histogram, KDE, boxplot) and each numerical feature
- Explored relationships between `age`/`bmi` and `charges`, segmented by smoking status
- Compared charge distributions across smokers vs. non-smokers
- Built a correlation heatmap across numerical features and the target

**2. Data Preparation**
- Split data into training (80%) and testing (20%) sets
- Built a `ColumnTransformer` preprocessing pipeline: `StandardScaler` for numerical features and `OneHotEncoder` for categorical features

**3. Baseline Modeling**
- Trained a **Linear Regression** model and evaluated it using MAE, RMSE, and R²
- Visualized actual vs. predicted charges

**4. Statistical Assumption Diagnostics (via `statsmodels` OLS)**
- Fitted an OLS model to obtain interpretable coefficients and a full statistical summary
- **Homoscedasticity:** Breusch-Pagan test on residuals
- **Normality of residuals:** Q-Q plot and Shapiro-Wilk test
- **Independence of errors:** Durbin-Watson statistic
- **Multicollinearity:** Variance Inflation Factor (VIF) for each predictor
- **Influential observations:** Cook's Distance analysis against a standard threshold
- Residual distribution plots to visually confirm assumption checks

**5. Regularized & Non-Linear Models**
- Trained and evaluated **Ridge**, **Lasso**, and **ElasticNet** regression
- Trained a **Polynomial Regression** model (degree 2) on numerical features via a pipeline
- Compared all models on MAE, RMSE, and R²

**6. Cross-Validation**
- Applied 5-fold `KFold` cross-validation across all models
- Compared cross-validated MAE, RMSE, R², and R² standard deviation
- Visualized CV performance with bar charts

**7. Hyperparameter Tuning**
- Used `GridSearchCV` (scored on RMSE) to tune:
  - Ridge's `alpha`
  - Lasso's `alpha`
  - ElasticNet's `alpha` and `l1_ratio`
- Refit tuned models and compared them against the baseline and polynomial models

**8. Final Model Selection**
- Compared all final models (Linear, Tuned Ridge, Tuned Lasso, Tuned ElasticNet, Polynomial) side by side
- Selected **Linear Regression** as the final model
- Performed final residual analysis and an actual-vs-predicted plot to confirm model fit

## 🛠️ Key Skills Demonstrated
- Exploratory data analysis and data visualization (`matplotlib`, `seaborn`)
- Data preprocessing pipelines with `scikit-learn` (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `Pipeline`)
- Regression modeling: Linear, Ridge, Lasso, ElasticNet, and Polynomial Regression
- Statistical inference and regression diagnostics with `statsmodels` (OLS summary, Breusch-Pagan, Durbin-Watson, VIF, Cook's Distance, Shapiro-Wilk, Q-Q plots)
- Model evaluation using MAE, RMSE, and R²
- Cross-validation (`KFold`, `cross_validate`) for robust performance estimation
- Hyperparameter tuning with `GridSearchCV`
- Model comparison and selection based on both single-split and cross-validated performance

## 💡 Key Learnings
- Feature relationships (e.g., smoking status) can dominate a target variable's variance, which becomes clear only through combined visual and statistical exploration.
- A good R² alone doesn't guarantee a valid model — checking OLS assumptions (heteroscedasticity, non-normal residuals, multicollinearity, influential points) is essential to trust the coefficients and predictions.
- Regularization (Ridge/Lasso/ElasticNet) doesn't always outperform plain linear regression; its benefit depends on the degree of multicollinearity and overfitting present in the data.
- Cross-validation gives a more reliable estimate of model performance than a single train/test split, especially when comparing multiple candidate models.
- Hyperparameter tuning must be paired with cross-validation to avoid overfitting to a particular test split.
- Simpler models (like Linear Regression) can outperform more complex or regularized alternatives when the underlying relationships in the data are largely linear.
