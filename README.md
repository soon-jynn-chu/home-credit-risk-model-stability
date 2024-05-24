# Home Credit - Credit Risk Model Stability (Kaggle Competition)
This repository stores the notebooks for my participation on this Kaggle competition: https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability. 

# Approach
## Feature Engineering
### Data Aggregation

For depth > 0 features, aggregation may be needed to condense the historical records associated with each `case_id` into a single feature.Thus, aggregation functions are defined based on their transformation groups:

- P, A, D - max, mean, var
- M, String/Boolean dtypes - mode
- others - max

### Feature Selection
For numerical columns, I use Pearson correlation to drop columns that are highly correlated, whereas Cramer's V for categorical columns.

## Models and Results
All models are trained using stratified group k-fold cross-validation, with `WEEK_NUM` as the group. One model is generated for each fold, and a voting classifier is used to combine the predictions with equal weights.

| Model | Public LB | Private LB |
|---|---|---|
| LGB | 0.580 |  |
| XGB | 0.586 |  |
| Cat | 0.579 |  |
| Hist | 0.585 |  |
| LGB + Cat | 0.590 |  |
| XGB + Hist | 0.588 |  |
| LGB + XGB + Cat | 0.591 | |
| LGB + Cat + Hist | **0.593** |  |
| LGB + XGB + Hist | 0.588 |  | 
| XGB + Cat + Hist | 0.592 |  |
| LGB + XGB + Cat + Hist | **0.593** |  | 
| LGB + XGB + Cat + Hist -> SGD(*log_loss*) | 0.579 |  |
| LGB + XGB + Cat + Hist -> LGB | 0.592 |  |
| LGB + XGB + Cat + Hist -> XGB |  |  | 

- LGB - LightGBM
- XGB - XGBoost
- Cat - CatBoost
- Hist - sklearn's HistGradientBoosting