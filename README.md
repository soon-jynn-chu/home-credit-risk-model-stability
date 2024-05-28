# Home Credit - Credit Risk Model Stability (Kaggle Competition)
This repository stores the notebooks for my participation on this Kaggle competition: https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability. 

# Approach
## Feature Engineering
### Data Aggregation

For depth > 0 features, aggregation may be needed to condense the historical records associated with each `case_id` into a single feature. Thus, aggregation functions are defined based on their transformation groups:

- All - max
- P, A, D - + mean, var

### Feature Selection
For numerical columns, I use Pearson correlation to drop columns that are highly correlated, whereas Cramer's V for categorical columns.

## Models and Results
All models are trained using stratified group k-fold cross-validation, with `WEEK_NUM` as the group. One model is generated for each fold, and a voting classifier is used to combine the predictions with equal weights.

| Model | Public LB | Private LB |
|---|---|---|
| LGB | 0.580 | 0.500 |
| XGB | 0.586 | 0.509 |
| Cat | 0.579 | 0.509 |
| Hist | 0.585 | 0.501 |
| LGB + Cat | 0.590 | 0.516 |
| XGB + Hist | 0.588 | 0.509 |
| LGB + XGB + Cat | 0.591 | 0.517* |
| LGB + Cat + Hist | 0.593* | 0.516 |
| LGB + XGB + Hist | 0.588 | 0.509 | 
| XGB + Cat + Hist | 0.592 | 0.516 |
| LGB + XGB + Cat + Hist | 0.593* | 0.516 | 
| 0.75 (LGB + Cat) + 0.25 (XGB + Hist) | 0.592 | 0.517* |
| 0.4 Cat + 0.6 (LGB + XGB + Hist)| 0.592 | 0.517* |
| LGB + XGB + Cat + Hist -> SGD(*log_loss*) | 0.579 | 0.499 |
| LGB + XGB + Cat + Hist -> LGB | 0.592 | 0.513 |
| LGB + XGB + Cat + Hist -> XGB | 0.585 | 0.502 | 

- LGB - LightGBM
- XGB - XGBoost
- Cat - CatBoost
- Hist - sklearn's HistGradientBoosting