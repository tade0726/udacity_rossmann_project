# Rossmann Store Prediction

## Summary

Base on Kaggle competetion on "Rossmann Store Sales"
- https://www.kaggle.com/c/rossmann-store-sales


## Analysis Report

See "report.pdf"

## Python env

- requirements.txt

- libs：
  - Pandas
  - Xgboost


## Notebooks (core experiments)

- xgb-report.ipynb
- lgb-report.ipynb


## running time

| model    | Validation Score | Private Score | Public Score | Train Time          |
| -------- | ---------------- | ------------- | ------------ | ------------------- |
| Lightgbm | 0.108946         | **0.11596**   | 0.11043      | 1724.3s (CPU)       |
| Xgboost  | 0.114742         | **0.11503**   | 0.11053      | 1598.6s (GPU)       |

## Results

![屏幕截图](screenshot.png)
