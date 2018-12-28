# Rossmann Store Prediction

## 介绍

本次项目是利用 Kaggle 比赛 Rossmann Store Sales 进行实验
- https://www.kaggle.com/c/rossmann-store-sales


## 分析报告
详细的分析报告参看"report.pdf"

## Python env

- 见本项目下的 requirements.txt

- 主要是使用的框架：
  - Pandas
  - Xgboost


## 项目代码

- xgb-report.ipynb
- lgb-report.ipynb


## 运行时间

| model    | Validation Score | Private Score | Public Score | Train Time          |
| -------- | ---------------- | ------------- | ------------ | ------------------- |
| Lightgbm | 0.108946         | **0.11596**   | 0.11043      | 1724.3s (CPU)       |
| Xgboost  | 0.114742         | **0.11503**   | 0.11053      | 1598.6s (GPU)       |

## 运行截图

![屏幕截图](screenshot.png)
