[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

# Poverty Prediction Challenge

## Goal of the Competition

Accurate poverty measurement is essential for directing global development efforts and informing evidence-based policies for poverty reduction and equity enhancement, yet many countries lack recent data due to the high costs and complexity of collecting comparable comprehensive household expenditure surveys.

This challenge simulated a common real-world scenario faced by economists, who are tasked with producing up-to-date poverty measurements and additional welfare indicators, even in cases where fully detailed recent information on household expenditure is unavailable. The goal was to develop survey-to-survey imputation models that predicted both poverty rates and per capita household consumption from anonymized historical survey data.

Performance was evaluated according to a weighted average of the household-level prediction error and the distribution-level prediction error:

* 90% of the weighted average was computed as the weighted mean absolute percentage error (w-MAPE) between predicted poverty rates and the actual rates at 19 specific consumption thresholds ranging from $3.17 to $27.37
* 10% consisted of a mean absolute percentage error between predicted household-level per capita consumption and actual per capita consumption (measured in 2017 USD PPP)

## What's in this Repository

This repository contains code from winning competitors in the Poverty Prediction DrivenData challenge. Code for all winning solutions are open source under the MIT License.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

| Place | Team or User | Public Score | Private Score | Summary of Model                                                                                                                                                                                                                                                     |
| ----- | ------------ | ------------ | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | dwivedy045   | 5.4466       | 5.7545        | LightGBM pipeline with grouped cross-validation (GroupKFold by survey), categorical handling, and quantile-mapped inference calibration for household consumption and poverty-rate prediction.                                                                         |
| 2     | Khartoum     | 12.0761      | 7.7052        | LightGBM with leave-one-survey-out cross-validation, anti-leakage survey-specific mean ratio features, top-75% feature selection by importance, and weighted quantile calibration for poverty rate distribution matching.                                            |
| 3     | selman       | 8.1626       | 8.2382        | 4-model gradient boosting ensemble (LightGBM ×2, XGBoost, CatBoost) with per-capita feature engineering, test-time augmentation over utility expense perturbations, P40-focused survey matching, and per-survey calibration via differential evolution optimization. |
