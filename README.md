[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

# Poverty Prediction Challenge

## Goal of the Competition

Accurate poverty measurement is essential for directing global development efforts and informing evidence-based policies for poverty reduction and equity enhancement, yet many countries lack recent data due to the high costs and complexity of collecting comparable comprehensive household expenditure surveys.

This challenge simulated a common real-world scenario faced by economists, who are tasked with producing up-to-date poverty measurements and additional welfare indicators, even in cases where fully detailed recent information on household expenditure is unavailable. The goal was to develop survey-to-survey imputation models that predicted both poverty rates and per capita household consumption from anonymized historical survey data.

Performance was evaluated according to a weighted average of the household-level prediction error and the distribution-level prediction error:

* 90% of the weighted average was computed as the weighted mean absolute percentage error (w-MAPE) between predicted poverty rates and the actual rates at 19 specific consumption thresholds ranging from $3.17 to $27.37
* 10% consisted of a mean absolute percentage error between predicted household-level per capita consumption and actual per capita consumption (measured in 2017 USD PPP)

See the World Bank's results page of this competition [here](https://www.worldbank.org/en/topic/measuringpoverty/brief/poverty-prediction-challenge).

## What's in this Repository

This repository contains code from winning competitors in the Poverty Prediction DrivenData challenge. Code for all winning solutions are open source under the MIT License.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

| Place | Team or User | Public Score | Private Score | Summary of Model                                                                                                                                                                                                                                                     |
| ----- | ------------ | ------------ | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | dwivedy045   | 5.4466       | 5.7545        | LightGBM pipeline with grouped cross-validation (GroupKFold by survey), categorical handling, and quantile-mapped inference calibration for household consumption and poverty-rate prediction.                                                                         |
| 2     | Khartoum     | 12.0761      | 7.7052        | LightGBM with leave-one-survey-out cross-validation, anti-leakage survey-specific mean ratio features, top-75% feature selection by importance, and weighted quantile calibration for poverty rate distribution matching.                                            |
| 3     | selman       | 8.1626       | 8.2382        | 4-model gradient boosting ensemble (LightGBM ×2, XGBoost, CatBoost) with per-capita feature engineering, test-time augmentation over utility expense perturbations, P40-focused survey matching, and per-survey calibration via differential evolution optimization. |


# Competition Leaderboard

| Rank | Participant | Private wS-wMAPE | Private wMAPE |
|------|-------------|---------------:|-------------:|
| 🥇 1 | dwivedy045 🏆 | 5.7545 | 2.6177 |
| 🥈 2 | shyboy | 7.1904 | 3.9436 |
| 🥉 3 | Khartoum 🏆 | 7.7052 | 4.5917 |
| 4 | Daiz | 7.7098 | 4.5757 |
| 5 | ganontha | 8.1882 | 5.1818 |
| 6 | selman 🏆 | 8.2382 | 4.9622 |
| 7 | javidkamal15 | 8.2692 | 4.9556 |
| 8 | pavnoval | 8.2782 | 5.0912 |
| 9 | chidubem | 8.4354 | 4.8644 |
| 10 | HeroVoltsy | 8.6809 | 4.7584 |
| 11 | jekiwantaufik | 8.8876 | 5.0395 |
| 12 | charles625 | 9.1140 | 6.2603 |
| 13 | oc166 | 9.2707 | 5.4388 |
| 14 | kasrsf | 9.2846 | 6.1969 |
| 15 | govinuts | 9.5113 | 5.4497 |
| 16 | jvillines | 10.1848 | 6.9231 |
| 17 | mgoulart | 10.2600 | 7.0326 |
| 18 | BoundaryLab | 10.3068 | 6.9856 |
| 19 | vbhv3773 | 10.4797 | 6.9985 |
| 20 | juliop_ | 10.4991 | 7.5255 |
| 21 | eferraz | 10.5644 | 7.6719 |
| 22 | rokket9000 | 10.7479 | 7.4929 |
| 23 | kefthyme | 10.7736 | 7.7727 |
| 24 | Nicole_Tiokhin | 10.8779 | 7.5150 |
| 25 | m39lee | 10.8856 | 7.5195 |
| 26 | ananya_s | 10.9986 | 7.5196 |
| 27 | hafidh24 | 11.0050 | 7.6435 |
| 28 | tiesp | 11.0223 | 7.5210 |
| 29 | 太上老君 | 11.0546 | 8.0528 |
| 30 | wilfred3235 | 11.2403 | 8.2777 |
| 31 | Lucian41 | 11.2548 | 7.9212 |
| 32 | rudecia | 11.4028 | 7.9651 |
| 33 | _0037 | 11.5254 | 7.5210 |
| 34 | TheIsoLab | 11.6938 | 8.0978 |
| 35 | lazarosgogos | 11.8047 | 8.8482 |
| 36 | azmiy | 11.8370 | 8.8265 |
| 37 | mithos | 11.9437 | 8.9990 |
| 38 | bbeum | 12.0760 | 9.1018 |
| 39 | Ayush_killer | 12.1643 | 8.8979 |
| 40 | pecama4267 | 12.1961 | 8.8370 |
| 41 | dmitrysarov | 12.4571 | 9.2217 |
| 42 | NxGTR | 12.4870 | 9.4115 |
| 43 | YESSEth *(team)* | 12.7809 | 9.6397 |
| 44 | Yuxinnn | 12.9049 | 9.4346 |
| 45 | Merch1000 | 13.0533 | 10.2115 |
| 46 | NataliaTAmv | 13.0591 | 9.6643 |
| 47 | EconAI *(team)* | 13.1496 | 9.8068 |
| 48 | limzero | 13.1733 | 10.2743 |
| 49 | 53un | 13.2333 | 9.8654 |
| 50 | No Comeback Crew *(team)* | 13.3083 | 9.7532 |
| 51 | barata.lade | 13.3276 | 10.1708 |
| 52 | phucdkbk | 13.3573 | 10.4526 |
| 53 | isidorat | 13.3869 | 9.3881 |
| 54 | PaulMcBride | 13.4013 | 10.5148 |
| 55 | Kosmas7 | 13.4500 | 10.2108 |
| 56 | jackson5 | 13.4801 | 10.5671 |
| 57 | Smasko | 13.5287 | 10.2108 |
| 58 | Stealth *(team)* | 13.5598 | 10.4494 |
| 59 | sachin__09 | 13.5902 | 10.4205 |
| 60 | Chinchpokli | 13.7670 | 10.4223 |
| 61 | Vincent Schuler *(team)* | 13.7782 | 10.9191 |
| 62 | sumatorikki | 13.7953 | 10.9119 |
| 63 | qgoens | 13.8934 | 10.5628 |
| 64 | kar_len | 13.9865 | 10.8769 |
| 65 | KallK | 14.0341 | 10.8172 |
| 66 | ssimp10032 | 14.0453 | 10.5108 |
| 67 | emmettsexton | 14.0570 | 9.8790 |
| 68 | ocitalis | 14.0791 | 11.0760 |
| 69 | structAS | 14.1863 | 11.0981 |
| 70 | deactivated-e9b8d89 | 14.1911 | 11.3454 |
| 71 | dami | 14.3137 | 10.8556 |
| 72 | meenalmurugesan | 14.4510 | 11.1949 |
| 73 | yummyfe | 14.4712 | 10.2904 |
| 74 | shenjianmantou *(team)* | 14.4881 | 11.5891 |
| 75 | Athirajan | 14.5131 | 11.1949 |
| 76 | DataForGood *(team)* | 14.5520 | 11.2561 |
| 77 | koni-team *(team)* | 14.5538 | 11.2346 |
| 78 | Surgis12 | 14.5605 | 11.5682 |
| 79 | Helio | 14.6824 | 11.3642 |
| 80 | shimaa | 14.7513 | 11.2211 |
| 81 | jakey | 14.7643 | 11.4462 |
| 82 | mugesh9085 | 14.7742 | 11.4561 |
| 83 | jil300 | 14.7742 | 11.4561 |
| 84 | kiochan | 14.7742 | 11.4561 |
| 85 | Himfs | 14.7742 | 11.4561 |
| 86 | sheldon9789 | 14.7785 | 11.4603 |
| 87 | mashy | 14.8077 | 11.4896 |
| 88 | tika | 14.8096 | 11.4915 |
| 89 | Watson9789 | 14.8106 | 11.4924 |
| 90 | ragnar9085 | 14.8176 | 11.4995 |
| 91 | lothbrok | 14.8277 | 11.5095 |
| 92 | Buhari2612 | 14.8367 | 11.5185 |
| 93 | CJ343 | 14.8440 | 11.9419 |
| 94 | Sherlock281 | 14.8701 | 11.5519 |
| 95 | Rmugesh | 14.8978 | 11.5797 |
| 96 | mugesh2000 | 14.8981 | 11.5799 |
| 97 | AmineSamoudi | 14.9007 | 12.0550 |
| 98 | magist6 | 14.9299 | 11.9307 |
| 99 | MAT-U | 14.9796 | 12.1361 |
| 100 | dishantsaini55 | 15.0494 | 12.1644 |

> 🏆 = prize winner &nbsp;|&nbsp; *(team)* = multi-person team entry

