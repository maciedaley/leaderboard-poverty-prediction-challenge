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

| Rank | Participant | Private WB-WMAPE | Public WMAPE |
|------|-------------|---------------:|-------------:|
| 🥇 1 | dwivedy045 🏆 | 5.7545 | 2.6627 |
| 🥈 2 | shyboy | 7.1904 | 1.4839 |
| 🥉 3 | Khartoum 🏆 | 7.7052 | 9.2999 |
| 4 | Daiz | 7.7098 | 5.8120 |
| 5 | ganontha | 8.1882 | 3.9346 |
| 6 | selman 🏆 | 8.2382 | 5.3635 |
| 7 | javidkamal15 | 8.2692 | 5.3851 |
| 8 | pavnoval | 8.2782 | 14.2759 |
| 9 | chidubem | 8.4354 | 5.3733 |
| 10 | HeroVoltsy | 8.6809 | 5.9140 |
| 11 | jekiwantaufik | 8.8876 | 5.1446 |
| 12 | charles625 | 9.1140 | 2.4928 |
| 13 | oc166 | 9.2707 | 4.6941 |
| 14 | kasrsf | 9.2846 | 6.6334 |
| 15 | govinuts | 9.5113 | 4.4663 |
| 16 | jvillines | 10.1848 | 4.1586 |
| 17 | mgoulart | 10.2600 | 3.0391 |
| 18 | BoundaryLab | 10.3068 | 3.2495 |
| 19 | vbhv3773 | 10.4797 | 3.4542 |
| 20 | juliop_ | 10.4991 | 2.0848 |
| 21 | eferraz | 10.5644 | 7.5535 |
| 22 | rokket9000 | 10.7479 | 4.2347 |
| 23 | kefthyme | 10.7736 | 6.6003 |
| 24 | Nicole_Tiokhin | 10.8779 | 2.1149 |
| 25 | m39lee | 10.8856 | 2.0824 |
| 26 | ananya_s | 10.9986 | 2.0815 |
| 27 | hafidh24 | 11.0050 | 26.3076 |
| 28 | tiesp | 11.0223 | 2.0799 |
| 29 | 太上老君 | 11.0546 | 2.7951 |
| 30 | wilfred3235 | 11.2403 | 8.4111 |
| 31 | Lucian41 | 11.2548 | 2.3295 |
| 32 | rudecia | 11.4028 | 5.4112 |
| 33 | _0037 | 11.5254 | 2.0799 |
| 34 | TheIsoLab | 11.6938 | 2.4817 |
| 35 | lazarosgogos | 11.8047 | 0.6371 |
| 36 | azmiy | 11.8370 | 4.7107 |
| 37 | mithos | 11.9437 | 2.8631 |
| 38 | bbeum | 12.0760 | 6.8865 |
| 39 | Ayush_killer | 12.1643 | 5.8538 |
| 40 | pecama4267 | 12.1961 | 1.1868 |
| 41 | dmitrysarov | 12.4571 | 1.6634 |
| 42 | NxGTR | 12.4870 | 1.9932 |
| 43 | YESSEth *(team)* | 12.7809 | 3.6343 |
| 44 | Yuxinnn | 12.9049 | 2.5473 |
| 45 | Merch1000 | 13.0533 | 5.5393 |
| 46 | NataliaTAmv | 13.0591 | 11.1515 |
| 47 | EconAI *(team)* | 13.1496 | 1.3372 |
| 48 | limzero | 13.1733 | 1.3372 |
| 49 | 53un | 13.2333 | 0.4949 |
| 50 | No Comeback Crew *(team)* | 13.3083 | 1.2109 |
| 51 | barata.lade | 13.3276 | 1.3151 |
| 52 | phucdkbk | 13.3573 | 9.0195 |
| 53 | isidorat | 13.3869 | 1.9944 |
| 54 | PaulMcBride | 13.4013 | 0.7731 |
| 55 | Kosmas7 | 13.4500 | 0.9095 |
| 56 | jackson5 | 13.4801 | 1.8915 |
| 57 | Smasko | 13.5287 | 0.9095 |
| 58 | Stealth *(team)* | 13.5598 | 0.6888 |
| 59 | sachin__09 | 13.5902 | 6.1771 |
| 60 | Chinchpokli | 13.7670 | 2.2188 |
| 61 | Santa | 13.7782 | 0.5213 |
| 62 | sumatorikki | 13.7953 | 0.6313 |
| 63 | qgoens | 13.8934 | 1.0951 |
| 64 | kar_len | 13.9865 | 1.5978 |
| 65 | KallK | 14.0341 | 4.2391 |
| 66 | ssimp10032 | 14.0453 | 0.7785 |
| 67 | emmettsexton | 14.0570 | 5.7702 |
| 68 | ocitalis | 14.0791 | 1.5591 |
| 69 | structAS | 14.1863 | 1.5141 |
| 70 | deactivated-e9b8d89 | 14.1911 | 1.6985 |
| 71 | dami | 14.3137 | 0.5586 |
| 72 | meenalmurugesan | 14.4510 | 0.6562 |
| 73 | yummyfe | 14.4712 | 2.4580 |
| 74 | shenjianmantou *(team)* | 14.4881 | 1.6592 |
| 75 | Athirajan | 14.5131 | 0.6562 |
| 76 | DataForGood *(team)* | 14.5520 | 1.6026 |
| 77 | koni-team *(team)* | 14.5538 | 0.8002 |
| 78 | Surgis12 | 14.5605 | 1.6586 |
| 79 | Helio | 14.6824 | 0.5062 |
| 80 | shimaa | 14.7513 | 5.8741 |
| 81 | jakey | 14.7643 | 0.5026 |
| 82 | mugesh9085 | 14.7742 | 0.4999 |
| 83 | jil300 | 14.7742 | 1.2294 |
| 84 | kiochan | 14.7742 | 0.5393 |
| 85 | Himfs | 14.7742 | 0.4924 |
| 86 | sheldon9789 | 14.7785 | 0.5007 |
| 87 | mashy | 14.8077 | 0.5573 |
| 88 | tika | 14.8096 | 0.5559 |
| 89 | Watson9789 | 14.8106 | 0.5870 |
| 90 | ragnar9085 | 14.8176 | 0.5441 |
| 91 | lothbrok | 14.8277 | 0.5622 |
| 92 | Buhari2612 | 14.8367 | 0.5653 |
| 93 | CJ343 | 14.8440 | 1.5758 |
| 94 | Sherlock281 | 14.8701 | 0.5872 |
| 95 | Rmugesh | 14.8978 | 0.5911 |
| 96 | mugesh2000 | 14.8981 | 0.5591 |
| 97 | AmineSamoudi | 14.9007 | 1.5745 |
| 98 | magist6 | 14.9299 | 6.8780 |
| 99 | MAT-U | 14.9796 | 1.8585 |
| 100 | dishantsaini55 | 15.0494 | 1.3737 |

> 🏆 = prize winner &nbsp;|&nbsp; *(team)* = multi-person team entry
