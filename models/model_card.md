# Model card: Logistic regression (baseline)

**Purpose.** Decision support for screening applications for fraud. It gives a risk score and a risk band.
It does not approve or reject anyone: High risk cases go to a human reviewer.

**Data.** Bank Account Fraud (BAF) dataset (Base.csv), synthetic, 60,000 applications, fraud rate 1.11%.
Time based split: months 0-3 train, months 4-5 validation (model choice, calibration, thresholds), months 6-7 test.

**Test results (months 6-7).** Recall at 5% false positive rate: 0.437
(95% CI 0.365 to 0.490). PR-AUC 0.122. ROC-AUC 0.868. Brier score 0.0141.

**Risk bands.** High = calibrated score at or above 0.0412; Medium = at or above 0.0131; otherwise Low.

**Fairness (age 50 and over vs under 50).** Predictive equality ratio 0.26
(95% CI 0.22 to 0.33); 1.0 means equal false positive rates.

**Limitations.** Synthetic bank data, not SASSA data. Labels are the dataset's own fraud flag (detected fraud). One protected attribute only.
Results must not be read as how the model would perform on real grant applications.
