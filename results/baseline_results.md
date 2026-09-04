# Baseline Results

This document tracks small, human-readable evaluation summaries for the temporal benchmark. 

*(Detailed massive CSVs of threshold logs and raw predictions are excluded from Git).*

## March 2, 2018 (Held-Out Test Set)

*Note: F1 scores above 99.7% for Persistence on ordinary $t+k$ evaluation heavily motivate our shift toward Pre-Attack Early Warning.*

```text
===================================================================================================================
FINAL COMPARISON ON HELD-OUT TEST SET (MAR 2)
===================================================================================================================
Model                                    | Horizon |     F1 | Precision | Recall | ROC-AUC |    FPR | Threshold | PosSamples
-------------------------------------------------------------------------------------------------------------------
Persistence                              | t+1     | 0.9985 |    0.9985 | 0.9985 |  0.9979 | 0.0028 |         - |       2031
Logistic Regression (latest-state)       | t+1     | 0.3565 |    0.9758 | 0.2181 |  0.9464 | 0.0101 |      0.60 |       2031
Persistence                              | t+2     | 0.9975 |    0.9975 | 0.9975 |  0.9965 | 0.0046 |         - |       2031
Logistic Regression (latest-state)       | t+2     | 0.3630 |    0.9742 | 0.2230 |  0.9465 | 0.0111 |      0.60 |       2031
Persistence                              | t+3     | 0.9970 |    0.9970 | 0.9970 |  0.9958 | 0.0055 |         - |       2031
Logistic Regression (latest-state)       | t+3     | 0.3610 |    0.9740 | 0.2216 |  0.9476 | 0.0111 |      0.60 |       2031
Persistence                              | t+5     | 0.9966 |    0.9966 | 0.9966 |  0.9950 | 0.0065 |         - |       2031
Logistic Regression (latest-state)       | t+5     | 0.4067 |    0.9739 | 0.2570 |  0.9465 | 0.0129 |      0.54 |       2031
===================================================================================================================

===================================================================================================================
TRAJECTORY-EVENT PREDICTION (MAX SCORING over t+1 to t+k)
Target: Any attack occurs within t+1 ... t+k
===================================================================================================================
Model                                    | Horizon |     F1 | Precision | Recall | ROC-AUC |    FPR | Threshold | PosSamples
-------------------------------------------------------------------------------------------------------------------
Persistence                              | 1 to 2  | 0.9978 |    0.9985 | 0.9971 |  0.9971 | 0.0028 |         - |       2034
Logistic Regression (latest-state)       | 1 to 2  | 0.3614 |    0.9762 | 0.2217 |  0.9457 | 0.0102 |      0.59 |       2034
Persistence                              | 1 to 3  | 0.9975 |    0.9990 | 0.9961 |  0.9971 | 0.0019 |         - |       2037
Logistic Regression (latest-state)       | 1 to 3  | 0.3485 |    0.9774 | 0.2121 |  0.9456 | 0.0093 |      0.61 |       2037
Persistence                              | 1 to 5  | 0.9973 |    0.9995 | 0.9951 |  0.9971 | 0.0009 |         - |       2040
Logistic Regression (latest-state)       | 1 to 5  | 0.3561 |    0.9780 | 0.2176 |  0.9453 | 0.0093 |      0.61 |       2040
===================================================================================================================
```
