# Kestrel V5 Experiment Report

Experiment: v5_run_012
Training status: EXPLORATORY ONLY

## Dataset

- Training rows: 8378
- Validation rows: 2126
- OOF rows: 2514
- Training positive rate: 11.40%
- Validation positive rate: 11.52%

## Threshold

- Selected threshold: 0.163857
- Target precision: 80.00%
- Target recall: 90.00%
- OOF target status: TARGETS_NOT_SIMULTANEOUSLY_MET_ON_OOF

## Validation Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| majority_class_baseline | 0.8848 | 0.0000 | 0.0000 | 0.0000 |
| lightgbm | 0.8838 | 0.4000 | 0.0163 | 0.0314 |
| catboost | 0.8434 | 0.2864 | 0.2408 | 0.2616 |
| xgboost | 0.7662 | 0.2225 | 0.4122 | 0.2890 |
| hybrid_stacking_default_threshold | 0.8852 | 1.0000 | 0.0041 | 0.0081 |
| hybrid_stacking | 0.7578 | 0.2289 | 0.4653 | 0.3069 |

## Important limitations

- Validation results were not used for model selection.
- Label maturity has not been independently verified.
- Point-in-time feature availability still requires auditing.
- OOF-to-final-refit prediction distribution stability requires monitoring.
- Target precision and recall are objectives, not guarantees.
