# Training summary

Generated 2026-09-25 08:38 UTC. **All data is synthetic** — see README.

## Data
- Rows after cleaning: 1200 (train 960, test 240, stratified by placement readiness, seed 42)
- Duplicates removed: 15
- Impossible values replaced with NaN (then median-imputed inside the pipeline): {'cgpa': 2, 'attendance_pct': 2, 'coding_score': 1, 'study_hours_per_week': 1}
- Model inputs: 14 academic/skill features. Age and gender are **excluded** from modelling.

## Selection rule
Classification: highest mean 5-fold cross-validated macro-F1 on the training set.
Regression: lowest mean 5-fold cross-validated RMSE. The test set is used only once, for final reporting.

## Academic performance — selected: **XGBoost**
| Model | CV macro-F1 | CV accuracy | Test accuracy | Test precision | Test recall | Test macro-F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|---|
| XGBoost | 0.836 | 0.834 | 0.800 | 0.814 | 0.811 | 0.812 | 0.959 |
| Logistic Regression | 0.831 | 0.833 | 0.817 | 0.829 | 0.821 | 0.824 | 0.967 |
| Random Forest | 0.826 | 0.826 | 0.783 | 0.810 | 0.788 | 0.797 | 0.953 |
| Gradient Boosting | 0.819 | 0.817 | 0.787 | 0.804 | 0.797 | 0.799 | 0.958 |
| Decision Tree | 0.802 | 0.801 | 0.767 | 0.797 | 0.777 | 0.786 | 0.934 |

## Placement readiness — selected: **Logistic Regression**
| Model | CV macro-F1 | CV accuracy | Test accuracy | Test precision | Test recall | Test macro-F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.781 | 0.823 | 0.867 | 0.842 | 0.829 | 0.835 | 0.975 |
| XGBoost | 0.710 | 0.765 | 0.796 | 0.755 | 0.757 | 0.754 | 0.947 |
| Gradient Boosting | 0.694 | 0.754 | 0.771 | 0.719 | 0.721 | 0.719 | 0.943 |
| Random Forest | 0.691 | 0.760 | 0.804 | 0.757 | 0.750 | 0.751 | 0.945 |
| Decision Tree | 0.621 | 0.692 | 0.713 | 0.658 | 0.651 | 0.651 | 0.882 |

## Placement probability — selected: **XGBoost**
| Model | CV RMSE | CV MAE | CV R² | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|---|---|
| XGBoost | 10.349 | 7.404 | 0.922 | 7.075 | 9.867 | 0.930 |
| Gradient Boosting | 10.586 | 7.641 | 0.918 | 7.405 | 10.390 | 0.923 |
| Random Forest | 11.479 | 8.145 | 0.904 | 7.950 | 11.211 | 0.910 |
| Ridge Regression | 14.760 | 12.164 | 0.842 | 10.009 | 12.778 | 0.883 |
| Decision Tree | 15.800 | 10.871 | 0.818 | 9.981 | 14.673 | 0.846 |

## Top drivers of placement readiness
| Feature | Share | Impact |
|---|---|---|
| Coding / DSA score | 0.196 | High |
| Internship | 0.183 | High |
| CGPA | 0.149 | Medium-high |
| Projects | 0.141 | Medium-high |
| Aptitude score | 0.071 | Medium |
| Backlogs | 0.063 | Medium |
| Certifications | 0.055 | Medium |
| Technical skills score | 0.054 | Medium |

## Top drivers of placement probability
| Feature | Share | Impact |
|---|---|---|
| Coding / DSA score | 0.201 | High |
| Internship | 0.175 | High |
| CGPA | 0.168 | High |
| Projects | 0.168 | High |
| Technical skills score | 0.102 | Medium-high |
| Aptitude score | 0.058 | Medium |
| Backlogs | 0.036 | Low |
| Certifications | 0.029 | Low |

## Fairness audit by gender (test set; gender is not a model input)
| gender   |   students |   readiness_accuracy |   mean_actual_probability |   mean_predicted_probability |   pct_predicted_ready_or_above |
|:---------|-----------:|---------------------:|--------------------------:|-----------------------------:|-------------------------------:|
| Female   |         87 |                0.885 |                    50.667 |                       50.643 |                          0.46  |
| Male     |        147 |                0.85  |                    43.948 |                       44.385 |                          0.415 |
| Other    |          6 |                1     |                    73.6   |                       71.558 |                          0.833 |
