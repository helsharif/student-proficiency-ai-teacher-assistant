# Student Proficiency Prediction & AI Teacher Assistant

An end-to-end educational data science project that uses longitudinal
ASSISTments practice data to predict a student's probability of first-attempt
success on upcoming skill practice.

The project compares an interpretable logistic regression baseline with
XGBoost, evaluates probability quality and calibration, and uses model-native
SHAP contributions to explain individual XGBoost predictions. The longer-term
goal is to feed student- and class-level analytics into a grounded teacher
assistant that translates structured evidence into teacher-friendly language.

## Project goals

- Engineer leakage-safe features using only information available before each
  predicted interaction.
- Compare predictive performance, calibration, interpretability, and practical
  intervention thresholds.
- Produce student-skill proficiency profiles and class-level analytics.
- Build a teacher-facing dashboard and evidence-grounded AI assistant.
- Keep educators responsible for instructional decisions and document the
  system's limitations and intended use.

## Data

The project uses the
[ASSISTments 2009–2010 Skill Builder dataset](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010).
Raw and processed CSV datasets are versioned with Git LFS. The repository also
includes project-specific data dictionaries under `data/data_dictionary/`.

After filtering to original problems with a valid skill identifier, the
modeling dataset contains 259,386 interactions from 4,163 students. First-attempt
correctness is 65.8%; the median student has 19 interactions, while the median
student-skill history contains five interactions.

## Latest findings

The current evaluation uses a global chronological split by `order_id`: the
earliest 70% of unique order identifiers for training, the next 15% for
validation, and the latest 15% for testing. All longitudinal predictors are
shifted so that a prediction for interaction `t` uses only information available
before `t`.

| Model | Test ROC-AUC | Test average precision | Test log loss | Test Brier score | Balanced accuracy at validation-selected threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.6746 | 0.8251 | 0.5565 | 0.1864 | 0.6121 at 0.590 |
| XGBoost | **0.7000** | **0.8454** | **0.5443** | **0.1823** | **0.6317 at 0.635** |
| Development-prevalence baseline | 0.5000 | 0.7169 | 0.6068 | 0.2078 | 0.5000 |

Key results:

- XGBoost provides a modest but consistent improvement over logistic regression
  on discrimination and probability-error metrics.
- Recent student-skill performance is the strongest nonlinear signal. XGBoost's
  leading gain feature is mean skill accuracy over the prior five relevant
  interactions; student prior accuracy, skill practice history, recent skill
  accuracy, streaks, and historical help seeking are also influential in SHAP
  explanations.
- The validation-selected XGBoost threshold favors a more even tradeoff between
  recognizing correct and incorrect outcomes than the default 0.50 threshold.
  It is an experimental operating point, not a validated educational cutoff.
- Results show temporal and population shift: training, validation, and test
  correctness rates are 65.6%, 60.9%, and 71.7%, respectively, and the final
  test window contains 38,909 interactions but only 87 students. The test result
  therefore should not be treated as evidence of broad student generalization.

## Current status

Completed work includes data-quality analysis, educational EDA, leakage-safe
longitudinal feature engineering, chronological dataset exports, an
L2-regularized logistic regression baseline, and a tuned XGBoost comparison with
global and local explanations. Both models use the same 170 documented
predictors and held-out evaluation protocol.

Next steps are subgroup and temporal-stability audits, student-clustered
uncertainty intervals, probability recalibration if needed, and an evaluation
split designed explicitly for unseen-student generalization. Dashboard and
grounded assistant development remain future work.

## Reproduce the analysis

Install the environment from `pyproject.toml` and `uv.lock`, then run the
notebooks in order:

1. `01_data_quality_and_filtering.ipynb`
2. `02_eda_learning_patterns.ipynb`
3. `03_feature_engineering.ipynb`
4. `04_logistic_regression_baseline.ipynb`
5. `05_xgboost_model.ipynb`

## Responsible use

This project is a demonstration of teacher-facing decision support. Its outputs
are not measures of intelligence, general ability, motivation, disability, or
psychological mastery, and should not automatically determine grading,
placement, or interventions.
