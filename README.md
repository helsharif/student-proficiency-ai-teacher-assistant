# Student Proficiency Prediction & AI Teacher Assistant

An end-to-end educational data science project that uses longitudinal
ASSISTments practice data to predict a student's probability of first-attempt
success on upcoming skill practice.

The project will compare an interpretable logistic regression baseline with
XGBoost, evaluate probability calibration, and use SHAP to explain individual
predictions. Predictions will feed student- and class-level teacher dashboards
and a grounded AI assistant that translates structured analytics into
teacher-friendly evidence and instructional suggestions.

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
Downloaded raw data and generated artifacts are excluded from version control.
The repository includes a project-specific data dictionary under
`data/data_dictionary/`.

## Status

Early development. The initial focus is data validation, educational exploratory
analysis, and leakage-safe longitudinal feature engineering.

## Responsible use

This project is a demonstration of teacher-facing decision support. Its outputs
are not measures of intelligence, general ability, motivation, disability, or
psychological mastery, and should not automatically determine grading,
placement, or interventions.
