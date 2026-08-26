# Deployable model artifacts

Each modeling notebook owns one self-contained deployment subfolder:

- `logistic_regression/` is exported by notebook 04. It contains the fitted
  scikit-learn preprocessing pipeline and classifier plus a JSON metadata
  contract.
- `xgboost/` is exported by notebook 05. It contains the native XGBoost JSON
  model used by Teacher Support Studio plus a JSON metadata contract.

The metadata files define the ordered feature list, prediction meaning,
validation-selected threshold, training scope, held-out metrics, package
versions, and reference predictions. Re-execute a notebook to refresh its own
subfolder.
