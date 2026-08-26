"""Add idempotent deployment-artifact export cells to modeling notebooks."""

from __future__ import annotations

from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_TAG = "deployment-artifact-export"


def _append_export_section(path: Path, title: str, code: str) -> None:
    notebook = nbformat.read(path, as_version=4)
    notebook.cells = [
        cell
        for cell in notebook.cells
        if EXPORT_TAG not in cell.get("metadata", {}).get("tags", [])
    ]
    notebook.cells.extend(
        [
            nbformat.v4.new_markdown_cell(
                title,
                metadata={"tags": [EXPORT_TAG]},
            ),
            nbformat.v4.new_code_cell(
                code.strip() + "\n",
                metadata={"tags": [EXPORT_TAG]},
            ),
        ]
    )
    nbformat.write(notebook, path)


LOGISTIC_CODE = r'''
import json
from datetime import datetime, timezone

import joblib
import sklearn

ARTIFACT_DIR = PROJECT_ROOT / "models" / "logistic_regression"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ARTIFACT_PATH = ARTIFACT_DIR / "logistic_first_attempt_pipeline.joblib"
MODEL_METADATA_PATH = ARTIFACT_DIR / "logistic_first_attempt_metadata.json"

joblib.dump(final_model, MODEL_ARTIFACT_PATH, compress=3)

selected_test_metrics = test_performance.iloc[1].to_dict()
metadata = {
    "artifact_version": 1,
    "model_family": "logistic_regression",
    "model_format": "joblib_sklearn_pipeline",
    "source_notebook": "notebooks/04_logistic_regression_baseline.ipynb",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "target": TARGET,
    "positive_class": 1,
    "prediction_semantics": "Probability of a correct first attempt for one interaction.",
    "feature_names": feature_names,
    "feature_count": len(feature_names),
    "selected_threshold": SELECTED_THRESHOLD,
    "training_scope": "chronological train and validation splits",
    "selection": {"C": BEST_C},
    "test_metrics": selected_test_metrics,
    "library_versions": {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    },
    "reference_predictions": [
        {
            "order_id": int(test.iloc[position]["order_id"]),
            "probability": float(test_probability[position]),
        }
        for position in range(min(5, len(test)))
    ],
}
MODEL_METADATA_PATH.write_text(
    json.dumps(metadata, indent=2, default=lambda value: value.item()),
    encoding="utf-8",
)

loaded_model = joblib.load(MODEL_ARTIFACT_PATH)
loaded_probability = loaded_model.predict_proba(X_test.iloc[:5])[:, 1]
np.testing.assert_allclose(loaded_probability, test_probability[:5], rtol=0, atol=1e-12)

print(f"Saved deployable model: {MODEL_ARTIFACT_PATH.relative_to(PROJECT_ROOT)}")
print(f"Saved metadata contract: {MODEL_METADATA_PATH.relative_to(PROJECT_ROOT)}")
print(f"Artifact size: {MODEL_ARTIFACT_PATH.stat().st_size / 1024**2:.2f} MiB")
'''


XGBOOST_CODE = r'''
import json
from datetime import datetime, timezone

ARTIFACT_DIR = PROJECT_ROOT / "models" / "xgboost"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ARTIFACT_PATH = ARTIFACT_DIR / "xgboost_first_attempt.json"
MODEL_METADATA_PATH = ARTIFACT_DIR / "xgboost_first_attempt_metadata.json"

# XGBoost's native JSON format is language-portable and avoids pickle compatibility risk.
final_model.save_model(MODEL_ARTIFACT_PATH)

selected_test_metrics = test_performance.iloc[1].to_dict()
metadata = {
    "artifact_version": 1,
    "model_family": "xgboost",
    "model_format": "xgboost_native_json",
    "source_notebook": "notebooks/05_xgboost_model.ipynb",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "target": TARGET,
    "positive_class": 1,
    "prediction_semantics": "Probability of a correct first attempt for one interaction.",
    "feature_names": feature_names,
    "feature_count": len(feature_names),
    "selected_threshold": SELECTED_THRESHOLD,
    "training_scope": "chronological train and validation splits",
    "selection": {
        "configuration": BEST_CONFIGURATION_NAME,
        "boosting_rounds": BEST_BOOSTING_ROUNDS,
        "parameters": BEST_PARAMETERS,
    },
    "test_metrics": selected_test_metrics,
    "library_versions": {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "xgboost": xgb.__version__,
    },
    "reference_predictions": [
        {
            "order_id": int(test.iloc[position]["order_id"]),
            "probability": float(test_probability[position]),
        }
        for position in range(min(5, len(test)))
    ],
}
MODEL_METADATA_PATH.write_text(
    json.dumps(metadata, indent=2, default=lambda value: value.item()),
    encoding="utf-8",
)

loaded_model = XGBClassifier()
loaded_model.load_model(MODEL_ARTIFACT_PATH)
loaded_probability = loaded_model.predict_proba(X_test.iloc[:5])[:, 1]
np.testing.assert_allclose(loaded_probability, test_probability[:5], rtol=0, atol=1e-7)

print(f"Saved deployable model: {MODEL_ARTIFACT_PATH.relative_to(PROJECT_ROOT)}")
print(f"Saved metadata contract: {MODEL_METADATA_PATH.relative_to(PROJECT_ROOT)}")
print(f"Artifact size: {MODEL_ARTIFACT_PATH.stat().st_size / 1024**2:.2f} MiB")
'''


def main() -> None:
    _append_export_section(
        PROJECT_ROOT / "notebooks" / "04_logistic_regression_baseline.ipynb",
        """## Export deployable logistic-regression artifact

The fitted preprocessing pipeline and classifier are saved together so deployment uses
the exact imputation, scaling, feature order, and coefficients evaluated above. A JSON
metadata contract records the decision threshold, held-out metrics, versions, and small
reference prediction set for deployment checks.""",
        LOGISTIC_CODE,
    )
    _append_export_section(
        PROJECT_ROOT / "notebooks" / "05_xgboost_model.ipynb",
        """## Export deployable XGBoost artifact

The final classifier is saved in XGBoost's native JSON format for portable serving. A
separate metadata contract records the ordered predictors, validation-selected threshold,
held-out metrics, package versions, and reference predictions used for deployment checks.
Teacher Support Studio loads this artifact rather than retraining at application startup.""",
        XGBOOST_CODE,
    )


if __name__ == "__main__":
    main()
