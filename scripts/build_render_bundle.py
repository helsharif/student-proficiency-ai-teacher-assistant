"""Build the self-contained Render deployment bundle.

The local application and full research datasets remain the source of truth. This
script refreshes only generated files under ``deployment/render``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_ROOT = PROJECT_ROOT / "deployment" / "render"
SOURCE_PACKAGE = PROJECT_ROOT / "src" / "teacher_support_studio"
SOURCE_DATA = PROJECT_ROOT / "data" / "processed" / "skill_builder_data_feature_eng.csv"
SOURCE_DICTIONARY = (
    PROJECT_ROOT / "data" / "data_dictionary" / "skill_builder_data_feature_eng_data_dictionary.csv"
)
SOURCE_MODEL_DIR = PROJECT_ROOT / "models" / "xgboost"
SOURCE_MAPPING_DIR = PROJECT_ROOT / "outputs" / "teacher_support_studio"
LOCAL_DEMO_BADGE = "Local demo · Deidentified data"
LIVE_DEMO_BADGE = "Live demo · Deidentified data"

BASE_DATA_COLUMNS = [
    "student_class_id",
    "user_id",
    "order_id",
    "correct",
    "skill_name",
    "hint_count",
    "attempt_count",
    "bottom_hint",
    "answer_type",
    "hint_total",
    "position",
]


def _is_enabled(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _require_sources() -> None:
    required = [
        SOURCE_PACKAGE,
        SOURCE_DATA,
        SOURCE_DICTIONARY,
        SOURCE_MODEL_DIR / "xgboost_first_attempt.json",
        SOURCE_MODEL_DIR / "xgboost_first_attempt_metadata.json",
        SOURCE_MAPPING_DIR / "teacher_support_name_mapping.xlsx",
        SOURCE_MAPPING_DIR / "skill_emoji_mapping.xlsx",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required bundle sources:\n" + "\n".join(missing))


def _reset_generated_directory(path: Path) -> None:
    resolved = path.resolve()
    deployment = DEPLOYMENT_ROOT.resolve()
    if deployment not in resolved.parents:
        raise RuntimeError(f"Refusing to replace directory outside {deployment}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def _copy_runtime_files() -> None:
    package_target = DEPLOYMENT_ROOT / "src" / "teacher_support_studio"
    _reset_generated_directory(package_target)
    shutil.copytree(
        SOURCE_PACKAGE,
        package_target,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    deployment_index = package_target / "static" / "index.html"
    index_html = deployment_index.read_text(encoding="utf-8")
    if index_html.count(LOCAL_DEMO_BADGE) != 1:
        raise RuntimeError(
            f"Expected exactly one local demo badge in {deployment_index}."
        )
    deployment_index.write_text(
        index_html.replace(LOCAL_DEMO_BADGE, LIVE_DEMO_BADGE),
        encoding="utf-8",
    )

    dictionary_target = DEPLOYMENT_ROOT / "data" / "data_dictionary"
    _reset_generated_directory(dictionary_target)
    shutil.copy2(SOURCE_DICTIONARY, dictionary_target / SOURCE_DICTIONARY.name)

    model_target = DEPLOYMENT_ROOT / "models" / "xgboost"
    _reset_generated_directory(model_target)
    for filename in ("xgboost_first_attempt.json", "xgboost_first_attempt_metadata.json"):
        shutil.copy2(SOURCE_MODEL_DIR / filename, model_target / filename)

    mapping_target = DEPLOYMENT_ROOT / "outputs" / "teacher_support_studio"
    _reset_generated_directory(mapping_target)
    for filename in ("teacher_support_name_mapping.xlsx", "skill_emoji_mapping.xlsx"):
        shutil.copy2(SOURCE_MAPPING_DIR / filename, mapping_target / filename)


def _visible_demo_ids() -> tuple[set[int], set[int]]:
    mapping_path = SOURCE_MAPPING_DIR / "teacher_support_name_mapping.xlsx"
    sheets = pd.read_excel(mapping_path, sheet_name=["Classes", "Students"])
    classes = sheets["Classes"].copy()
    students = sheets["Students"].copy()
    classes["include_in_demo"] = classes["include_in_demo"].map(_is_enabled)
    students["include_in_demo"] = students["include_in_demo"].map(_is_enabled)

    class_ids = {
        int(value) for value in classes.loc[classes["include_in_demo"], "class_id"].dropna()
    }
    student_ids = {
        int(value)
        for value in students.loc[
            students["include_in_demo"] & students["class_id"].isin(class_ids),
            "student_id",
        ].dropna()
    }
    if not class_ids or not student_ids:
        raise RuntimeError("The mapping workbook does not enable any demo classes and students.")
    return class_ids, student_ids


def _write_compact_data() -> tuple[int, int]:
    metadata_path = SOURCE_MODEL_DIR / "xgboost_first_attempt_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_features = [str(value) for value in metadata["feature_names"]]
    use_columns = list(dict.fromkeys([*BASE_DATA_COLUMNS, *model_features]))
    class_ids, student_ids = _visible_demo_ids()

    data_target = DEPLOYMENT_ROOT / "data" / "processed"
    _reset_generated_directory(data_target)
    output_path = data_target / "skill_builder_data_feature_eng.csv"
    row_count = 0
    wrote_header = False
    for chunk in pd.read_csv(SOURCE_DATA, usecols=use_columns, chunksize=25_000):
        selected = chunk[
            chunk["student_class_id"].isin(class_ids) | chunk["user_id"].isin(student_ids)
        ]
        if selected.empty:
            continue
        selected.to_csv(output_path, mode="a", index=False, header=not wrote_header)
        wrote_header = True
        row_count += len(selected)

    if not wrote_header:
        raise RuntimeError("No rows matched the enabled demo classes and students.")
    return row_count, output_path.stat().st_size


def build_bundle() -> None:
    _require_sources()
    DEPLOYMENT_ROOT.mkdir(parents=True, exist_ok=True)
    _copy_runtime_files()
    rows, size_bytes = _write_compact_data()
    print(
        f"Render bundle refreshed: {rows:,} rows, "
        f"{size_bytes / (1024 * 1024):.2f} MB compact dataset."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build_bundle()


if __name__ == "__main__":
    main()
