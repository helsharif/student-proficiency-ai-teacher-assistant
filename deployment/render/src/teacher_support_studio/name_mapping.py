"""Reloadable synthetic display-name mappings for the teacher demo."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING_PATH = (
    PROJECT_ROOT / "outputs" / "teacher_support_studio" / "teacher_support_name_mapping.xlsx"
)


def _is_enabled(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


class NameMappingService:
    """Read editable class/student aliases and reload after a saved workbook change."""

    def __init__(self, path: Path = DEFAULT_MAPPING_PATH) -> None:
        self.path = path
        self._modified_ns: int | None = None
        self._classes = pd.DataFrame()
        self._students = pd.DataFrame()

    def _reload_if_changed(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"Expected display-name mapping workbook at {self.path}")
        modified_ns = self.path.stat().st_mtime_ns
        if modified_ns == self._modified_ns:
            return

        sheets = pd.read_excel(self.path, sheet_name=["Classes", "Students"])
        classes = sheets["Classes"].copy()
        students = sheets["Students"].copy()
        required_class_columns = {"class_id", "synthetic_class_name", "include_in_demo"}
        required_student_columns = {
            "class_id",
            "student_id",
            "synthetic_student_name",
            "include_in_demo",
        }
        if not required_class_columns.issubset(classes.columns):
            missing = sorted(required_class_columns - set(classes.columns))
            raise ValueError(f"Classes sheet is missing columns: {missing}")
        if not required_student_columns.issubset(students.columns):
            missing = sorted(required_student_columns - set(students.columns))
            raise ValueError(f"Students sheet is missing columns: {missing}")

        classes = classes.dropna(subset=["class_id", "synthetic_class_name"])
        students = students.dropna(
            subset=["class_id", "student_id", "synthetic_student_name"]
        )
        classes["class_id"] = classes["class_id"].astype("int64")
        students[["class_id", "student_id"]] = students[
            ["class_id", "student_id"]
        ].astype("int64")
        classes["include_in_demo"] = classes["include_in_demo"].map(_is_enabled)
        students["include_in_demo"] = students["include_in_demo"].map(_is_enabled)
        if classes["class_id"].duplicated().any():
            raise ValueError("Classes sheet contains duplicate class_id values.")
        if students[["class_id", "student_id"]].duplicated().any():
            raise ValueError("Students sheet contains duplicate class_id/student_id pairs.")

        self._classes = classes
        self._students = students
        self._modified_ns = modified_ns

    def class_ids(self) -> list[int]:
        self._reload_if_changed()
        visible = self._classes[self._classes["include_in_demo"]]
        return [int(value) for value in visible["class_id"].tolist()]

    def class_label(self, class_id: int) -> str:
        self._reload_if_changed()
        rows = self._classes[
            self._classes["class_id"].eq(class_id) & self._classes["include_in_demo"]
        ]
        if rows.empty:
            raise KeyError(f"Unknown or hidden demo class {class_id}.")
        return str(rows.iloc[0]["synthetic_class_name"]).strip()

    def student_labels(self, class_id: int) -> dict[int, str]:
        self.class_label(class_id)
        rows = self._students[
            self._students["class_id"].eq(class_id) & self._students["include_in_demo"]
        ]
        return {
            int(row.student_id): str(row.synthetic_student_name).strip()
            for row in rows.itertuples(index=False)
        }

    def student_label(self, class_id: int, student_id: int) -> str:
        try:
            return self.student_labels(class_id)[student_id]
        except KeyError as exc:
            raise KeyError(
                f"Student {student_id} is not visible in demo class {class_id}."
            ) from exc
