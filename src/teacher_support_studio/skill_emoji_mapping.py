"""Reloadable skill emoji mappings for the teacher demo."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from teacher_support_studio.name_mapping import _is_enabled

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING_PATH = (
    PROJECT_ROOT / "outputs" / "teacher_support_studio" / "skill_emoji_mapping.xlsx"
)


class SkillEmojiMappingService:
    """Read editable skill icons and reload after a saved workbook change."""

    def __init__(self, path: Path = DEFAULT_MAPPING_PATH) -> None:
        self.path = path
        self._modified_ns: int | None = None
        self._mapping: dict[str, str] = {}

    def _reload_if_changed(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"Expected skill emoji mapping workbook at {self.path}")
        modified_ns = self.path.stat().st_mtime_ns
        if modified_ns == self._modified_ns:
            return

        frame = pd.read_excel(self.path, sheet_name="Skill Emojis", header=3)
        required_columns = {"skill_name", "emoji", "include_in_app"}
        if not required_columns.issubset(frame.columns):
            missing = sorted(required_columns - set(frame.columns))
            raise ValueError(f"Skill Emojis sheet is missing columns: {missing}")

        frame = frame.dropna(subset=["skill_name", "emoji"])
        frame["skill_name"] = frame["skill_name"].astype(str).str.strip()
        frame["emoji"] = frame["emoji"].astype(str).str.strip()
        frame["include_in_app"] = frame["include_in_app"].map(_is_enabled)
        if frame["skill_name"].str.casefold().duplicated().any():
            raise ValueError("Skill Emojis sheet contains duplicate skill_name values.")

        enabled = frame[frame["include_in_app"]]
        self._mapping = {
            row.skill_name.casefold(): row.emoji for row in enabled.itertuples(index=False)
        }
        self._modified_ns = modified_ns

    def mapping(self) -> dict[str, str]:
        self._reload_if_changed()
        return dict(self._mapping)
