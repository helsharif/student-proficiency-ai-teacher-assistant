"""Descriptive analytics used by the local teacher-support demo."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from teacher_support_studio.model_service import XGBoostModelService
from teacher_support_studio.name_mapping import NameMappingService
from teacher_support_studio.readiness import MIN_PRIOR_SKILL_INTERACTIONS, ReadinessService
from teacher_support_studio.schemas import (
    DashboardSummary,
    EntityOption,
    MetricCard,
    SkillMetric,
    SkillReadiness,
    TrendPoint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "skill_builder_data_feature_eng.csv"

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


def _percent(value: float) -> str:
    return f"{value * 100:.0f}%"


def _points(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.0f} pts"


def _recent_and_earlier(frame: pd.DataFrame, size: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(["user_id", "order_id"])
    position_from_end = ordered.groupby("user_id").cumcount(ascending=False)
    recent = ordered[position_from_end < size]
    earlier = ordered[(position_from_end >= size) & (position_from_end < size * 2)]
    return recent, earlier


def _trend_points(frame: pd.DataFrame, bins: int = 8) -> list[TrendPoint]:
    ordered = frame.sort_values("order_id").reset_index(drop=True)
    if ordered.empty:
        return []
    index_groups = np.array_split(np.arange(len(ordered)), min(bins, len(ordered)))
    groups = [ordered.iloc[indexes] for indexes in index_groups]
    return [
        TrendPoint(
            label=f"{index + 1}",
            success_rate=round(float(group["correct"].mean()), 4),
            interactions=len(group),
        )
        for index, group in enumerate(groups)
        if not group.empty
    ]


@lru_cache(maxsize=8)
def _load_demo_data(
    class_ids: tuple[int, ...],
    student_ids: tuple[int, ...],
    model_features: tuple[str, ...],
) -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Expected modeling data at {DATA_PATH}")

    use_columns = list(dict.fromkeys([*BASE_DATA_COLUMNS, *model_features]))
    selected: list[pd.DataFrame] = []
    for chunk in pd.read_csv(DATA_PATH, usecols=use_columns, chunksize=25_000):
        subset = chunk[
            chunk["student_class_id"].isin(class_ids) | chunk["user_id"].isin(student_ids)
        ].copy()
        if not subset.empty:
            selected.append(subset)

    if not selected:
        raise RuntimeError("None of the configured demo classes were found in the data.")

    data = pd.concat(selected, ignore_index=True)
    data["correct"] = pd.to_numeric(data["correct"], errors="coerce").fillna(0)
    data["hint_count"] = pd.to_numeric(data["hint_count"], errors="coerce").fillna(0)
    data["skill_name"] = data["skill_name"].fillna("Unlabeled skill")
    return data.sort_values("order_id").reset_index(drop=True)


class AnalyticsService:
    """Load a small set of demo classes and return teacher-friendly summaries."""

    def __init__(
        self,
        names: NameMappingService | None = None,
        model: XGBoostModelService | None = None,
    ) -> None:
        self.names = names or NameMappingService()
        self.model = model or XGBoostModelService()
        self.readiness = ReadinessService(self.model)

    @property
    def data(self) -> pd.DataFrame:
        student_ids = {
            student_id
            for class_id in self.names.class_ids()
            for student_id in self.names.student_labels(class_id)
        }
        return _load_demo_data(
            tuple(self.names.class_ids()),
            tuple(sorted(student_ids)),
            tuple(self.model.feature_names),
        )

    def class_options(self) -> list[EntityOption]:
        options: list[EntityOption] = []
        for class_id in self.names.class_ids():
            frame = self.data[self.data["student_class_id"] == class_id]
            options.append(
                EntityOption(
                    id=class_id,
                    label=self.names.class_label(class_id),
                    detail=(
                        f"{frame['user_id'].nunique()} students · "
                        f"{frame['skill_name'].nunique()} skills"
                    ),
                )
            )
        return options

    def student_options(self, class_id: int) -> list[EntityOption]:
        frame = self._class_frame(class_id)
        labels = self.names.student_labels(class_id)
        student_ids = sorted(
            int(value) for value in frame["user_id"].unique() if int(value) in labels
        )
        counts = frame.groupby("user_id").size()
        return [
            EntityOption(
                id=student_id,
                label=labels[student_id],
                detail=f"{int(counts.loc[student_id])} recorded interactions",
            )
            for student_id in sorted(student_ids, key=lambda value: (-counts.loc[value], value))
        ]

    def class_summary(self, class_id: int) -> DashboardSummary:
        frame = self._class_frame(class_id)
        recent, earlier = _recent_and_earlier(frame)
        recent_rate = float(recent["correct"].mean())

        latest_by_student = frame.sort_values(["user_id", "order_id"]).groupby(
            "user_id", as_index=False
        ).tail(1)
        model_probability = self.model.predict_probability(latest_by_student)
        model_average = float(model_probability.mean())
        support_signal_count = int((model_probability < self.model.threshold).sum())

        skills = self._skill_metrics(recent, earlier, include_students=True)
        priority = min(skills, key=lambda item: item.success_rate) if skills else None
        priority_text = priority.label if priority else "recently practiced skills"

        return DashboardSummary(
            scope="class",
            entity_id=class_id,
            entity_label=self.names.class_label(class_id),
            context_label="Class focus",
            headline=f"Start with {priority_text}",
            cards=[
                MetricCard(
                    label="XGBoost success estimate",
                    value=_percent(model_average),
                    detail="Average estimate from each student's latest modeled interaction",
                    tone="positive" if model_average >= self.model.threshold else "neutral",
                ),
                MetricCard(
                    label="Recent observed success",
                    value=_percent(recent_rate),
                    detail="Across each student's last 10 interactions",
                    tone="positive" if recent_rate >= 0.7 else "neutral",
                ),
                MetricCard(
                    label="Below model support signal",
                    value=str(support_signal_count),
                    detail=f"Below the notebook-selected {self.model.threshold:.3f} threshold",
                    tone="attention" if support_signal_count else "neutral",
                ),
                MetricCard(
                    label="Skills represented",
                    value=str(recent["skill_name"].nunique()),
                    detail=f"Across {len(recent):,} recent interactions",
                ),
            ],
            skills=skills,
            trend=_trend_points(frame),
            suggested_questions=[
                "What should I review next?",
                "Where is the class making progress?",
                "Which students may need a quick check-in?",
                f"Plan a 10-minute warm-up for {priority_text}.",
            ],
            evidence=self._class_evidence(
                frame,
                recent,
                earlier,
                skills,
                model_average,
                support_signal_count,
                self.model.threshold,
            ),
        )

    def student_summary(self, class_id: int, student_id: int) -> DashboardSummary:
        class_frame = self._class_frame(class_id)
        class_membership = class_frame[class_frame["user_id"] == student_id]
        if class_membership.empty:
            raise KeyError(f"Student {student_id} is not in demo class {class_id}.")
        frame = self.data[self.data["user_id"] == student_id]

        all_readiness = self.readiness.score_student(class_frame, frame)
        readiness = [
            item
            for item in all_readiness
            if item.prior_interactions >= MIN_PRIOR_SKILL_INTERACTIONS
        ]
        excluded_count = len(all_readiness) - len(readiness)
        focus = min(readiness, key=lambda item: item.estimated_readiness) if readiness else None
        focus_text = focus.label if focus else None
        label = self._student_label(class_id, student_id)

        return DashboardSummary(
            scope="student",
            entity_id=student_id,
            entity_label=label,
            context_label=f"Student focus · {self.names.class_label(class_id)}",
            headline=(
                f"Review estimated readiness for {focus_text}"
                if focus_text
                else "No skills yet meet the evidence threshold"
            ),
            cards=[],
            skills=[],
            trend=[],
            readiness=readiness,
            readiness_min_interactions=MIN_PRIOR_SKILL_INTERACTIONS,
            suggested_questions=[
                "Which skill should I check first, and why?",
                "Why were some practiced skills excluded?",
                "What do the highest readiness estimates suggest?",
                "Suggest a low-stakes check-in for the priority skill.",
            ],
            evidence=self._readiness_evidence(frame, readiness, excluded_count),
        )

    def _class_frame(self, class_id: int) -> pd.DataFrame:
        self.names.class_label(class_id)
        return self.data[self.data["student_class_id"] == class_id]

    def _student_label(self, class_id: int, student_id: int) -> str:
        return self.names.student_label(class_id, student_id)

    @staticmethod
    def _skill_metrics(
        recent: pd.DataFrame,
        earlier: pd.DataFrame,
        *,
        include_students: bool,
    ) -> list[SkillMetric]:
        recent_group = recent.groupby("skill_name").agg(
            interactions=("correct", "size"),
            success_rate=("correct", "mean"),
            students=("user_id", "nunique"),
        )
        earlier_rate = earlier.groupby("skill_name")["correct"].mean()
        recent_group = recent_group.sort_values("interactions", ascending=False).head(6)
        metrics: list[SkillMetric] = []
        for skill_name, row in recent_group.iterrows():
            previous = earlier_rate.get(skill_name, np.nan)
            change = (
                None if pd.isna(previous) else (float(row["success_rate"]) - float(previous)) * 100
            )
            metrics.append(
                SkillMetric(
                    label=str(skill_name),
                    success_rate=round(float(row["success_rate"]), 4),
                    interactions=int(row["interactions"]),
                    students=int(row["students"]) if include_students else None,
                    change_points=round(change, 1) if change is not None else None,
                )
            )
        return metrics

    @staticmethod
    def _class_evidence(
        frame: pd.DataFrame,
        recent: pd.DataFrame,
        earlier: pd.DataFrame,
        skills: list[SkillMetric],
        model_average: float,
        support_signal_count: int,
        model_threshold: float,
    ) -> list[str]:
        recent_rate = float(recent["correct"].mean())
        earlier_rate = float(earlier["correct"].mean()) if not earlier.empty else recent_rate
        evidence = [
            (
                f"The class includes {frame['user_id'].nunique()} students and "
                f"{len(frame):,} modeled interactions."
            ),
            (
                f"The persisted XGBoost model estimates {_percent(model_average)} average "
                "first-attempt success across each student's latest modeled interaction."
            ),
            (
                f"{support_signal_count} students fall below the model's {model_threshold:.3f} "
                "support signal; this is a review prompt, not a diagnosis."
            ),
            (
                f"Recent observed first-attempt success is {_percent(recent_rate)}, "
                f"{_points((recent_rate - earlier_rate) * 100)} from the preceding window."
            ),
        ]
        if skills:
            lowest = min(skills, key=lambda item: item.success_rate)
            evidence.append(
                f"{lowest.label} has {_percent(lowest.success_rate)} recent "
                f"first-attempt success across {lowest.interactions} interactions."
            )
        return evidence

    @staticmethod
    def _readiness_evidence(
        frame: pd.DataFrame,
        readiness: list[SkillReadiness],
        excluded_count: int,
    ) -> list[str]:
        evidence = [
            (
                f"The learner state uses all {len(frame)} recorded interactions across "
                f"{frame['skill_name'].nunique()} historically practiced skills."
            ),
            (
                "Each estimate is the median XGBoost probability across up to ten next-practice "
                "contexts that actually occurred for the same skill in the selected class."
            ),
            (
                "Scenarios vary supported question format and problem context; they do not "
                "represent specific question wording or guaranteed future performance."
            ),
            (
                f"Only skills with at least {MIN_PRIOR_SKILL_INTERACTIONS} prior learner "
                f"interactions are considered; {excluded_count} skills were excluded for "
                "insufficient evidence."
            ),
        ]
        if readiness:
            lowest = min(readiness, key=lambda item: item.estimated_readiness)
            evidence.append(
                f"{lowest.label} has an estimated {_percent(lowest.estimated_readiness)} "
                f"first-attempt readiness across {lowest.scenario_count} plausible scenarios; "
                f"the learner has {lowest.prior_interactions} prior interactions in this skill."
            )
        return evidence
