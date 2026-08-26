"""Plausible next-practice scenario construction and readiness scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from teacher_support_studio.model_service import XGBoostModelService
from teacher_support_studio.schemas import SkillReadiness

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DICTIONARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "data_dictionary"
    / "skill_builder_data_feature_eng_data_dictionary.csv"
)

ANSWER_TYPE_LABELS = {
    "algebra": "Algebra entry",
    "choose_1": "Single choice",
    "choose_n": "Multiple choice",
    "fill_in_1": "Fill-in",
}
ANSWER_TYPE_FEATURES = {
    answer_type: f"answer_type_{answer_type}" for answer_type in ANSWER_TYPE_LABELS
}
MIN_PRIOR_SKILL_INTERACTIONS = 5


@dataclass(frozen=True)
class PracticeContext:
    answer_type: str
    hint_total: float
    position: float
    historical_examples: int


def _recent_mean(values: pd.Series, window: int) -> float:
    return float(values.tail(window).mean()) if len(values) else 0.0


def _trailing_count(values: pd.Series, target: int) -> int:
    count = 0
    for value in reversed(values.astype("int8").tolist()):
        if value != target:
            break
        count += 1
    return count


def _evidence_level(prior_interactions: int) -> str:
    if prior_interactions == 0:
        return "No prior practice"
    if prior_interactions < 5:
        return "Limited"
    if prior_interactions < 15:
        return "Moderate"
    return "Established"


class ReadinessService:
    """Score historically supported next-practice contexts for named skills."""

    def __init__(self, model: XGBoostModelService) -> None:
        self.model = model
        self.skill_features = [
            name for name in self.model.feature_names if name.startswith("Skill_")
        ]
        dictionary = pd.read_csv(FEATURE_DICTIONARY_PATH, keep_default_na=False)
        lookup = dictionary.set_index("column_name")["skill_plain_language_name"]
        self.named_skill_features = [
            feature for feature in self.skill_features if str(lookup.get(feature, "")).strip()
        ]
        self.skill_labels = {
            feature: str(lookup.get(feature, "")).strip() or feature.replace("_", " ")
            for feature in self.skill_features
        }
        self._assert_supported_contract()

    def _assert_supported_contract(self) -> None:
        generated = {
            "position",
            "hint_total",
            "opportunity",
            "prior_interaction_count",
            "min_prior_skill_interaction_count",
            "mean_prior_skill_interaction_count",
            "median_prior_skill_interaction_count",
            "max_prior_skill_interaction_count",
            "is_first_skill_interaction",
            "student_prior_attempts",
            "student_prior_correct",
            "student_prior_accuracy",
            "student_recent_accuracy_3",
            "student_recent_accuracy_5",
            "student_recent_accuracy_10",
            "student_correct_streak",
            "student_incorrect_streak",
            "no_prior_history",
            "insufficient_history_3",
            "insufficient_history_5",
            "insufficient_history_10",
            "student_prior_hint_rate",
            "student_prior_bottom_hint_rate",
            *ANSWER_TYPE_FEATURES.values(),
            *self.skill_features,
        }
        for base in (
            "skill_prior_attempts",
            "skill_prior_accuracy",
            "skill_recent_accuracy_3",
            "skill_recent_accuracy_5",
            "skill_recent_accuracy_10",
        ):
            generated.update(
                f"{statistic}_{base}" for statistic in ("min", "mean", "median", "max")
            )
        missing = sorted(set(self.model.feature_names) - generated)
        if missing:
            raise ValueError(f"Scenario builder does not support model features: {missing}")

    def build_scenario_row(
        self,
        history: pd.DataFrame,
        skill_feature: str,
        context: PracticeContext,
        *,
        opportunity_override: float | None = None,
    ) -> pd.DataFrame:
        """Build one model row using only completed historical interactions."""
        if skill_feature not in self.skill_features:
            raise KeyError(f"Unknown model skill feature {skill_feature}.")
        if context.answer_type not in ANSWER_TYPE_FEATURES:
            raise KeyError(f"Unsupported answer type {context.answer_type}.")

        ordered = history.sort_values("order_id", kind="stable")
        count = len(ordered)
        correct = ordered["correct"].astype("int8")
        skill_history = ordered[ordered[skill_feature].eq(1)]
        skill_correct = skill_history["correct"].astype("int8")
        skill_count = len(skill_history)

        row = {feature: 0.0 for feature in self.model.feature_names}
        row.update(
            {
                "position": float(context.position),
                "hint_total": float(context.hint_total),
                "prior_interaction_count": float(count),
                "student_prior_attempts": float(ordered["attempt_count"].sum()),
                "student_prior_correct": float(correct.sum()),
                "student_prior_accuracy": float(correct.mean()) if count else 0.0,
                "student_recent_accuracy_3": _recent_mean(correct, 3),
                "student_recent_accuracy_5": _recent_mean(correct, 5),
                "student_recent_accuracy_10": _recent_mean(correct, 10),
                "student_correct_streak": float(_trailing_count(correct, 1)),
                "student_incorrect_streak": float(_trailing_count(correct, 0)),
                "no_prior_history": float(count == 0),
                "insufficient_history_3": float(count < 3),
                "insufficient_history_5": float(count < 5),
                "insufficient_history_10": float(count < 10),
                "student_prior_hint_rate": (
                    float(ordered["hint_count"].gt(0).mean()) if count else 0.0
                ),
                "student_prior_bottom_hint_rate": (
                    float(ordered["bottom_hint"].eq(1).mean()) if count else 0.0
                ),
                skill_feature: 1.0,
                ANSWER_TYPE_FEATURES[context.answer_type]: 1.0,
            }
        )

        for statistic in ("min", "mean", "median", "max"):
            row[f"{statistic}_prior_skill_interaction_count"] = float(skill_count)

        row["is_first_skill_interaction"] = float(skill_count == 0)
        skill_history_values = {
            "skill_prior_attempts": float(skill_history["attempt_count"].sum()),
            "skill_prior_accuracy": float(skill_correct.mean()) if skill_count else 0.0,
            "skill_recent_accuracy_3": _recent_mean(skill_correct, 3),
            "skill_recent_accuracy_5": _recent_mean(skill_correct, 5),
            "skill_recent_accuracy_10": _recent_mean(skill_correct, 10),
        }
        for base, value in skill_history_values.items():
            for statistic in ("min", "mean", "median", "max"):
                row[f"{statistic}_{base}"] = value

        if opportunity_override is not None:
            row["opportunity"] = float(opportunity_override)
        elif skill_count:
            row["opportunity"] = float(skill_history["opportunity"].max() + 1)
        else:
            row["opportunity"] = 1.0

        return pd.DataFrame([row], columns=self.model.feature_names)

    def _contexts_for_skill(
        self,
        class_frame: pd.DataFrame,
        skill_feature: str,
        limit: int = 10,
    ) -> list[PracticeContext]:
        rows = class_frame[class_frame[skill_feature].eq(1)]
        if rows.empty:
            return []
        single_skill_rows = rows[rows[self.skill_features].sum(axis=1).eq(1)]
        if not single_skill_rows.empty:
            rows = single_skill_rows
        grouped = (
            rows.groupby(["answer_type", "hint_total", "position"], dropna=False)
            .size()
            .rename("historical_examples")
            .reset_index()
            .sort_values("historical_examples", ascending=False)
        )
        contexts: list[PracticeContext] = []
        for answer_type in grouped.groupby("answer_type", sort=False).size().index:
            representative = grouped[grouped["answer_type"].eq(answer_type)].iloc[0]
            contexts.append(
                PracticeContext(
                    answer_type=str(representative["answer_type"]),
                    hint_total=float(representative["hint_total"]),
                    position=float(representative["position"]),
                    historical_examples=int(representative["historical_examples"]),
                )
            )
            if len(contexts) == limit:
                return contexts

        selected = {(item.answer_type, item.hint_total, item.position) for item in contexts}
        for representative in grouped.itertuples(index=False):
            key = (
                str(representative.answer_type),
                float(representative.hint_total),
                float(representative.position),
            )
            if key in selected:
                continue
            contexts.append(
                PracticeContext(
                    answer_type=key[0],
                    hint_total=key[1],
                    position=key[2],
                    historical_examples=int(representative.historical_examples),
                )
            )
            if len(contexts) == limit:
                break
        return contexts

    def score_student(
        self,
        class_frame: pd.DataFrame,
        student_frame: pd.DataFrame,
    ) -> list[SkillReadiness]:
        scenario_rows: list[pd.DataFrame] = []
        scenario_index: list[tuple[str, PracticeContext]] = []
        for skill_feature in self.named_skill_features:
            for context in self._contexts_for_skill(class_frame, skill_feature):
                scenario_rows.append(
                    self.build_scenario_row(student_frame, skill_feature, context)
                )
                scenario_index.append((skill_feature, context))
        if not scenario_rows:
            return []

        scenario_frame = pd.concat(scenario_rows, ignore_index=True)
        predictions = self.model.predict_probability(scenario_frame)
        records = pd.DataFrame(
            {
                "skill_feature": [item[0] for item in scenario_index],
                "answer_type": [item[1].answer_type for item in scenario_index],
                "prediction": predictions,
            }
        )

        readiness: list[SkillReadiness] = []
        for skill_feature, skill_scenarios in records.groupby("skill_feature", sort=False):
            history = student_frame[student_frame[skill_feature].eq(1)]
            prior_interactions = len(history)
            readiness.append(
                SkillReadiness(
                    label=self.skill_labels[skill_feature],
                    estimated_readiness=round(float(skill_scenarios["prediction"].median()), 4),
                    scenario_low=round(float(skill_scenarios["prediction"].min()), 4),
                    scenario_high=round(float(skill_scenarios["prediction"].max()), 4),
                    historical_success=(
                        round(float(history["correct"].mean()), 4)
                        if prior_interactions
                        else None
                    ),
                    prior_interactions=prior_interactions,
                    scenario_count=len(skill_scenarios),
                    answer_types=[
                        ANSWER_TYPE_LABELS[value]
                        for value in skill_scenarios["answer_type"].drop_duplicates()
                    ],
                    evidence_level=_evidence_level(prior_interactions),
                )
            )
        return sorted(
            readiness,
            key=lambda item: (-item.estimated_readiness, item.label),
        )
