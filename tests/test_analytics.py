import numpy as np

from teacher_support_studio.analytics import AnalyticsService
from teacher_support_studio.readiness import PracticeContext


def test_class_summary_has_teacher_facing_content():
    summary = AnalyticsService().class_summary(12309)

    assert summary.scope == "class"
    assert len(summary.cards) == 4
    assert summary.skills
    assert summary.trend
    assert len(summary.suggested_questions) == 4
    assert summary.cards[0].label == "XGBoost success estimate"


def test_student_summary_uses_synthetic_label():
    service = AnalyticsService()
    student = service.student_options(12309)[0]
    summary = service.student_summary(12309, student.id)

    assert summary.scope == "student"
    assert summary.entity_label == student.label
    assert str(student.id) not in summary.entity_label
    assert summary.evidence
    assert summary.readiness
    assert not summary.skills
    assert not summary.trend
    assert all(0 <= item.estimated_readiness <= 1 for item in summary.readiness)
    assert all(1 <= item.scenario_count <= 10 for item in summary.readiness)
    assert all(item.prior_interactions >= 5 for item in summary.readiness)
    assert summary.readiness_min_interactions == 5
    assert not summary.cards
    estimates = [item.estimated_readiness for item in summary.readiness]
    assert estimates == sorted(estimates, reverse=True)


def test_scenario_builder_reconstructs_historical_model_row():
    service = AnalyticsService()
    class_frame = service._class_frame(12309)
    student_frame = (
        class_frame[class_frame["user_id"].eq(64634)]
        .sort_values("order_id", kind="stable")
        .reset_index(drop=True)
    )
    target_position = next(
        position
        for position in range(20, len(student_frame))
        if student_frame.loc[position, service.readiness.skill_features].sum() == 1
    )
    target = student_frame.loc[target_position]
    skill_feature = next(
        feature for feature in service.readiness.skill_features if target[feature] == 1
    )
    context = PracticeContext(
        answer_type=str(target["answer_type"]),
        hint_total=float(target["hint_total"]),
        position=float(target["position"]),
        historical_examples=1,
    )

    reconstructed = service.readiness.build_scenario_row(
        student_frame.iloc[:target_position],
        skill_feature,
        context,
        opportunity_override=float(target["opportunity"]),
    ).iloc[0]

    np.testing.assert_allclose(
        reconstructed.to_numpy(dtype=float),
        target[service.model.feature_names].to_numpy(dtype=float),
    )


def test_saved_xgboost_reference_predictions_match():
    service = AnalyticsService().model
    assert service.metadata["model_family"] == "xgboost"
    assert len(service.feature_names) == 170
    assert 0 < service.threshold < 1


def test_student_state_includes_history_outside_selected_class():
    service = AnalyticsService()
    student_id = 80906
    selected_class_interactions = len(
        service._class_frame(12361)[service._class_frame(12361)["user_id"].eq(student_id)]
    )
    all_interactions = len(service.data[service.data["user_id"].eq(student_id)])
    summary = service.student_summary(12361, student_id)

    assert all_interactions > selected_class_interactions
    assert str(all_interactions) in summary.evidence[0]
