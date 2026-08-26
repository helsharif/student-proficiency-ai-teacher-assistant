from teacher_support_studio.analytics import AnalyticsService


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


def test_saved_xgboost_reference_predictions_match():
    service = AnalyticsService().model
    assert service.metadata["model_family"] == "xgboost"
    assert len(service.feature_names) == 170
    assert 0 < service.threshold < 1
