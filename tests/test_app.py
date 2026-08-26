from fastapi.testclient import TestClient

from teacher_support_studio.main import app

client = TestClient(app)


def test_home_and_health():
    assert client.get("/").status_code == 200
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_mapping_workbook_drives_class_and_student_names():
    classes = client.get("/api/v1/classes").json()
    assert classes[0]["label"] == "Riverside Math 7"
    students = client.get("/api/v1/classes/12309/students").json()
    assert students[0]["label"]
    assert not students[0]["label"].startswith("Student ")


def test_mapping_workbook_drives_skill_emojis():
    response = client.get("/api/v1/skill-emojis")
    assert response.status_code == 200
    mapping = response.json()
    assert len(mapping) == 44
    assert mapping["pattern finding"].strip()


def test_class_summary_endpoint():
    response = client.get("/api/v1/classes/12309/summary")
    assert response.status_code == 200
    assert response.json()["scope"] == "class"


def test_student_summary_endpoint_returns_next_practice_readiness():
    response = client.get("/api/v1/classes/12309/students/64634/summary")
    assert response.status_code == 200
    payload = response.json()
    assert payload["scope"] == "student"
    assert payload["readiness"]
    assert payload["readiness"][0]["scenario_count"] >= 1
    assert payload["readiness_min_interactions"] == 5
    assert all(item["prior_interactions"] >= 5 for item in payload["readiness"])


def test_student_ui_describes_readiness_instead_of_old_metrics():
    body = client.get("/").text
    assert "Estimated readiness by skill" in body
    assert "Learning activity trend" not in body
    assert "XGBoost success estimate" not in body
    assert "Historical success is shown" not in body


def test_guided_chat_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = client.post(
        "/api/v1/chat",
        json={
            "scope": "class",
            "class_id": 12309,
            "question": "What should I review next?",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["response_mode"] == "guided"
    assert payload["supporting_evidence"]
