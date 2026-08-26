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


def test_class_summary_endpoint():
    response = client.get("/api/v1/classes/12309/summary")
    assert response.status_code == 200
    assert response.json()["scope"] == "class"


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
