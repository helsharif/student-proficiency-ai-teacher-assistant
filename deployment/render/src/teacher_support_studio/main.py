"""FastAPI entrypoint for Teacher Support Studio."""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from teacher_support_studio import __version__
from teacher_support_studio.analytics import AnalyticsService
from teacher_support_studio.assistant import TeacherAssistant
from teacher_support_studio.schemas import ChatRequest, ChatResponse, DashboardSummary, EntityOption
from teacher_support_studio.skill_emoji_mapping import SkillEmojiMappingService

STATIC_DIR = Path(__file__).resolve().parent / "static"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

app = FastAPI(
    title="Teacher Support Studio API",
    description=(
        "Teacher-facing summaries and grounded support responses for the "
        "ASSISTments portfolio project."
    ),
    version=__version__,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

analytics = AnalyticsService()
assistant = TeacherAssistant(analytics)
skill_emojis = SkillEmojiMappingService()


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/api/v1/skill-emojis", response_model=dict[str, str])
def list_skill_emojis() -> dict[str, str]:
    return skill_emojis.mapping()


@app.get("/api/v1/classes", response_model=list[EntityOption])
def list_classes() -> list[EntityOption]:
    return analytics.class_options()


@app.get("/api/v1/classes/{class_id}/students", response_model=list[EntityOption])
def list_students(class_id: int) -> list[EntityOption]:
    try:
        return analytics.student_options(class_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/classes/{class_id}/summary", response_model=DashboardSummary)
def class_summary(class_id: int) -> DashboardSummary:
    try:
        return analytics.class_summary(class_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get(
    "/api/v1/classes/{class_id}/students/{student_id}/summary",
    response_model=DashboardSummary,
)
def student_summary(class_id: int, student_id: int) -> DashboardSummary:
    try:
        return analytics.student_summary(class_id, student_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        return assistant.ask(request)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
