"""API schemas for Teacher Support Studio."""

from typing import Literal

from pydantic import BaseModel, Field


class EntityOption(BaseModel):
    id: int
    label: str
    detail: str


class MetricCard(BaseModel):
    label: str
    value: str
    detail: str
    tone: Literal["neutral", "positive", "attention"] = "neutral"


class SkillMetric(BaseModel):
    label: str
    success_rate: float
    interactions: int
    students: int | None = None
    change_points: float | None = None


class TrendPoint(BaseModel):
    label: str
    success_rate: float
    interactions: int


class DashboardSummary(BaseModel):
    scope: Literal["class", "student"]
    entity_id: int
    entity_label: str
    context_label: str
    headline: str
    cards: list[MetricCard]
    skills: list[SkillMetric]
    trend: list[TrendPoint]
    suggested_questions: list[str]
    evidence: list[str]


class ChatRequest(BaseModel):
    scope: Literal["class", "student"]
    class_id: int
    student_id: int | None = None
    question: str = Field(min_length=2, max_length=500)


class TeacherAnswer(BaseModel):
    what_i_noticed: str
    what_you_might_try: str
    what_to_keep_in_mind: str
    supporting_evidence: list[str] = Field(default_factory=list, max_length=4)


class ChatResponse(TeacherAnswer):
    response_mode: Literal["openai", "guided"]
