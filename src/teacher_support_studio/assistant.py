"""LangGraph workflow for grounded, teacher-friendly responses."""

from __future__ import annotations

import json
import os
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from teacher_support_studio.analytics import AnalyticsService
from teacher_support_studio.schemas import (
    ChatRequest,
    ChatResponse,
    DashboardSummary,
    TeacherAnswer,
)


class AssistantState(TypedDict, total=False):
    request: ChatRequest
    summary: DashboardSummary
    answer: TeacherAnswer
    response_mode: str


SYSTEM_PROMPT = """You are Teacher Support Studio, a concise teacher-facing assistant.
Use only the supplied evidence. Never invent a number, learner characteristic, diagnosis,
mastery judgment, grade, placement, or mandatory intervention. Describe patterns as prompts
for teacher observation. Keep each section to 1-3 short sentences. Suggested actions must be
low-stakes, practical, and preserve teacher judgment. If evidence is sparse, say so plainly.
Do not mention SHAP, thresholds, algorithms, or data-science jargon unless directly asked.
"""

PROHIBITED_PHRASES = (
    "learning disability",
    "has mastered",
    "has not mastered",
    "must be placed",
    "should be retained",
)


class TeacherAssistant:
    def __init__(self, analytics: AnalyticsService) -> None:
        self.analytics = analytics
        self.graph = self._build_graph()

    def ask(self, request: ChatRequest) -> ChatResponse:
        state = self.graph.invoke({"request": request})
        answer = state["answer"]
        return ChatResponse(**answer.model_dump(), response_mode=state["response_mode"])

    def _build_graph(self):
        builder = StateGraph(AssistantState)
        builder.add_node("gather_evidence", self._gather_evidence)
        builder.add_node("openai_response", self._openai_response)
        builder.add_node("guided_response", self._guided_response)
        builder.add_node("validate", self._validate)
        builder.add_edge(START, "gather_evidence")
        builder.add_conditional_edges(
            "gather_evidence",
            self._route_generation,
            {"openai": "openai_response", "guided": "guided_response"},
        )
        builder.add_edge("openai_response", "validate")
        builder.add_edge("guided_response", "validate")
        builder.add_edge("validate", END)
        return builder.compile()

    def _gather_evidence(self, state: AssistantState) -> AssistantState:
        request = state["request"]
        if request.scope == "class":
            summary = self.analytics.class_summary(request.class_id)
        else:
            if request.student_id is None:
                raise ValueError("student_id is required for student focus")
            summary = self.analytics.student_summary(request.class_id, request.student_id)
        return {"summary": summary}

    @staticmethod
    def _route_generation(_: AssistantState) -> str:
        return "openai" if os.getenv("OPENAI_API_KEY", "").strip() else "guided"

    def _openai_response(self, state: AssistantState) -> AssistantState:
        request = state["request"]
        summary = state["summary"]
        evidence_packet = {
            "scope": summary.scope,
            "selected_entity": summary.entity_label,
            "headline": summary.headline,
            "metrics": [card.model_dump() for card in summary.cards],
            "skill_metrics": [skill.model_dump() for skill in summary.skills],
            "evidence": summary.evidence,
        }
        prompt = (
            f"Teacher question: {request.question}\n\n"
            f"Approved evidence packet:\n{json.dumps(evidence_packet, indent=2)}"
        )
        try:
            llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
                api_key=os.environ["OPENAI_API_KEY"],
                max_tokens=500,
                temperature=0.2,
                use_responses_api=True,
            ).with_structured_output(TeacherAnswer)
            answer = llm.invoke(
                [("system", SYSTEM_PROMPT), ("human", prompt)],
                config={"tags": ["teacher-support-studio", summary.scope]},
            )
            return {"answer": answer, "response_mode": "openai"}
        except Exception:
            return {
                "answer": self._guided_answer(request.question, summary),
                "response_mode": "guided",
            }

    def _guided_response(self, state: AssistantState) -> AssistantState:
        return {
            "answer": self._guided_answer(state["request"].question, state["summary"]),
            "response_mode": "guided",
        }

    def _validate(self, state: AssistantState) -> AssistantState:
        answer = state["answer"]
        combined = " ".join(
            [answer.what_i_noticed, answer.what_you_might_try, answer.what_to_keep_in_mind]
        ).lower()
        if any(phrase in combined for phrase in PROHIBITED_PHRASES):
            return {
                "answer": self._guided_answer(state["request"].question, state["summary"]),
                "response_mode": "guided",
            }
        answer.supporting_evidence = state["summary"].evidence[:4]
        return {"answer": answer}

    @staticmethod
    def _guided_answer(question: str, summary: DashboardSummary) -> TeacherAnswer:
        lowered = question.lower()
        skills = summary.skills
        lowest = min(skills, key=lambda item: item.success_rate) if skills else None
        improving = max(
            (skill for skill in skills if skill.change_points is not None),
            key=lambda item: item.change_points or 0,
            default=None,
        )

        if "progress" in lowered or "improv" in lowered:
            noticed = (
                f"{improving.label} shows the strongest recent improvement "
                "among the displayed skills."
                if improving
                else "The available interactions do not yet show a clear improvement pattern."
            )
            action = (
                "Acknowledge the progress and use one short retrieval question to see "
                "whether it holds without extra support."
            )
        elif "warm-up" in lowered or "warmup" in lowered or "review" in lowered:
            noticed = (
                f"{lowest.label} is the clearest skill to review among the recent activity shown."
                if lowest
                else "The recent activity does not identify one clear review priority."
            )
            action = (
                "Try a 10-minute sequence: one worked visual example, one problem solved together, "
                "and one independent check for understanding."
            )
        elif "question" in lowered or "ask" in lowered:
            noticed = (
                "A brief conversation can help distinguish a procedural slip from a "
                "conceptual misunderstanding."
            )
            action = (
                "Ask: ‘Can you show me how you started?’, ‘What does this step mean?’, and "
                "‘What would you try if the numbers changed?’"
            )
        elif "check-in" in lowered or "check in" in lowered or "need help" in lowered:
            noticed = (
                f"The recent pattern makes {lowest.label} a reasonable place for a quick check-in."
                if lowest
                else "The available evidence does not identify a single check-in priority."
            )
            action = (
                "Use one low-stakes example and ask the learner to explain their thinking "
                "before choosing additional practice."
            )
        else:
            noticed = summary.headline + "."
            action = (
                "Use a short formative check to confirm the pattern, then adjust instruction "
                "based on what students explain and produce."
            )

        return TeacherAnswer(
            what_i_noticed=noticed,
            what_you_might_try=action,
            what_to_keep_in_mind=(
                "These are patterns in recorded practice, not conclusions about ability or "
                "mastery. "
                "Use them alongside classroom observation and student work."
            ),
            supporting_evidence=summary.evidence[:4],
        )
