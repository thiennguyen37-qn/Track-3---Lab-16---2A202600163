from __future__ import annotations
from typing import Literal, Optional, TypedDict
from pydantic import BaseModel, Field

class ContextChunk(BaseModel):
    title: str
    text: str

class QAExample(BaseModel):
    qid: str
    difficulty: Literal["easy", "medium", "hard"]
    question: str
    gold_answer: str
    context: list[ContextChunk]

from typing import Optional
from pydantic import BaseModel, Field

class JudgeResult(BaseModel):
    """Kết quả đánh giá từ Evaluator (Judge)."""
    score: int = Field(..., description="Điểm số đánh giá câu trả lời (ví dụ: 0 là sai, 1 là đúng, hoặc thang điểm 1-10).")
    reason: str = Field(..., description="Giải thích chi tiết lý do tại sao lại chấm mức điểm này.")
    

class ReflectionEntry(BaseModel):
    """Nội dung Reflection sau một lần trả lời sai."""
    attempt_id: int = Field(..., description="ID của lần thử (attempt) đang được phân tích.")
    failure_reason: str = Field(..., description="Nguyên nhân cụ thể khiến câu trả lời hoặc cách tiếp cận trước đó bị sai.")
    lesson: str = Field(..., description="Bài học tổng quát rút ra từ sai lầm (failure_reason) đó.")
    next_strategy: str = Field(..., description="Chiến lược chi tiết và các bước hành động để làm tốt hơn trong lần thử tiếp theo.")

class AttemptTrace(BaseModel):
    attempt_id: int
    answer: str
    score: int
    reason: str
    reflection: Optional[ReflectionEntry] = None
    token_estimate: int = 0
    latency_ms: int = 0

class RunRecord(BaseModel):
    qid: str
    question: str
    gold_answer: str
    agent_type: Literal["react", "reflexion"]
    predicted_answer: str
    is_correct: bool
    attempts: int
    token_estimate: int
    latency_ms: int
    failure_mode: Literal["none", "entity_drift", "incomplete_multi_hop", "wrong_final_answer", "looping", "reflection_overfit"]
    reflections: list[ReflectionEntry] = Field(default_factory=list)
    traces: list[AttemptTrace] = Field(default_factory=list)

class ReportPayload(BaseModel):
    meta: dict
    summary: dict
    failure_modes: dict
    examples: list[dict]
    extensions: list[str]
    discussion: str

class ReflexionState(TypedDict):
    question: str
    context: list[str]
    trajectory: list[str]
    reflection_memory: list[str]
    attempt_count: int
    success: bool
    final_answer: str
