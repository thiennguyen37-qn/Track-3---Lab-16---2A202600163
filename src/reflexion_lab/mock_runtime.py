# from __future__ import annotations
# from .schemas import QAExample, JudgeResult, ReflectionEntry
# from .utils import normalize_answer

# FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
# FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}

# def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> tuple[str, int, int]:
#     if example.qid not in FIRST_ATTEMPT_WRONG:
#         return example.gold_answer, 100, 50  # dummy tokens and latency
#     if agent_type == "react":
#         return FIRST_ATTEMPT_WRONG[example.qid], 100, 50
#     if attempt_id == 1 and not reflection_memory:
#         return FIRST_ATTEMPT_WRONG[example.qid], 100, 50
#     return example.gold_answer, 100, 50

# def evaluator(example: QAExample, answer: str) -> tuple[JudgeResult, int, int]:
#     if normalize_answer(example.gold_answer) == normalize_answer(answer):
#         return JudgeResult(score=1, reason="Final answer matches the gold answer after normalization."), 50, 25
#     if normalize_answer(answer) == "london":
#         return JudgeResult(score=0, reason="The answer stopped at the birthplace city and never completed the second hop to the river.", missing_evidence=["Need to identify the river that flows through London."], spurious_claims=[]), 50, 25
#     return JudgeResult(score=0, reason="The final answer selected the wrong second-hop entity.", missing_evidence=["Need to ground the answer in the second paragraph."], spurious_claims=[answer]), 50, 25

# def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> tuple[ReflectionEntry, int, int]:
#     strategy = "Do the second hop explicitly: birthplace city -> river through that city." if example.qid == "hp2" else "Verify the final entity against the second paragraph before answering."
#     return ReflectionEntry(attempt_id=attempt_id, failure_reason=judge.reason, lesson="A partial first-hop answer is not enough; the final answer must complete all hops.", next_strategy=strategy), 75, 30



from __future__ import annotations
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM

import json
import os
import re
import time
from typing import Tuple
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["NVIDIA_API_KEY"],
    base_url="https://integrate.api.nvidia.com/v1"
)

MODEL_NAME = "openai/gpt-oss-120b"
_MAX_RETRIES = 2
_RETRY_DELAY = 1  # seconds


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from a model response string."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*({[\s\S]+?})\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r"{[\s\S]+}", text)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"No JSON object found in response: {text[:200]}")


def _call_with_retry(messages: list, temperature: float) -> tuple:
    """Call LLM with retry. Returns (content, tokens, latency_ms)."""
    for attempt in range(_MAX_RETRIES):
        try:
            start = time.time()

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
            )

            latency_ms = int((time.time() - start) * 1000)
            tokens = response.usage.total_tokens if response.usage else 0
            content = response.choices[0].message.content.strip()

            return content, tokens, latency_ms

        except Exception as exc:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY * (attempt + 1))
            else:
                raise exc


def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str]
) -> Tuple[str, int, int]:
    """
    Gọi LLM để lấy câu trả lời từ Actor Agent.
    """

    # 🔧 FIX 1: limit context để tránh overflow token
    MAX_CHARS = 4000
    context_str = "\n\n".join(
        [f"--- {c.title} ---\n{c.text}" for c in example.context]
    )[:MAX_CHARS]

    # 🔧 FIX 2: inject reflection memory rõ ràng hơn
    reflections_str = ""
    if reflection_memory:
        reflections_str = "\n\nPAST REFLECTIONS:\n" + "\n".join(reflection_memory)

    user_prompt = f"""
Context:
{context_str}
{reflections_str}

Question: {example.question}

IMPORTANT:
- Answer concisely
- Output ONLY the final answer (no explanation)
"""

    messages = [
        {"role": "system", "content": ACTOR_SYSTEM},
        {"role": "user", "content": user_prompt}
    ]

    # 🔧 FIX 3: dùng retry (trước đó bạn gọi trực tiếp API)
    content, tokens, latency_ms = _call_with_retry(messages, temperature=0.1)

    return content.strip(), tokens, latency_ms


def evaluator(example: QAExample, answer: str) -> Tuple[JudgeResult, int, int]:
    """
    Gọi LLM (Evaluator) để chấm điểm câu trả lời.
    """

    user_prompt = (
        f"Question: {example.question}\n"
        f"Gold Answer: {example.gold_answer}\n"
        f"Predicted Answer: {answer}\n\n"
        "Return JSON with keys: score (0 or 1), reason"
    )

    messages = [
        {"role": "system", "content": EVALUATOR_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    content, tokens, latency_ms = _call_with_retry(messages, temperature=0.0)

    # 🔧 FIX 4: handle JSON lỗi
    try:
        data = _extract_json(content)
    except Exception:
        return JudgeResult(score=0, reason="Invalid JSON from evaluator"), tokens, latency_ms

    score = int(data.get("score", 0))
    reason = data.get("reason", "")

    return JudgeResult(score=score, reason=reason), tokens, latency_ms


def reflector(
    example: QAExample,
    attempt_id: int,
    judge: JudgeResult
) -> Tuple[ReflectionEntry, int, int]:
    """
    Gọi LLM (Reflector) để phân tích lỗi sai.
    """

    # 🔧 FIX 5: limit context
    context_str = "\n".join(
        [f"Title: {c.title}\nText: {c.text}" for c in example.context]
    )[:3000]

    user_prompt = (
        f"Context provided to Actor:\n{context_str}\n\n"
        f"Question: {example.question}\n"
        f"Gold Answer: {example.gold_answer}\n"
        f"Failed Attempt ID: {attempt_id}\n"
        f"Evaluator's Feedback: {judge.reason} (Score: {judge.score})\n\n"
        "Return JSON with keys: lesson, next_strategy"
    )

    messages = [
        {"role": "system", "content": REFLECTOR_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    content, tokens, latency_ms = _call_with_retry(messages, temperature=0.2)

    # 🔧 FIX 6: robust parsing
    try:
        data = _extract_json(content)
    except Exception:
        data = {"lesson": "Parsing failed", "next_strategy": "Be more precise"}

    return ReflectionEntry(
        attempt_id=attempt_id,  # 🔧 FIX 7: force đúng attempt_id
        failure_reason=judge.reason,
        lesson=data.get("lesson", ""),
        next_strategy=data.get("next_strategy", ""),
    ), tokens, latency_ms
