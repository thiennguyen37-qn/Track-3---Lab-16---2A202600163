from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

# Import các hàm đã được kết nối với API LLM thật từ runtime.py
from .mock_runtime import actor_answer, evaluator, reflector
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0

        for attempt_id in range(1, self.max_attempts + 1):
            # 1. Gọi Actor (nhận về câu trả lời, token và độ trễ)
            answer, actor_tokens, actor_latency = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            
            # 2. Gọi Evaluator (nhận về kết quả chấm điểm, token và độ trễ)
            judge, eval_tokens, eval_latency = evaluator(example, answer)

            # 3. Tính tổng token và latency cho bước hiện tại (chưa tính reflection)
            step_tokens = actor_tokens + eval_tokens
            step_latency = actor_latency + eval_latency

            # Khởi tạo trace tạm thời
            trace = AttemptTrace(
                attempt_id=attempt_id, 
                answer=answer, 
                score=judge.score, 
                reason=judge.reason, 
                token_estimate=step_tokens, 
                latency_ms=step_latency
            )
            
            final_answer = answer
            final_score = judge.score

            # Nếu Evaluator chấm đúng (score == 1), kết thúc sớm vòng lặp
            if judge.score == 1:
                traces.append(trace)
                break
            
            # --- LOGIC REFLEXION ---
            # 1. Kiểm tra nếu agent_type là 'reflexion' và chưa đến lượt attempt cuối cùng
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                
                # 2. Gọi hàm reflector để phân tích lỗi và lấy chiến thuật mới
                reflection_entry, ref_tokens, ref_latency = reflector(example, attempt_id, judge)
                
                # Cập nhật thêm token và latency của Reflector vào trace này
                trace.token_estimate += ref_tokens
                trace.latency_ms += ref_latency
                trace.reflection = reflection_entry
                
                # Lưu vào danh sách tổng
                reflections.append(reflection_entry)
                
                # 3. Cập nhật reflection_memory để Actor dùng cho vòng lặp tiếp theo
                memory_str = (
                    f"[Attempt {attempt_id} Failed]\n"
                    f"- Mistake: {reflection_entry.failure_reason}\n"
                    f"- Lesson Learned: {reflection_entry.lesson}\n"
                    f"- Action Plan for Next Attempt: {reflection_entry.next_strategy}"
                )
                reflection_memory.append(memory_str)

            # Lưu lại lịch sử của attempt này
            traces.append(trace)

        # Tổng hợp dữ liệu cho RunRecord
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        
        # Với LLM thật, ta không thể biết trước failure mode là gì nếu không có 1 LLM khác phân loại.
        # Nên nếu sai, ta chỉ việc gán "wrong_final_answer" là hợp lý và tuân thủ đúng Schema.
        failure_mode = "none" if final_score == 1 else "wrong_final_answer"

        return RunRecord(
            qid=example.qid, 
            question=example.question, 
            gold_answer=example.gold_answer, 
            agent_type=self.agent_type, 
            predicted_answer=final_answer, 
            is_correct=bool(final_score), 
            attempts=len(traces), 
            token_estimate=total_tokens, 
            latency_ms=total_latency, 
            failure_mode=failure_mode, 
            reflections=reflections, 
            traces=traces
        )

class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)