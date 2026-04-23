# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_mini.json
- Mode: mock
- Records: 200
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.98 | 1.0 | 0.02 |
| Avg attempts | 1 | 1.01 | 0.01 |
| Avg token estimate | 710.46 | 724.78 | 14.32 |
| Avg latency (ms) | 2067.16 | 2403.73 | 336.57 |

## Failure modes
```json
{
  "react": {
    "none": 98,
    "wrong_final_answer": 2
  },
  "reflexion": {
    "none": 100
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding

## Discussion
Reflexion helps when the first attempt stops after the first hop or drifts to a wrong second-hop entity. The tradeoff is higher attempts, token cost, and latency. In a real report, students should explain when the reflection memory was useful, which failure modes remained, and whether evaluator quality limited gains.
