"""
Multi-seed aggregation over already-computed per-run metrics - never recomputes ground truth or reopens per-run evaluation
(evaluation/offline/). The primary statistical unit is a RUN (scenario x seed x pipeline configuration) - 
summarized first WITHIN each run (already done by evaluate_run.py), only then aggregated across runs.
"""
