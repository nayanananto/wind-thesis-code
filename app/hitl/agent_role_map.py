AGENT_ROLE_MAP = {
    "Forecast": {
        "role": "semantic_phase_forecast_producer",
        "purpose": "Predict the next semantic wind regime from compressed live wind-state sequences.",
        "hitl_use": "Human reviews phase probabilities, regime labels, and transition evidence.",
        "required_for_core_result": True,
    },
    "Energy": {
        "role": "derived_metric_producer",
        "purpose": "Map forecast wind speed to downstream energy estimates.",
        "hitl_use": "Human treats this as a downstream view, not the main thesis forecast contribution.",
        "required_for_core_result": False,
    },
    "SummarizerRAG": {
        "role": "forecast_explainer",
        "purpose": "Turn forecast outputs and semantic states into human-readable explanations.",
        "hitl_use": "Human checks whether the explanation matches the numeric evidence and semantic state.",
        "required_for_core_result": False,
    },
    "WhatIf": {
        "role": "scenario_reasoner",
        "purpose": "Handle counterfactual and conceptual questions around forecast states and transitions.",
        "hitl_use": "Human validates whether the scenario assumptions are realistic before accepting the answer.",
        "required_for_core_result": False,
    },
    "Clarifier": {
        "role": "approval_gate",
        "purpose": "Ask follow-up questions when the user request is ambiguous or underspecified.",
        "hitl_use": "Human supplies missing constraints before the system continues.",
        "required_for_core_result": True,
    },
    "Chat": {
        "role": "freeform_interface",
        "purpose": "Handle open-ended interaction that does not require a structured forecast action.",
        "hitl_use": "Human uses this to explore the system without changing the forecasting core.",
        "required_for_core_result": False,
    },
    "Plotter": {
        "role": "inspection_surface",
        "purpose": "Provide visual inspection of forecasts or derived energy series.",
        "hitl_use": "Human visually checks anomalies, ramps, or suspicious transitions.",
        "required_for_core_result": False,
    },
    "TimeToEnergy": {
        "role": "goal_query",
        "purpose": "Answer planning-style questions over derived energy trajectories.",
        "hitl_use": "Human treats this as an interactive convenience layer.",
        "required_for_core_result": False,
    },
}


HITL_REVIEW_STAGES = [
    {
        "stage": "Clarify",
        "goal": "Ask for missing horizon, scenario details, or assumptions before executing.",
    },
    {
        "stage": "Forecast",
        "goal": "Run semantic phase prediction and produce reviewable transition candidates.",
    },
    {
        "stage": "Review",
        "goal": "Attach current regime, nearest historical analogs, and supporting evidence.",
    },
    {
        "stage": "Explain",
        "goal": "Use the LLM/explainer layer to describe the result in natural language.",
    },
    {
        "stage": "Feedback",
        "goal": "Let the human accept, flag, relabel, or annotate the result for later analysis.",
    },
]
