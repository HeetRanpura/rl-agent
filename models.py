from pydantic import BaseModel, Field, model_validator
from typing import Optional, Dict, List, Any, Literal
from openenv.core.env_server.interfaces import Observation as BaseObservation, Action as BaseAction


class Action(BaseAction):
    # The type of action the agent wants to take — must be one of the 5 valid action types
    action_type: Literal["ask_question", "request_document", "approve_scheme", "reject_applicant", "escalate"] = Field(
        description="Must be one of: ask_question, request_document, approve_scheme, reject_applicant, escalate"
    )
    # The argument for the action — field name, document name, scheme name, or rejection reason
    value: Optional[str] = Field(
        None,
        description=(
            "For ask_question use: age, income, occupation, has_aadhaar. "
            "For request_document use: aadhaar_card, pan_card, aadhaar, pan. "
            "For approve_scheme use: PMKVY, MGNREGS, PMAY. "
            "For reject_applicant or escalate use a concise category such as "
            "AGE_EXCEEDED, INCOME_TOO_HIGH, NO_ELIGIBLE_SCHEME, "
            "MISSING_REQUIRED_DATA, DATA_MISMATCH, DOCUMENT_CONFLICT, "
            "or MANUAL_REVIEW_REQUIRED."
        )
    )

    @model_validator(mode="after")
    def validate_value(self) -> "Action":
        value = (self.value or "").strip()

        if self.action_type == "ask_question":
            allowed = ("age", "income", "occupation", "has_aadhaar")
            if value not in allowed:
                raise ValueError(f"ask_question value must be one of {allowed}")
        elif self.action_type == "request_document":
            allowed = ("aadhaar_card", "pan_card", "aadhaar", "pan")
            if value.lower() not in allowed:
                raise ValueError(f"request_document value must be one of {allowed}")
        elif self.action_type == "approve_scheme":
            allowed = ("PMKVY", "MGNREGS", "PMAY")
            if value not in allowed:
                raise ValueError(f"approve_scheme value must be one of {allowed}")
        elif self.action_type in {"reject_applicant", "escalate"}:
            allowed = (
                "",
                "AGE_EXCEEDED",
                "INCOME_TOO_HIGH",
                "NO_ELIGIBLE_SCHEME",
                "MISSING_REQUIRED_DATA",
                "DATA_MISMATCH",
                "DOCUMENT_CONFLICT",
                "MANUAL_REVIEW_REQUIRED",
            )
            if value not in allowed:
                raise ValueError(f"{self.action_type} value must be one of {allowed}")

        return self


class Observation(BaseObservation):
    # Applicant data collected so far — grows as agent asks valid questions
    known_profile: Dict[str, Any] = Field(default_factory=dict)

    # Fields still needed before the agent can make a terminal decision
    missing_data: List[str] = Field(default_factory=list)

    # Feedback from the environment about the last action taken
    notification: Optional[str] = Field(None)

    # True when the episode has ended — agent must call reset() to start a new one
    is_terminated: bool = Field(False)

    # Continuous grader score between 0.0 and 1.0 — set only when episode terminates
    grader_score: Optional[float] = Field(None)

    # Internal episode tracking — noise query count, redundant query count, task id
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentObservation(BaseObservation):
    known_profile: Dict[str, Any] = Field(default_factory=dict)
    missing_data: List[str] = Field(default_factory=list)
    notification: Optional[str] = Field(None)
    is_terminated: bool = Field(False)
    grader_score: Optional[float] = Field(None)
