from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from openenv.core.env_server.interfaces import Observation as BaseObservation, Action as BaseAction


class Action(BaseAction):
    # The type of action the agent wants to take — must be one of the 5 valid action types
    action_type: str = Field(
        description="Must be one of: ask_question, request_document, approve_scheme, reject_applicant, escalate"
    )
    # The argument for the action — field name, document name, scheme name, or rejection reason
    value: Optional[str] = Field(
        None,
        description="The specific question field, document name, scheme name, or reason."
    )


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