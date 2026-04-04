from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from openenv.core.env_server.interfaces import Observation as BaseObservation, Action as BaseAction

class Action(BaseAction):
    action_type: str = Field(description="Must be one of: ask_question, request_document, approve_scheme, reject_applicant, escalate")
    value: Optional[str] = Field(None, description="The specific question, document, scheme, or reason.")

class Observation(BaseObservation):
    known_profile: Dict[str, Any] = Field(default_factory=dict)
    missing_data: List[str] = Field(default_factory=list)
    notification: Optional[str] = Field(None)
    is_terminated: bool = Field(False)