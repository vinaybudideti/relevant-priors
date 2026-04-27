from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class StudyIn(BaseModel):
    model_config = ConfigDict(extra='ignore')
    study_id: str
    study_description: str
    study_date: str  # NOT date type — must accept malformed dates


class CaseIn(BaseModel):
    model_config = ConfigDict(extra='ignore')
    case_id: str
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    current_study: StudyIn
    prior_studies: List[StudyIn] = Field(default_factory=list)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra='ignore')
    challenge_id: Optional[str] = None
    schema_version: Optional[int] = None
    generated_at: Optional[str] = None
    cases: List[CaseIn] = Field(default_factory=list)


class Prediction(BaseModel):
    case_id: str
    study_id: str
    predicted_is_relevant: bool


class PredictResponse(BaseModel):
    predictions: List[Prediction]
