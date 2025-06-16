from typing import Literal

from pydantic import BaseModel, Field


# json schema for employee goals generation output
class EmployeeGoals(BaseModel):
    goals: list[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="List of goals generated for the employee",
    )


# json schema for llm judge goal evaluation output
class ClarityEvaluation(BaseModel):
    score: Literal["Low", "Medium", "High"]
    reason: str


class SpecificityEvaluation(BaseModel):
    score: Literal["Low", "Medium", "High"]
    reason: str


class RoleFitEvaluation(BaseModel):
    score: Literal["No", "Somewhat", "Yes"]
    reason: str


class MeasurabilityEvaluation(BaseModel):
    score: Literal["No", "Somewhat", "Yes"]
    reason: str


class GoalEvaluation(BaseModel):
    clarity: ClarityEvaluation
    specificity: SpecificityEvaluation
    role_fit: RoleFitEvaluation
    measurability: MeasurabilityEvaluation
