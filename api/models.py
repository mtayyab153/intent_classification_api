from pydantic import BaseModel
from typing import List

class SingleQuery(BaseModel):
    text: str

class BatchQuery(BaseModel):
    texts: List[str]

class ClassificationResult(BaseModel):
    intent: str
    confidence: float
