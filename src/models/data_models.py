from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import numpy as np

class AnalysisRequest(BaseModel):
    """Analiz isteği modeli."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    data: List[List[float]]
    method: str = Field(default="static")

class AnalysisResponse(BaseModel):
    """Analiz yanıt modeli."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    status: str
    analysis_id: Optional[str] = None
    error: Optional[str] = None
    results: Optional[Dict[str, Any]] = None 