#models_api.py = request/response schemas

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k:int=Field(default=6, ge=1, le=30)
    #optional filters for later (doc_id, source, tenant_id, etc.,)
    filters: Optional[Dict[str,Any]] = None

class SourceItem(BaseModel):
    chunk_id: str
    doc_id: str
    title: Optional[str]=None
    source_type: Optional[str]=None
    source_path:Optional[str]=None
    score: float

class AskResponse(BaseModel):
    answer:str
    sources:List[SourceItem]
    latency_ms:int
    refused:bool=False


