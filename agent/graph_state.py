"""
This file contains the GraphState class, which is used to save and pass information between nodes.   
"""

from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from typing import Optional, List

class GraphState(BaseModel):
    """Graph State to save and pass information between nodes"""
    messages: List[BaseMessage]  # The human and agent messages
    question: Optional[str] = None  # The user question
    category: Optional[str] = None  # The category of the question
