"""
Enhanced data models for the LLM-powered nb2scripts framework.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

class OperationType(str, Enum):
    """Supported operation types."""
    LOAD = "load"
    RECONCILE = "reconcile"
    EXTEND = "extend"
    UNKNOWN = "unknown"

@dataclass
class Chunk:
    """Represents a chunk of notebook content for LLM processing."""
    text: str
    language: str  # "markdown" | "python" | "mixed"
    number: int
    line_start: int = 0
    line_end: int = 0

@dataclass
class Cell:
    """Represents a single notebook cell."""
    source: str
    kind: str  # "code" | "markdown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Clean up common notebook artifacts
        if self.kind == "code":
            self.source = self._clean_code(self.source)
    
    def _clean_code(self, code: str) -> str:
        """Clean notebook artifacts from code."""
        import re
        # Remove In[]: prefixes
        code = re.sub(r'^In \[\d*\]:\s*', '', code, flags=re.MULTILINE)
        code = re.sub(r'^Out\[\d*\]:\s*', '', code, flags=re.MULTILINE)
        return code.strip()

@dataclass
class Operation:
    """Represents a logical operation (load, reconcile, extend, etc.)."""
    op_type: str  # "load", "reconcile", "extend", etc.
    name: str     # unique identifier for this operation
    cells: List[Cell] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)  # For LLM-based classification
    meta: Dict[str, Any] = field(default_factory=dict)
    order: int = 0
    confidence: float = 1.0  # LLM confidence score
    
    @property
    def code_cells(self) -> List[Cell]:
        """Get only code cells from this operation."""
        return [cell for cell in self.cells if cell.kind == "code"]
    
    @property
    def combined_source(self) -> str:
        """Get all source code combined."""
        if self.chunks:
            # Use chunks if available (LLM-based)
            return "\n\n".join(chunk.text for chunk in self.chunks if chunk.language == "python")
        else:
            # Fallback to cells (rule-based)
            return "\n\n".join(cell.source for cell in self.code_cells if cell.source.strip())

@dataclass
class Script:
    """Represents a generated script file."""
    name: str  # filename without .py extension
    stage: int
    operations: List[Operation] = field(default_factory=list)
    
    @property
    def filename(self) -> str:
        """Get the script filename."""
        return f"{self.name}.py"

@dataclass
class LLMClassificationResult:
    """Result from LLM classification of a chunk."""
    op_type: str
    name: str
    order: int
    meta: Dict[str, Any]
    confidence: float
    reasoning: str = ""  # Optional explanation from LLM