import re
import ast
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from openai import AzureOpenAI
import black
import autopep8

from .schema import Cell, Operation

@dataclass
class SmartChunk:
    """Represents an intelligently created chunk."""
    content: str
    chunk_type: str  # "setup", "load", "reconcile", "extend", "cleanup"
    language: str    # "python", "markdown", "mixed"
    start_line: int
    end_line: int
    metadata: Dict[str, Any]
    confidence: float = 0.0

class EnhancedLLMClassifier:
    """Enhanced LLM classifier with smart chunking and syntax validation."""
    
    def __init__(self, api_key: str, endpoint: str, deployment: str = "gpt-4o-mini"):
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-08-01-preview",
            azure_endpoint=endpoint
        )
        self.deployment = deployment
        self.logger = logging.getLogger(__name__)
    
    def classify(self, cells: List[Cell]) -> List[Operation]:
        """
        Classify cells using enhanced LLM-based intelligent chunking.
        """
        self.logger.info("🤖 Starting enhanced LLM classification...")
        
        # Step 1: Convert notebook to structured markdown
        markdown_content = self._cells_to_structured_markdown(cells)
        
        # Step 2: Use LLM to create intelligent chunks
        smart_chunks = self._create_intelligent_chunks(markdown_content)
        
        # Step 3: Classify and validate chunks
        operations = self._classify_and_validate_chunks(smart_chunks)
        
        # Step 4: Post-process and fix syntax
        operations = self._post_process_operations(operations)
        
        return operations
    
    def _cells_to_structured_markdown(self, cells: List[Cell]) -> str:
        """Convert cells to structured markdown with clear boundaries."""
        content_parts = []
        
        for i, cell in enumerate(cells):
            if not cell.source.strip():
                continue
                
            if cell.kind == "markdown":
                content_parts.append(f"\n## MARKDOWN_CELL_{i}\n{cell.source}\n")
            else:  # code
                content_parts.append(f"\n## CODE_CELL_{i}\n```python\n{cell.source}\n```\n")
        
        return "\n".join(content_parts)
    
    def _create_intelligent_chunks(self, markdown_content: str) -> List[SmartChunk]:
        """Use LLM to create intelligent, contextually-aware chunks."""
        
        prompt = f"""
You are an expert at analyzing Jupyter notebooks for data processing pipelines. 
Your task is to intelligently chunk the following notebook content into logical operations.

Rules:
1. Each chunk should represent ONE complete logical operation
2. Don't split related code blocks (setup + execution should stay together)
3. Identify these operation types: setup, load, reconcile, extend, cleanup
4. For each chunk, provide the complete code needed for that operation
5. Ensure each chunk is syntactically complete and executable

Notebook content:
{markdown_content}

Please analyze this and return a JSON array of chunks with this structure:
[
  {{
    "content": "complete code for this operation",
    "chunk_type": "load|reconcile|extend|setup|cleanup",
    "description": "brief description of what this chunk does",
    "key_variables": ["list", "of", "key", "variables", "created"],
    "dependencies": ["list", "of", "variables", "needed", "from", "previous", "chunks"],
    "confidence": 0.95
  }}
]

Focus on creating 3-5 major chunks that represent the main workflow stages.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "You are an expert Python developer and data scientist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            chunks_data = json.loads(response.choices[0].message.content)
            
            smart_chunks = []
            for i, chunk_data in enumerate(chunks_data.get("chunks", chunks_data)):
                chunk = SmartChunk(
                    content=chunk_data["content"],
                    chunk_type=chunk_data["chunk_type"],
                    language="python",
                    start_line=i * 10,  # Approximate
                    end_line=(i + 1) * 10,
                    metadata={
                        "description": chunk_data.get("description", ""),
                        "key_variables": chunk_data.get("key_variables", []),
                        "dependencies": chunk_data.get("dependencies", []),
                    },
                    confidence=chunk_data.get("confidence", 0.9)
                )
                smart_chunks.append(chunk)
            
            self.logger.info(f"📦 Created {len(smart_chunks)} intelligent chunks")
            return smart_chunks
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create intelligent chunks: {e}")
            # Fallback to basic chunking
            return self._fallback_chunking(markdown_content)
    
    def _fallback_chunking(self, markdown_content: str) -> List[SmartChunk]:
        """Fallback chunking method if LLM fails."""
        chunks = []
        
        # Simple pattern-based chunking as fallback
        sections = markdown_content.split("## CODE_CELL_")
        
        current_chunk = ""
        chunk_type = "setup"
        
        for section in sections[1:]:  # Skip first empty split
            code_match = re.search(r'```python\n(.*?)\n```', section, re.DOTALL)
            if code_match:
                code = code_match.group(1)
                current_chunk += code + "\n\n"
                
                # Determine chunk type
                if any(keyword in code.lower() for keyword in ['import', 'authmanager', 'def get_input']):
                    chunk_type = "setup"
                elif any(keyword in code.lower() for keyword in ['read_csv', 'add_table']):
                    if current_chunk.strip():
                        chunks.append(SmartChunk(
                            content=current_chunk.strip(),
                            chunk_type="load",
                            language="python",
                            start_line=0,
                            end_line=10,
                            metadata={},
                            confidence=0.8
                        ))
                        current_chunk = ""
                elif 'reconcile' in code.lower():
                    chunk_type = "reconcile"
                elif 'extend' in code.lower():
                    chunk_type = "extend"
        
        if current_chunk.strip():
            chunks.append(SmartChunk(
                content=current_chunk.strip(),
                chunk_type=chunk_type,
                language="python",
                start_line=0,
                end_line=10,
                metadata={},
                confidence=0.7
            ))
        
        return chunks
    
    def _classify_and_validate_chunks(self, chunks: List[SmartChunk]) -> List[Operation]:
        """Classify chunks and validate their syntax."""
        operations = []
        
        for i, chunk in enumerate(chunks):
            # Validate and fix syntax
            fixed_code = self._fix_python_syntax(chunk.content)
            
            if fixed_code:
                operation = Operation(
                    op_type=chunk.chunk_type,
                    name=f"{chunk.chunk_type}_{chunk.metadata.get('description', f'operation_{i+1}').replace(' ', '_').lower()}",
                    cells=[],  # We'll store the fixed code in meta
                    meta={
                        "fixed_code": fixed_code,
                        "original_code": chunk.content,
                        "description": chunk.metadata.get("description", ""),
                        "key_variables": chunk.metadata.get("key_variables", []),
                        "dependencies": chunk.metadata.get("dependencies", []),
                        "confidence": chunk.confidence
                    },
                    order=i + 1
                )
                operations.append(operation)
                self.logger.info(f"✅ Created operation: {operation.name} (confidence: {chunk.confidence:.2f})")
            else:
                self.logger.warning(f"⚠️ Skipped chunk due to unfixable syntax errors")
        
        return operations
    
    def _fix_python_syntax(self, code: str) -> Optional[str]:
        """Fix Python syntax errors using multiple approaches."""
        if not code.strip():
            return None
        
        # Step 1: Clean up common notebook artifacts
        cleaned_code = self._clean_notebook_artifacts(code)
        
        # Step 2: Try to parse and validate syntax
        if self._validate_python_syntax(cleaned_code):
            return self._format_code(cleaned_code)
        
        # Step 3: Use LLM to fix syntax errors
        fixed_code = self._llm_fix_syntax(cleaned_code)
        if fixed_code and self._validate_python_syntax(fixed_code):
            return self._format_code(fixed_code)
        
        # Step 4: Try autopep8 for basic fixes
        try:
            autopep8_fixed = autopep8.fix_code(cleaned_code)
            if self._validate_python_syntax(autopep8_fixed):
                return self._format_code(autopep8_fixed)
        except:
            pass
        
        self.logger.warning("❌ Could not fix syntax errors in code block")
        return None
    
    def _clean_notebook_artifacts(self, code: str) -> str:
        """Clean common notebook artifacts from code."""
        # Remove markdown code block markers
        code = re.sub(r'^```python\s*\n?', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*$', '', code, flags=re.MULTILINE)
        
        # Remove In[]: Out[]: markers
        code = re.sub(r'^In\s*\[\d*\]:\s*', '', code, flags=re.MULTILINE)
        code = re.sub(r'^Out\[\d*\]:\s*', '', code, flags=re.MULTILINE)
        
        # Fix common string quote issues
        code = re.sub(r'["""]', '"', code)  # Replace smart quotes
        code = re.sub(r"['']", "'", code)  # Replace smart apostrophes
        
        # Remove extra whitespace but preserve structure
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            cleaned_line = line.rstrip()
            if cleaned_line or (cleaned_lines and cleaned_lines[-1].strip()):
                cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def _validate_python_syntax(self, code: str) -> bool:
        """Validate Python syntax."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
        except Exception:
            return False
    
    def _llm_fix_syntax(self, code: str) -> Optional[str]:
        """Use LLM to fix syntax errors."""
        prompt = f"""
Fix the Python syntax errors in this code. Return ONLY the corrected Python code, no explanations:

```python
{code}
```

Rules:
1. Fix syntax errors while preserving the original logic
2. Ensure all strings are properly quoted
3. Fix indentation issues
4. Remove any markdown artifacts
5. Return only valid Python code
"""

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "You are an expert Python developer. Fix syntax errors in code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            fixed_code = response.choices[0].message.content.strip()
            
            # Remove code block markers if present
            fixed_code = re.sub(r'^```python\s*\n?', '', fixed_code)
            fixed_code = re.sub(r'^```\s*$', '', fixed_code, flags=re.MULTILINE)
            
            return fixed_code.strip()
            
        except Exception as e:
            self.logger.error(f"❌ LLM syntax fix failed: {e}")
            return None
    
    def _format_code(self, code: str) -> str:
        """Format code using black."""
        try:
            formatted = black.format_str(code, mode=black.FileMode())
            return formatted
        except Exception:
            # If black fails, return the code as-is
            return code
    
    def _post_process_operations(self, operations: List[Operation]) -> List[Operation]:
        """Post-process operations to ensure quality and completeness."""
        processed_operations = []
        
        for operation in operations:
            # Skip low-confidence operations
            if operation.meta.get("confidence", 0) < 0.7:
                self.logger.info(f"⚠️ Skipping low-confidence operation: {operation.name}")
                continue
            
            # Skip operations with no useful code
            fixed_code = operation.meta.get("fixed_code", "")
            if len(fixed_code.strip()) < 50:  # Too short to be useful
                self.logger.info(f"⚠️ Skipping operation with insufficient code: {operation.name}")
                continue
            
            # Map chunk types to our standard operation types
            if operation.op_type == "setup":
                operation.op_type = "load"  # Combine setup with load
            elif operation.op_type == "cleanup":
                continue  # Skip cleanup operations
            
            processed_operations.append(operation)
        
        return processed_operations

    def _group_related_operations(self, operations: List[Operation]) -> List[Operation]:
        """Group related operations together to reduce script count."""
        if len(operations) <= 4:  # Already optimal
            return operations
        
        grouped = []
        current_group = None
        
        for operation in operations:
            if operation.op_type == "load":
                # Load operations stay separate
                grouped.append(operation)
            elif operation.op_type in ["reconcile", "extend"]:
                # Group reconcile/extend operations of the same type
                if (current_group and 
                    current_group.op_type == operation.op_type):
                    # Merge into current group
                    current_group.meta["fixed_code"] += "\n\n" + operation.meta["fixed_code"]
                    current_group.name += f"_and_{operation.name}"
                else:
                    # Start new group
                    if current_group:
                        grouped.append(current_group)
                    current_group = operation
            else:
                grouped.append(operation)
        
        if current_group:
            grouped.append(current_group)
        
        return grouped