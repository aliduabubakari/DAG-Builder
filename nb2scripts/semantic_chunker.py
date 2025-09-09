"""
Semantic chunking based on operation boundaries.
"""
import re
from typing import List
from .schema import Chunk

class SemanticChunker:
    """Chunks notebook content based on semantic operation boundaries."""
    
    # Operation boundary patterns
    OPERATION_PATTERNS = {
        'load_start': [
            r'# Load a dataset into a DataFrame',
            r'import pandas as pd',
            r'# This is a base file',
            r'from semt_py import'
        ],
        'reconcile_start': [
            r'# Reconciliation operation for column',
            r'## Reconciliation operation for column',
            r'reconciliator_id\s*=',
            r'reconciliation_manager\.reconcile'
        ],
        'extend_start': [
            r'# Extension operation for column',
            r'## Extension operation for column',
            r'extension_manager\.extend_column',
            r'extender_id\s*='
        ]
    }
    
    @classmethod
    def chunk_by_operations(cls, markdown_content: str) -> List[Chunk]:
        """
        Chunk markdown content based on semantic operation boundaries.
        
        Args:
            markdown_content: Full markdown content
            
        Returns:
            List of semantically meaningful chunks
        """
        lines = markdown_content.split('\n')
        chunks = []
        
        # Find operation boundaries
        boundaries = cls._find_operation_boundaries(lines)
        
        # Create chunks based on boundaries
        for i, (start_line, op_type, description) in enumerate(boundaries):
            # Determine end line (start of next operation or end of file)
            end_line = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(lines)
            
            # Extract chunk content
            chunk_lines = lines[start_line:end_line]
            chunk_text = '\n'.join(chunk_lines).strip()
            
            # Skip empty chunks
            if not chunk_text:
                continue
            
            # Determine language based on content
            language = cls._determine_chunk_language(chunk_text)
            
            chunk = Chunk(
                text=chunk_text,
                language=language,
                number=len(chunks) + 1,
                line_start=start_line,
                line_end=end_line - 1
            )
            chunks.append(chunk)
            
            print(f"📦 Created semantic chunk {chunk.number}: {op_type} ({end_line - start_line} lines)")
        
        return chunks
    
    @classmethod
    def _find_operation_boundaries(cls, lines: List[str]) -> List[tuple]:
        """Find lines where operations start."""
        boundaries = []
        
        for i, line in enumerate(lines):
            line_clean = line.strip().lower()
            
            # Check for load operation start
            if cls._matches_patterns(line, cls.OPERATION_PATTERNS['load_start']):
                if not boundaries:  # First load operation
                    boundaries.append((max(0, i - 2), 'load', 'Load and setup operations'))
                continue
            
            # Check for reconciliation operation start
            if cls._matches_patterns(line, cls.OPERATION_PATTERNS['reconcile_start']):
                # Determine which reconciliation this is
                context = ' '.join(lines[i:i+10]).lower()
                if 'point of interest' in context and 'wikidataalligator' in context:
                    op_desc = 'Reconcile Point of Interest'
                elif 'place' in context and 'wikidataopenrefine' in context:
                    op_desc = 'Reconcile Place'
                else:
                    op_desc = 'Reconciliation operation'
                
                boundaries.append((i, 'reconcile', op_desc))
                continue
            
            # Check for extension operation start
            if cls._matches_patterns(line, cls.OPERATION_PATTERNS['extend_start']):
                boundaries.append((i, 'extend', 'Extension operation'))
                continue
        
        # Ensure we have at least one boundary (start of file)
        if not boundaries:
            boundaries.append((0, 'unknown', 'Unknown operations'))
        
        return boundaries
    
    @classmethod
    def _matches_patterns(cls, line: str, patterns: List[str]) -> bool:
        """Check if line matches any of the given patterns."""
        line_lower = line.lower()
        for pattern in patterns:
            if re.search(pattern.lower(), line_lower):
                return True
        return False
    
    @classmethod
    def _determine_chunk_language(cls, chunk_text: str) -> str:
        """Determine the primary language of a chunk."""
        lines = chunk_text.split('\n')
        code_lines = 0
        markdown_lines = 0
        in_code_block = False
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
                
            if stripped.startswith('```'):
                in_code_block = not in_code_block
                continue
            
            if in_code_block or (not stripped.startswith('#') and 
                                ('=' in line or 'import ' in line or 'def ' in line or 
                                 'try:' in line or 'except' in line)):
                code_lines += 1
            else:
                markdown_lines += 1
        
        if code_lines > markdown_lines:
            return 'python'
        elif markdown_lines > 0:
            return 'mixed'
        else:
            return 'python'