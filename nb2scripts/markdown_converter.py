"""
Convert Jupyter notebooks to markdown using nbconvert.
"""
import subprocess
import tempfile
import os
from pathlib import Path
from typing import List
from .schema import Chunk

class NotebookToMarkdownConverter:
    """Converts Jupyter notebooks to markdown using nbconvert."""
    
    @staticmethod
    def convert_notebook_to_markdown(notebook_path: str) -> str:
        """
        Convert a Jupyter notebook to markdown using nbconvert.
        
        Args:
            notebook_path: Path to the .ipynb file
            
        Returns:
            Markdown content as string
        """
        if not os.path.exists(notebook_path):
            raise FileNotFoundError(f"Notebook not found: {notebook_path}")
        
        if not notebook_path.endswith('.ipynb'):
            raise ValueError("File must be a Jupyter notebook (.ipynb)")
        
        # Create temporary file for markdown output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as temp_file:
            temp_md_path = temp_file.name
        
        try:
            # Run nbconvert
            cmd = [
                'jupyter', 'nbconvert',
                '--to', 'markdown',
                '--output', temp_md_path,
                notebook_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Read the generated markdown
            with open(temp_md_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            return markdown_content
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"nbconvert failed: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError("nbconvert not found. Please install with: pip install nbconvert")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_md_path):
                os.unlink(temp_md_path)
    
    @staticmethod
    def chunk_markdown(markdown_content: str, max_lines_per_chunk: int = 50) -> List[Chunk]:  # Reduced from 100
        """
        Split markdown content into manageable chunks for LLM processing.
        Uses smarter boundaries to avoid splitting operations.
        """
        lines = markdown_content.split('\n')
        chunks = []
        current_chunk_lines = []
        current_language = "markdown"
        chunk_number = 1
        line_start = 0
        
        in_code_block = False
        code_language = ""
        
        # Operation boundary indicators
        operation_boundaries = [
            'reconciliation operation',
            'extension operation',
            'import semt_py',
            'pd.read_csv',
            'table_manager.add_table',
            'reconciliation_manager.reconcile',
            'extension_manager.extend_column'
        ]
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Detect code block boundaries
            if line.strip().startswith('```'):
                if in_code_block:
                    in_code_block = False
                    current_language = "mixed" if current_language == "markdown" else current_language
                else:
                    in_code_block = True
                    code_language = line.strip()[3:].strip() or "unknown"
                    if code_language.lower() in ['python', 'py']:
                        current_language = "python"
                    else:
                        current_language = "mixed"
            
            current_chunk_lines.append(line)
            
            # Check if we should create a new chunk
            is_operation_boundary = any(boundary in line_lower for boundary in operation_boundaries)
            at_size_limit = len(current_chunk_lines) >= max_lines_per_chunk
            
            should_split = (
                (at_size_limit or (is_operation_boundary and len(current_chunk_lines) > 10)) and
                not in_code_block and  # Don't split in the middle of code blocks
                len(current_chunk_lines) > 5  # Minimum chunk size
            )
            
            if should_split or i == len(lines) - 1:
                # Create chunk
                chunk_text = '\n'.join(current_chunk_lines).strip()
                
                if chunk_text and len(chunk_text) > 50:  # Only create substantial chunks
                    chunk = Chunk(
                        text=chunk_text,
                        language=current_language,
                        number=chunk_number,
                        line_start=line_start,
                        line_end=i
                    )
                    chunks.append(chunk)
                    chunk_number += 1
                
                # Reset for next chunk
                current_chunk_lines = []
                current_language = "markdown"
                line_start = i + 1
        
        return chunks