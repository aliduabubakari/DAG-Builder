"""
Notebook loading and parsing utilities.
"""
import json
import nbformat
from pathlib import Path
from typing import List
from .schema import Cell

class NotebookLoader:
    """Handles loading and parsing Jupyter notebooks."""
    
    @staticmethod
    def load_notebook(path: str) -> List[Cell]:
        """
        Load a Jupyter notebook and return a list of Cell objects.
        
        Args:
            path: Path to the .ipynb file
            
        Returns:
            List of Cell objects
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Notebook not found: {path}")
        
        if not path.endswith('.ipynb'):
            raise ValueError("File must be a Jupyter notebook (.ipynb)")
        
        try:
            nb = nbformat.read(path, as_version=4)
        except Exception as e:
            raise RuntimeError(f"Failed to load notebook {path}: {e}")
        
        cells = []
        for raw_cell in nb.cells:
            source = raw_cell.get('source', '')
            if isinstance(source, list):
                source = ''.join(source)
            
            cell = Cell(
                source=source,
                kind=raw_cell.get('cell_type', 'code'),
                metadata=raw_cell.get('metadata', {})
            )
            cells.append(cell)
        
        return cells
    
    @staticmethod
    def validate_notebook(cells: List[Cell]) -> bool:
        """
        Validate that the notebook has expected structure.
        
        Args:
            cells: List of cells to validate
            
        Returns:
            True if valid
        """
        if not cells:
            return False
        
        # Check if we have at least some code cells
        code_cells = [c for c in cells if c.kind == "code" and c.source.strip()]
        return len(code_cells) > 0