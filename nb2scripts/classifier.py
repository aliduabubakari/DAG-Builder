"""
Cell classification and operation detection.
"""
import re
import yaml
from typing import List, Dict, Any, Optional
from .schema import Cell, Operation

class OperationClassifier:
    """Classifies notebook cells into logical operations."""
    
    # Heuristic patterns for different operation types
    PATTERNS = {
        'load': [
            r'pd\.read_csv',
            r'add_table',
            r'import.*pandas',
            r'DatasetManager',
            r'TableManager'
        ],
        'reconcile': [
            r'reconciliation_manager\.reconcile',
            r'ReconciliationManager',
            r'wikidataAlligator',
            r'wikidataOpenRefine',
            r'reconciliator_id'
        ],
        'extend': [
            r'extension_manager\.extend',
            r'ExtensionManager',
            r'wikidataPropertySPARQL',
            r'extend_column',
            r'extender_id'
        ]
    }
    
    def __init__(self):
        self.operations = []
        self.current_operation = None
    
    def classify(self, cells: List[Cell]) -> List[Operation]:
        """
        Classify cells into operations using tags first, then heuristics.
        
        Args:
            cells: List of cells to classify
            
        Returns:
            List of Operation objects
        """
        self.operations = []
        self.current_operation = None
        
        for i, cell in enumerate(cells):
            # Skip empty cells
            if not cell.source.strip():
                continue
            
            # Try tag-based classification first
            op_info = self._extract_tag_info(cell)
            if op_info:
                self._handle_tagged_cell(cell, op_info)
            else:
                # Fall back to heuristic classification
                self._handle_heuristic_cell(cell, i)
        
        # Add the last operation if exists
        if self.current_operation:
            self.operations.append(self.current_operation)
        
        # Post-process: extract metadata from operations
        self._extract_operation_metadata()
        
        return self.operations
    
    def _extract_tag_info(self, cell: Cell) -> Optional[Dict[str, Any]]:
        """Extract operation info from cell tags."""
        metadata = cell.metadata
        
        # Check for our custom tags
        if 'op_type' in metadata:
            return {
                'op_type': metadata['op_type'],
                'name': metadata.get('name', metadata['op_type']),
                'order': metadata.get('order', 0),
                'meta': metadata.get('meta', {})
            }
        
        # Check for legacy script tags
        if 'script' in metadata:
            script_name = metadata['script']
            if 'load' in script_name:
                op_type = 'load'
            elif 'reconcile' in script_name:
                op_type = 'reconcile'
            elif 'extend' in script_name:
                op_type = 'extend'
            else:
                op_type = 'unknown'
            
            return {
                'op_type': op_type,
                'name': script_name,
                'order': self._extract_order_from_name(script_name),
                'meta': {}
            }
        
        return None
    
    def _extract_order_from_name(self, name: str) -> int:
        """Extract order number from script name like '01_load_table'."""
        match = re.match(r'^(\d+)', name)
        return int(match.group(1)) if match else 0
    
    def _handle_tagged_cell(self, cell: Cell, op_info: Dict[str, Any]):
        """Handle a cell with explicit tags."""
        op_type = op_info['op_type']
        name = op_info['name']
        
        # Start new operation if type changes or first cell
        if (not self.current_operation or 
            self.current_operation.op_type != op_type or
            self.current_operation.name != name):
            
            if self.current_operation:
                self.operations.append(self.current_operation)
            
            self.current_operation = Operation(
                op_type=op_type,
                name=name,
                order=op_info['order'],
                meta=op_info['meta']
            )
        
        self.current_operation.cells.append(cell)
    
    def _handle_heuristic_cell(self, cell: Cell, index: int):
        """Handle a cell using heuristic pattern matching."""
        detected_type = self._detect_operation_type(cell)
        
        # Start new operation if type changes
        if (not self.current_operation or 
            self.current_operation.op_type != detected_type):
            
            if self.current_operation:
                self.operations.append(self.current_operation)
            
            # Generate a name based on type and index
            name = f"{detected_type}_{len([op for op in self.operations if op.op_type == detected_type]) + 1}"
            
            self.current_operation = Operation(
                op_type=detected_type,
                name=name,
                order=len(self.operations) + 1
            )
        
        self.current_operation.cells.append(cell)
    
    def _detect_operation_type(self, cell: Cell) -> str:
        """Detect operation type using pattern matching."""
        source_lower = cell.source.lower()
        
        # Check patterns in order of specificity
        for op_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, source_lower, re.IGNORECASE):
                    return op_type
        
        # Default fallback
        return 'unknown'
    
    def _extract_operation_metadata(self):
        """Extract metadata from operation source code."""
        for operation in self.operations:
            source = operation.combined_source
            
            if operation.op_type == 'reconcile':
                operation.meta.update(self._extract_reconcile_metadata(source))
            elif operation.op_type == 'extend':
                operation.meta.update(self._extract_extend_metadata(source))
            elif operation.op_type == 'load':
                operation.meta.update(self._extract_load_metadata(source))
    
    def _extract_reconcile_metadata(self, source: str) -> Dict[str, Any]:
        """Extract reconciliation-specific metadata."""
        meta = {}
        
        # Extract reconciliator_id
        match = re.search(r'reconciliator_id\s*=\s*["\']([^"\']+)["\']', source)
        if match:
            meta['reconciliator_id'] = match.group(1)
        
        # Extract column_name
        match = re.search(r'column_name\s*=\s*["\']([^"\']+)["\']', source)
        if match:
            meta['column_name'] = match.group(1)
        
        # Extract from function call
        match = re.search(r'reconcile\([^,]+,\s*["\']([^"\']+)["\'][^,]*,\s*["\']([^"\']+)["\']', source)
        if match:
            meta['column_name'] = match.group(1)
            meta['reconciliator_id'] = match.group(2)
        
        return meta
    
    def _extract_extend_metadata(self, source: str) -> Dict[str, Any]:
        """Extract extension-specific metadata."""
        meta = {}
        
        # Extract extender_id
        match = re.search(r'extender_id\s*=\s*["\']([^"\']+)["\']', source)
        if match:
            meta['extender_id'] = match.group(1)
        
        # Extract column_name
        match = re.search(r'column_name\s*=\s*["\']([^"\']+)["\']', source)
        if match:
            meta['column_name'] = match.group(1)
        
        # Extract properties
        match = re.search(r'properties\s*=\s*["\']([^"\']+)["\']', source)
        if match:
            meta['properties'] = match.group(1)
        
        return meta
    
    def _extract_load_metadata(self, source: str) -> Dict[str, Any]:
        """Extract load-specific metadata."""
        meta = {}
        
        # Extract CSV file path
        match = re.search(r'pd\.read_csv\(["\']([^"\']+)["\']', source)
        if match:
            meta['csv_file'] = match.group(1)
        
        # Extract dataset_id
        match = re.search(r'dataset_id\s*=\s*["\']([^"\']+)["\']', source)
        if match:
            meta['dataset_id'] = match.group(1)
        
        return meta