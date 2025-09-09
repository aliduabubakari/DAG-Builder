"""
Intelligent script grouping with semantic naming.
"""
import re
from typing import List, Dict, Any
from .schema import Operation, Script

class SemanticScriptGrouper:
    """Groups operations into scripts with meaningful, semantic names."""
    
    def __init__(self):
        self.operation_keywords = {
            'reconcile': {
                'poi': ['point_of_interest', 'poi', 'point', 'interest'],
                'place': ['place', 'location', 'geographic', 'geo'],
                'entity': ['entity', 'entities'],
                'person': ['person', 'people', 'individual'],
                'organization': ['organization', 'org', 'company'],
                'event': ['event', 'events', 'occurrence']
            },
            'extend': {
                'poi': ['point_of_interest', 'poi', 'point', 'interest'],
                'place': ['place', 'location', 'geographic', 'geo'],
                'properties': ['properties', 'props', 'attributes'],
                'metadata': ['metadata', 'meta', 'data'],
                'enrichment': ['enrichment', 'enrich', 'enhancement']
            }
        }
    
    def group_operations_intelligently(self, operations: List[Operation], max_scripts: int = None) -> List[Script]:
        """
        Group operations intelligently with semantic naming.
        
        Args:
            operations: List of operations to group
            max_scripts: Maximum number of scripts (None for automatic)
            
        Returns:
            List of Script objects with semantic names
        """
        if max_scripts:
            return self._group_with_limit(operations, max_scripts)
        else:
            return self._group_semantically(operations)
    
    def _group_semantically(self, operations: List[Operation]) -> List[Script]:
        """Group operations based on semantic understanding (no limit)."""
        
        scripts = []
        
        # Separate operations by type first
        load_ops = [op for op in operations if op.op_type == 'load']
        reconcile_ops = [op for op in operations if op.op_type == 'reconcile']
        extend_ops = [op for op in operations if op.op_type == 'extend']
        other_ops = [op for op in operations if op.op_type not in ['load', 'reconcile', 'extend']]
        
        script_counter = 1
        
        # 1. Create load script(s)
        if load_ops:
            if len(load_ops) == 1:
                script_name = f"{script_counter:02d}_load_table"
            else:
                script_name = f"{script_counter:02d}_load_data"
            
            scripts.append(Script(
                name=script_name,
                stage=script_counter,
                operations=load_ops
            ))
            script_counter += 1
        
        # 2. Create reconcile scripts (group by subject)
        reconcile_groups = self._group_reconcile_operations(reconcile_ops)
        for group_name, group_ops in reconcile_groups.items():
            script_name = f"{script_counter:02d}_reconcile_{group_name}"
            scripts.append(Script(
                name=script_name,
                stage=script_counter,
                operations=group_ops
            ))
            script_counter += 1
        
        # 3. Create extend scripts (group by subject)
        extend_groups = self._group_extend_operations(extend_ops)
        for group_name, group_ops in extend_groups.items():
            script_name = f"{script_counter:02d}_extend_{group_name}"
            scripts.append(Script(
                name=script_name,
                stage=script_counter,
                operations=group_ops
            ))
            script_counter += 1
        
        # 4. Handle other operations
        if other_ops:
            script_name = f"{script_counter:02d}_other_operations"
            scripts.append(Script(
                name=script_name,
                stage=script_counter,
                operations=other_ops
            ))
        
        return scripts
    
    def _group_with_limit(self, operations: List[Operation], max_scripts: int) -> List[Script]:
        """Group operations with a maximum script limit."""
        
        if len(operations) <= max_scripts:
            # Can create individual scripts with semantic names
            return self._create_individual_semantic_scripts(operations)
        
        # Need to merge operations intelligently
        return self._merge_operations_semantically(operations, max_scripts)
    
    def _create_individual_semantic_scripts(self, operations: List[Operation]) -> List[Script]:
        """Create individual scripts with semantic names."""
        
        scripts = []
        
        for i, operation in enumerate(operations, 1):
            script_name = self._generate_semantic_name(operation, i)
            
            scripts.append(Script(
                name=script_name,
                stage=i,
                operations=[operation]
            ))
        
        return scripts
    
    def _merge_operations_semantically(self, operations: List[Operation], max_scripts: int) -> List[Script]:
        """Merge operations while maintaining semantic meaning and maximizing separation when possible."""
        
        # Group by type first
        type_groups = {}
        for op in operations:
            if op.op_type not in type_groups:
                type_groups[op.op_type] = []
            type_groups[op.op_type].append(op)
        
        scripts = []
        script_counter = 1
        
        # Calculate how many scripts we can allocate per type
        total_operations = len(operations)
        
        # Strategy: Try to give each operation its own script if we have room
        if total_operations <= max_scripts:
            # We have room - create individual scripts with semantic names
            for op in operations:
                script_name = self._generate_semantic_name_for_individual(op, script_counter)
                scripts.append(Script(
                    name=script_name,
                    stage=script_counter,
                    operations=[op]
                ))
                script_counter += 1
            return scripts
        
        # We need to merge some operations
        # Priority: Keep reconcile and extend operations separate if possible
        
        remaining_slots = max_scripts
        
        # 1. Handle load operations (can be grouped together)
        if 'load' in type_groups:
            load_ops = type_groups['load']
            script_name = f"{script_counter:02d}_load_operations"
            scripts.append(Script(
                name=script_name,
                stage=script_counter,
                operations=load_ops
            ))
            script_counter += 1
            remaining_slots -= 1
            del type_groups['load']
        
        # 2. Handle reconcile operations (try to keep separate)
        if 'reconcile' in type_groups and remaining_slots > 0:
            reconcile_ops = type_groups['reconcile']
            
            if len(reconcile_ops) <= remaining_slots:
                # We can give each reconcile operation its own script
                for op in reconcile_ops:
                    subject = self._identify_operation_subject(op, 'reconcile')
                    script_name = f"{script_counter:02d}_reconcile_{subject}"
                    scripts.append(Script(
                        name=script_name,
                        stage=script_counter,
                        operations=[op]
                    ))
                    script_counter += 1
                    remaining_slots -= 1
            else:
                # Need to group some reconcile operations
                reconcile_groups = self._group_reconcile_operations(reconcile_ops)
                
                # If we have room for all groups, create separate scripts
                if len(reconcile_groups) <= remaining_slots:
                    for subject, group_ops in reconcile_groups.items():
                        script_name = f"{script_counter:02d}_reconcile_{subject}"
                        scripts.append(Script(
                            name=script_name,
                            stage=script_counter,
                            operations=group_ops
                        ))
                        script_counter += 1
                        remaining_slots -= 1
                else:
                    # Merge all reconcile operations into one script
                    all_subjects = list(reconcile_groups.keys())
                    combined_name = '_'.join(all_subjects[:2])  # Use first 2 subjects
                    if len(all_subjects) > 2:
                        combined_name += '_multi'
                    
                    script_name = f"{script_counter:02d}_reconcile_{combined_name}"
                    scripts.append(Script(
                        name=script_name,
                        stage=script_counter,
                        operations=reconcile_ops
                    ))
                    script_counter += 1
                    remaining_slots -= 1
            
            del type_groups['reconcile']
        
        # 3. Handle extend operations (try to keep separate)
        if 'extend' in type_groups and remaining_slots > 0:
            extend_ops = type_groups['extend']
            
            if len(extend_ops) <= remaining_slots:
                # We can give each extend operation its own script
                for op in extend_ops:
                    subject = self._identify_operation_subject(op, 'extend')
                    script_name = f"{script_counter:02d}_extend_{subject}"
                    scripts.append(Script(
                        name=script_name,
                        stage=script_counter,
                        operations=[op]
                    ))
                    script_counter += 1
                    remaining_slots -= 1
            else:
                # Group extend operations
                extend_groups = self._group_extend_operations(extend_ops)
                
                if len(extend_groups) <= remaining_slots:
                    for subject, group_ops in extend_groups.items():
                        script_name = f"{script_counter:02d}_extend_{subject}"
                        scripts.append(Script(
                            name=script_name,
                            stage=script_counter,
                            operations=group_ops
                        ))
                        script_counter += 1
                        remaining_slots -= 1
                else:
                    # Merge all extend operations
                    script_name = f"{script_counter:02d}_extend_operations"
                    scripts.append(Script(
                        name=script_name,
                        stage=script_counter,
                        operations=extend_ops
                    ))
                    script_counter += 1
                    remaining_slots -= 1
            
            del type_groups['extend']
        
        # 4. Handle any remaining operation types
        for op_type, ops in type_groups.items():
            if remaining_slots > 0:
                script_name = f"{script_counter:02d}_{op_type}_operations"
                scripts.append(Script(
                    name=script_name,
                    stage=script_counter,
                    operations=ops
                ))
                script_counter += 1
                remaining_slots -= 1
            else:
                # Add to the last script if no room
                if scripts:
                    scripts[-1].operations.extend(ops)
        
        return scripts

    def _generate_semantic_name_for_individual(self, operation: Operation, index: int) -> str:
        """Generate a semantic name for an individual operation script."""
        
        op_type = operation.op_type
        
        if op_type == 'load':
            return f"{index:02d}_load_table"
        elif op_type == 'reconcile':
            subject = self._identify_operation_subject(operation, 'reconcile')
            return f"{index:02d}_reconcile_{subject}"
        elif op_type == 'extend':
            subject = self._identify_operation_subject(operation, 'extend')
            return f"{index:02d}_extend_{subject}"
        else:
            # Clean up the operation name for use in filename
            clean_name = re.sub(r'[^\w]', '_', operation.name.lower())
            clean_name = re.sub(r'_+', '_', clean_name).strip('_')
            return f"{index:02d}_{op_type}_{clean_name}"
        
    def _group_reconcile_operations(self, operations: List[Operation]) -> Dict[str, List[Operation]]:
        """Group reconcile operations by subject."""
        
        groups = {}
        
        for op in operations:
            subject = self._identify_operation_subject(op, 'reconcile')
            if subject not in groups:
                groups[subject] = []
            groups[subject].append(op)
        
        return groups
    
    def _group_extend_operations(self, operations: List[Operation]) -> Dict[str, List[Operation]]:
        """Group extend operations by subject."""
        
        groups = {}
        
        for op in operations:
            subject = self._identify_operation_subject(op, 'extend')
            if subject not in groups:
                groups[subject] = []
            groups[subject].append(op)
        
        return groups
    
    def _identify_operation_subject(self, operation: Operation, op_type: str) -> str:
        """Identify the subject of an operation (poi, place, etc.)."""
        
        # Check operation name and description
        text_to_analyze = f"{operation.name} {operation.meta.get('description', '')}".lower()
        
        # Extract column name if available
        column_name = operation.meta.get('column_name', '').lower()
        if column_name:
            text_to_analyze += f" {column_name}"
        
        # Check for keywords
        if op_type in self.operation_keywords:
            for subject, keywords in self.operation_keywords[op_type].items():
                for keyword in keywords:
                    if keyword in text_to_analyze:
                        return subject
        
        # Fallback: try to extract from operation name
        if 'poi' in text_to_analyze or 'point' in text_to_analyze:
            return 'poi'
        elif 'place' in text_to_analyze or 'location' in text_to_analyze:
            return 'place'
        elif 'person' in text_to_analyze or 'people' in text_to_analyze:
            return 'person'
        elif 'organization' in text_to_analyze or 'org' in text_to_analyze:
            return 'organization'
        else:
            return 'data'  # Generic fallback
    
    def _identify_main_subject(self, operations: List[Operation], op_type: str) -> str:
        """Identify the main subject for a group of operations."""
        
        subjects = []
        for op in operations:
            subject = self._identify_operation_subject(op, op_type)
            subjects.append(subject)
        
        # Return most common subject, or combined name if multiple
        from collections import Counter
        subject_counts = Counter(subjects)
        
        if len(subject_counts) == 1:
            return list(subject_counts.keys())[0]
        elif len(subject_counts) <= 2:
            return '_'.join(sorted(subject_counts.keys()))
        else:
            return 'multi'  # Multiple subjects
    
    def _generate_semantic_name(self, operation: Operation, index: int) -> str:
        """Generate a semantic name for a single operation."""
        
        op_type = operation.op_type
        
        if op_type == 'load':
            return f"{index:02d}_load_table"
        elif op_type == 'reconcile':
            subject = self._identify_operation_subject(operation, 'reconcile')
            return f"{index:02d}_reconcile_{subject}"
        elif op_type == 'extend':
            subject = self._identify_operation_subject(operation, 'extend')
            return f"{index:02d}_extend_{subject}"
        else:
            return f"{index:02d}_{op_type}_operation"