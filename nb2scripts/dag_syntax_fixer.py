import ast
import logging
import re
from pathlib import Path
from typing import List, Tuple, Optional
from openai import AzureOpenAI

class DAGSyntaxFixer:
    """Fixes syntax errors in generated Airflow DAGs using LLM."""
    
    def __init__(self, api_key: str, endpoint: str, deployment: str = "gpt-4o-mini"):
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-08-01-preview", 
            azure_endpoint=endpoint
        )
        self.deployment = deployment
        self.logger = logging.getLogger(__name__)
    
    def fix_dag_syntax(self, dag_file: Path) -> bool:
        """
        Fix syntax errors in a DAG file using LLM.
        
        Args:
            dag_file: Path to the DAG file
            
        Returns:
            True if all errors were fixed, False otherwise
        """
        self.logger.info(f"🔍 Checking DAG syntax: {dag_file.name}")
        
        # Check for both syntax errors and common DAG issues
        errors = self._find_all_errors(dag_file)
        
        if not errors:
            self.logger.info(f"✅ No errors in {dag_file.name}")
            return True
        
        self.logger.info(f"📋 Found {len(errors)} issues in {dag_file.name}")
        
        # Apply LLM fixes
        success = self._apply_llm_fixes(dag_file, errors)
        
        if success:
            self.logger.info(f"✅ All issues fixed in {dag_file.name}")
            return True
        else:
            self.logger.error(f"❌ Could not fix all issues in {dag_file.name}")
            return False
    
    def _find_all_errors(self, file_path: Path) -> List[str]:
        """Find syntax errors and common DAG issues."""
        errors = []
        content = file_path.read_text(encoding='utf-8')
        
        # Check for syntax errors
        try:
            ast.parse(content)
        except SyntaxError as e:
            errors.append(f"Syntax Error Line {e.lineno}: {e.msg}")
        except Exception as e:
            errors.append(f"Parse error: {str(e)}")
        
        # Check for common DAG issues
        dag_issues = self._find_dag_issues(content)
        errors.extend(dag_issues)
        
        return errors
    
    def _find_dag_issues(self, content: str) -> List[str]:
        """Find common DAG-specific issues."""
        issues = []
        lines = content.split('\n')
        
        # Check for invalid task variable names (starting with numbers)
        for i, line in enumerate(lines, 1):
            if re.match(r'^\s*\d+\w*\s*=\s*DockerOperator', line):
                issues.append(f"Line {i}: Task variable name starts with number (invalid Python identifier)")
        
        # Check for malformed dependency definitions
        for i, line in enumerate(lines, 1):
            # Look for dependencies not properly formatted
            if '>>' in line and not line.strip().startswith('#'):
                # Check if it's a standalone dependency line (common issue)
                if re.match(r'^\s*\w+\s*>>\s*\w+\s*$', line.strip()):
                    issues.append(f"Line {i}: Dependency definition should be indented properly within DAG context")
        
        # Check for unterminated triple quotes in doc_md
        in_triple_quote = False
        quote_start_line = 0
        for i, line in enumerate(lines, 1):
            if '"""' in line:
                quote_count = line.count('"""')
                if quote_count % 2 == 1:  # Odd number means opening or closing
                    if not in_triple_quote:
                        in_triple_quote = True
                        quote_start_line = i
                    else:
                        in_triple_quote = False
        
        if in_triple_quote:
            issues.append(f"Line {quote_start_line}: Unterminated triple-quoted string")
        
        return issues
    
    def _apply_llm_fixes(self, file_path: Path, errors: List[str]) -> bool:
        """Apply LLM-powered fixes to the DAG file."""
        
        original_content = file_path.read_text(encoding='utf-8')
        
        # Create fix prompt
        prompt = self._create_enhanced_fix_prompt(original_content, errors)
        
        try:
            self.logger.info(f"🤖 LLM fixing DAG syntax for {file_path.name}")
            
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Python and Airflow developer. Fix all syntax errors and DAG issues while preserving functionality."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=6000
            )
            
            fixed_content = response.choices[0].message.content.strip()
            
            # Extract code from markdown if present
            if "```python" in fixed_content:
                fixed_content = fixed_content.split("```python")[1].split("```")[0].strip()
            elif "```" in fixed_content:
                fixed_content = fixed_content.split("```")[1].split("```")[0].strip()
            
            # Validate the fix
            try:
                ast.parse(fixed_content)
                self.logger.info("🤖 LLM provided valid fix")
            except SyntaxError as e:
                self.logger.error(f"🤖 LLM fix still has syntax errors: {e}")
                return False
            
            # Write the fixed content
            file_path.write_text(fixed_content, encoding='utf-8')
            self.logger.info(f"✅ Applied LLM fix to {file_path.name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ LLM fix failed: {e}")
            return False
    
    def _create_enhanced_fix_prompt(self, content: str, errors: List[str]) -> str:
        """Create an enhanced prompt for LLM to fix all DAG issues."""
        
        return f"""Fix ALL issues in this Airflow DAG file.

ISSUES FOUND:
{chr(10).join([f"- {error}" for error in errors])}

ORIGINAL DAG CODE:
```python
{content}
```

SPECIFIC REQUIREMENTS:
1. Fix ALL syntax errors and DAG issues listed above
2. Task variable names MUST be valid Python identifiers (no starting with numbers)
   - Change "01_load_operations_task" to "load_operations_task"
   - Change "02_reconcile_poi_task" to "reconcile_poi_task" 
   - Change "03_reconcile_place_task" to "reconcile_place_task"
   - Change "04_extend_poi_task" to "extend_poi_task"
3. Fix task dependency definitions - they should be properly indented within the DAG context
4. Ensure all triple-quoted strings are properly terminated
5. Keep all DockerOperator configurations intact
6. Maintain proper Airflow DAG structure
7. Preserve all environment variables and XCom references
8. Keep all task documentation and metadata

COMMON FIXES NEEDED:
- Task variable names: Change numbered prefixes to descriptive names
- Dependencies: Ensure proper indentation and format like:
  ```
  # Dependencies
  find_input_file_task >> load_operations_task
  load_operations_task >> reconcile_poi_task  
  reconcile_poi_task >> reconcile_place_task
  reconcile_place_task >> extend_poi_task
  extend_poi_task >> cleanup_task
  ```
- String literals: Fix any unterminated quotes
- Proper spacing and indentation throughout

Return ONLY the corrected Python code without explanation or markdown formatting."""

def fix_generated_dag(dag_file: Path, api_key: str, endpoint: str) -> bool:
        """
        Convenience function to fix a generated DAG file.
        
        Args:
            dag_file: Path to the DAG file
            api_key: Azure OpenAI API key
            endpoint: Azure OpenAI endpoint
            
        Returns:
            True if successful, False otherwise  
        """
        try:
            fixer = DAGSyntaxFixer(api_key, endpoint)
            return fixer.fix_dag_syntax(dag_file)
        except Exception as e:
            logging.error(f"❌ DAG syntax fixing failed: {e}")
            return False