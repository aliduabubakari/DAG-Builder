"""
Advanced syntax fixing and post-processing for generated scripts.
"""
import re
import ast
import json
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

try:
    import black
    BLACK_AVAILABLE = True
except ImportError:
    BLACK_AVAILABLE = False

try:
    import autopep8
    AUTOPEP8_AVAILABLE = True
except ImportError:
    AUTOPEP8_AVAILABLE = False

class AdvancedSyntaxFixer:
    """Advanced syntax fixing with multiple strategies."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def fix_script_content(self, content: str, script_name: str = "unknown") -> str:
        """
        Apply comprehensive syntax fixing to script content.
        """
        self.logger.info(f"🔧 Fixing syntax for {script_name}")
        
        # Step 1: Fix common template rendering issues
        content = self._fix_template_issues(content)
        
        # Step 2: Fix string and quote issues
        content = self._fix_string_issues(content)
        
        # Step 3: Fix indentation and structure issues
        content = self._fix_structure_issues(content)
        
        # Step 4: Fix Python-specific syntax issues
        content = self._fix_python_syntax_issues(content)
        
        # Step 5: Validate and auto-correct
        content = self._validate_and_correct(content)
        
        # Step 6: Apply formatting if possible
        content = self._apply_formatting(content)
        
        return content
    
    def _fix_template_issues(self, content: str) -> str:
        """Fix common Jinja2 template rendering issues."""
        
        # Fix missing newlines in template output
        content = re.sub(r'(\w+)(\s+)(\w+)(\s*=)', r'\1\n\2\3\4', content)
        
        # Fix operations list formatting issues
        content = re.sub(
            r"'operations': \[\s*\{'type': '(\w+)', 'name': '([^']+)'\},\s*\]",
            r"'operations': [{'type': '\1', 'name': '\2'}]",
            content
        )
        
        # Fix malformed list continuations
        content = re.sub(r'results = \[\]\s*result_(\d+)', r'results = []\n    result_\1', content)
        
        # Fix function call continuations
        content = re.sub(r'(\w+)\(([^)]+)\)\s*(\w+)', r'\1(\2)\n    \3', content)
        
        return content
    
    def _fix_string_issues(self, content: str) -> str:
        """Fix string quoting and escaping issues."""
        
        # Fix single quotes in names that break string literals
        def fix_quoted_names(match):
            full_match = match.group(0)
            inner_content = match.group(1)
            # Replace single quotes with underscores in the inner content
            fixed_inner = inner_content.replace("'", "_").replace('"', '_')
            return full_match.replace(inner_content, fixed_inner)
        
        # Fix problematic quotes in operation names
        content = re.sub(r"'name': '([^']*'[^']*)'", fix_quoted_names, content)
        content = re.sub(r'"name": "([^"]*"[^"]*)"', fix_quoted_names, content)
        
        # Fix triple quotes and other quote issues
        content = re.sub(r'"""([^"]*"""[^"]*?)"""', r'"""\1"""', content)
        
        # Escape backslashes in strings
        content = re.sub(r'\\(?![nrt"\'\\])', r'\\\\', content)
        
        return content
    
    def _fix_structure_issues(self, content: str) -> str:
        """Fix indentation and structural issues."""
        
        lines = content.split('\n')
        fixed_lines = []
        in_function = False
        in_class = False
        current_indent = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if not stripped:
                fixed_lines.append('')
                continue
            
            # Detect function/class definitions
            if stripped.startswith('def ') or stripped.startswith('class '):
                in_function = True
                current_indent = 0
                fixed_lines.append(line)
                continue
            
            # Fix indentation for function bodies
            if in_function:
                if stripped and not line.startswith(' ') and not stripped.startswith('#'):
                    # This should be indented
                    line = '    ' + stripped
                elif line.startswith(' ') and len(line) - len(line.lstrip()) < 4:
                    # Insufficient indentation
                    line = '    ' + stripped
            
            # Fix common indentation mistakes
            if stripped.startswith('try:') or stripped.startswith('except') or stripped.startswith('finally:'):
                # Ensure proper indentation for try-except blocks
                if not line.startswith('    '):
                    line = '    ' + stripped
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_python_syntax_issues(self, content: str) -> str:
        """Fix Python-specific syntax issues."""
        
        # Fix missing commas in lists/dicts
        content = re.sub(r"(\w+)'(\s*})", r"\1'\2", content)  # Fix trailing quotes before dict close
        content = re.sub(r"(\w+)'\s*([,}])", r"\1'\2", content)  # Fix spacing around quotes
        
        # Fix malformed dictionary definitions
        content = re.sub(r"{\s*'(\w+)':\s*'([^']*)',?\s*}", r"{'\1': '\2'}", content)
        
        # Fix function parameter issues
        content = re.sub(r'def\s+(\w+)\s*\(\s*([^)]*)\s*\)\s*:', r'def \1(\2):', content)
        
        # Fix assignment issues
        content = re.sub(r'(\w+)\s*=\s*([^=\n]+)\s*([^=\n]+)\s*=', r'\1 = \2\n    \3 =', content)
        
        # Fix return statement issues
        content = re.sub(r'return\s+(\w+)\s*\n\s*([^r])', r'return \1\n\n\2', content)
        
        return content
    
    def _validate_and_correct(self, content: str) -> str:
        """Validate syntax and apply corrections."""
        
        max_attempts = 5
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Try to parse the content
                ast.parse(content)
                self.logger.info(f"✅ Syntax validation passed on attempt {attempt + 1}")
                return content
            except SyntaxError as e:
                self.logger.warning(f"⚠️ Syntax error on attempt {attempt + 1}: {e}")
                
                # Try to fix the specific error
                content = self._fix_specific_syntax_error(content, e)
                attempt += 1
            except Exception as e:
                self.logger.error(f"❌ Unexpected error during validation: {e}")
                break
        
        self.logger.warning(f"⚠️ Could not fully fix syntax after {max_attempts} attempts")
        return content
    
    def _fix_specific_syntax_error(self, content: str, error: SyntaxError) -> str:
        """Fix specific syntax errors based on the error message."""
        
        lines = content.split('\n')
        
        if error.lineno and error.lineno <= len(lines):
            error_line = lines[error.lineno - 1]
            
            # Fix common specific errors
            if "invalid syntax" in str(error):
                if "'" in error_line and error.offset:
                    # Likely a quote issue
                    fixed_line = self._fix_line_quotes(error_line)
                    lines[error.lineno - 1] = fixed_line
                
                elif "=" in error_line:
                    # Likely an assignment issue
                    fixed_line = self._fix_assignment_line(error_line)
                    lines[error.lineno - 1] = fixed_line
                
                elif error_line.strip().endswith('{') or error_line.strip().endswith('('):
                    # Likely missing closing bracket
                    next_line_idx = error.lineno
                    if next_line_idx < len(lines):
                        lines[next_line_idx] = '    ' + lines[next_line_idx].strip()
            
            elif "Perhaps you forgot a comma" in str(error):
                # Add missing comma
                if error.offset and error.offset > 0:
                    line = error_line
                    pos = min(error.offset - 1, len(line) - 1)
                    if pos > 0 and line[pos] != ',':
                        line = line[:pos] + ',' + line[pos:]
                        lines[error.lineno - 1] = line
        
        return '\n'.join(lines)
    
    def _fix_line_quotes(self, line: str) -> str:
        """Fix quote issues in a single line."""
        # Replace problematic single quotes in names
        if "'name':" in line:
            # Extract the value part and fix quotes
            match = re.search(r"'name':\s*'([^']*)'", line)
            if match:
                name_value = match.group(1)
                # Replace internal quotes with underscores
                fixed_name = name_value.replace("'", "_").replace('"', '_')
                line = line.replace(match.group(0), f"'name': '{fixed_name}'")
        
        return line
    
    def _fix_assignment_line(self, line: str) -> str:
        """Fix assignment issues in a line."""
        # Fix multiple assignments on one line
        if line.count('=') > 1 and not any(op in line for op in ['==', '!=', '<=', '>=']):
            parts = line.split('=')
            if len(parts) > 2:
                # Split into multiple lines
                first_assignment = f"{parts[0].strip()} = {parts[1].strip()}"
                return first_assignment  # Return first assignment, others will be handled separately
        
        return line
    
    def _apply_formatting(self, content: str) -> str:
        """Apply code formatting using available tools."""
        
        if BLACK_AVAILABLE:
            try:
                formatted = black.format_str(content, mode=black.FileMode(line_length=100))
                self.logger.info("✅ Applied Black formatting")
                return formatted
            except Exception as e:
                self.logger.warning(f"⚠️ Black formatting failed: {e}")
        
        if AUTOPEP8_AVAILABLE:
            try:
                formatted = autopep8.fix_code(content, options={'max_line_length': 100})
                self.logger.info("✅ Applied autopep8 formatting")
                return formatted
            except Exception as e:
                self.logger.warning(f"⚠️ autopep8 formatting failed: {e}")
        
        return content

def fix_generated_scripts(output_dir: Path) -> List[Path]:
    """Fix all generated scripts in the output directory."""
    
    fixer = AdvancedSyntaxFixer()
    fixed_files = []
    
    for script_file in output_dir.glob("*.py"):
        if script_file.name in ['requirements.txt', 'README.md']:
            continue
        
        try:
            # Read original content
            original_content = script_file.read_text(encoding='utf-8')
            
            # Apply fixes
            fixed_content = fixer.fix_script_content(original_content, script_file.name)
            
            # Write back if changed
            if fixed_content != original_content:
                script_file.write_text(fixed_content, encoding='utf-8')
                fixed_files.append(script_file)
                print(f"   🔧 Fixed {script_file.name}")
            else:
                print(f"   ✅ {script_file.name} - no fixes needed")
                
        except Exception as e:
            print(f"   ❌ Failed to fix {script_file.name}: {e}")
    
    return fixed_files