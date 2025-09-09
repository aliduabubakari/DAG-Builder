"""
LLM-powered syntax fixing and post-processing for generated scripts.
"""
import re
import ast
import json
import logging
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from openai import AzureOpenAI

try:
    import black
    BLACK_AVAILABLE = True
except ImportError:
    BLACK_AVAILABLE = False

class LLMSyntaxFixer:
    """LLM-powered syntax fixing with detailed error reporting."""
    
    def __init__(self, api_key: str, endpoint: str, deployment: str = "gpt-4o-mini"):
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-08-01-preview",
            azure_endpoint=endpoint
        )
        self.deployment = deployment
        self.logger = logging.getLogger(__name__)
    
    def fix_script_with_llm(self, script_content: str, script_name: str) -> Tuple[str, List[str]]:
        """
        Fix script syntax using LLM with detailed error reporting.
        
        Returns:
            Tuple[str, List[str]]: (fixed_content, list_of_errors_found)
        """
        self.logger.info(f"🤖 LLM fixing syntax for {script_name}")
        
        # Step 1: Analyze and collect all syntax errors
        errors = self._collect_syntax_errors(script_content)
        
        if not errors:
            self.logger.info(f"✅ No syntax errors found in {script_name}")
            return script_content, []
        
        # Step 2: Use LLM to fix all errors at once
        fixed_content = self._llm_fix_multiple_errors(script_content, errors, script_name)
        
        # Step 3: Validate the fix
        final_errors = self._collect_syntax_errors(fixed_content)
        
        if not final_errors:
            self.logger.info(f"✅ All syntax errors fixed in {script_name}")
        else:
            self.logger.warning(f"⚠️ {len(final_errors)} errors remain in {script_name}")
        
        return fixed_content, final_errors
    
    def _collect_syntax_errors(self, content: str) -> List[Dict[str, Any]]:
        """Collect all syntax errors with detailed information."""
        errors = []
        
        try:
            ast.parse(content)
            return []  # No errors
        except SyntaxError as e:
            error_info = {
                'type': 'SyntaxError',
                'message': str(e),
                'line': e.lineno,
                'column': e.offset,
                'text': e.text.strip() if e.text else '',
                'filename': e.filename or 'unknown'
            }
            errors.append(error_info)
            
            # Try to find additional errors by fixing this one temporarily
            try:
                lines = content.split('\n')
                if e.lineno and e.lineno <= len(lines):
                    # Comment out the problematic line and check for more errors
                    temp_lines = lines.copy()
                    temp_lines[e.lineno - 1] = f"# TEMP_COMMENT: {temp_lines[e.lineno - 1]}"
                    temp_content = '\n'.join(temp_lines)
                    
                    # Recursively find more errors
                    additional_errors = self._collect_syntax_errors(temp_content)
                    for additional_error in additional_errors:
                        if additional_error not in errors:
                            errors.append(additional_error)
            except:
                pass
                
        except Exception as e:
            error_info = {
                'type': type(e).__name__,
                'message': str(e),
                'line': None,
                'column': None,
                'text': '',
                'filename': 'unknown'
            }
            errors.append(error_info)
        
        return errors
    
    def _llm_fix_multiple_errors(self, content: str, errors: List[Dict], script_name: str) -> str:
        """Use LLM to fix multiple syntax errors at once."""
        
        # Create detailed error report
        error_report = self._create_error_report(content, errors)
        
        prompt = f"""
You are an expert Python developer. Fix ALL syntax errors in this Python script.

SCRIPT NAME: {script_name}

DETECTED ERRORS:
{error_report}

ORIGINAL SCRIPT:
```python
{content}
```

INSTRUCTIONS:
1. Fix ALL syntax errors while preserving the original logic
2. Pay special attention to:
   - Unterminated string literals (add missing quotes)
   - Malformed f-strings or format strings
   - Missing commas in lists/dictionaries
   - Incorrect indentation
   - Missing colons after function/class definitions
   - Unmatched parentheses, brackets, or braces
3. Do NOT change the overall structure or functionality
4. Return ONLY the corrected Python code, no explanations
5. Ensure the code is properly formatted and follows Python syntax rules

CORRECTED SCRIPT:
"""

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert Python developer specializing in fixing syntax errors. You return only valid Python code."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            fixed_code = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            fixed_code = re.sub(r'^```python\s*\n?', '', fixed_code)
            fixed_code = re.sub(r'^```\s*$', '', fixed_code, flags=re.MULTILINE)
            
            self.logger.info(f"🤖 LLM provided fix for {script_name}")
            return fixed_code
            
        except Exception as e:
            self.logger.error(f"❌ LLM fix failed for {script_name}: {e}")
            return content  # Return original if LLM fails
    
    def _create_error_report(self, content: str, errors: List[Dict]) -> str:
        """Create a detailed error report for the LLM."""
        
        lines = content.split('\n')
        report_parts = []
        
        for i, error in enumerate(errors, 1):
            report_parts.append(f"ERROR {i}:")
            report_parts.append(f"  Type: {error['type']}")
            report_parts.append(f"  Message: {error['message']}")
            
            if error['line']:
                report_parts.append(f"  Line {error['line']}: {error['text']}")
                
                # Add context lines
                line_num = error['line'] - 1
                start_line = max(0, line_num - 2)
                end_line = min(len(lines), line_num + 3)
                
                report_parts.append("  Context:")
                for j in range(start_line, end_line):
                    marker = " >>> " if j == line_num else "     "
                    report_parts.append(f"  {j+1:3d}{marker}{lines[j]}")
            
            report_parts.append("")
        
        return "\n".join(report_parts)
    
    def apply_final_formatting(self, content: str) -> str:
        """Apply final formatting to the corrected code."""
        if BLACK_AVAILABLE:
            try:
                formatted = black.format_str(content, mode=black.FileMode(line_length=100))
                self.logger.info("✅ Applied Black formatting")
                return formatted
            except Exception as e:
                self.logger.warning(f"⚠️ Black formatting failed: {e}")
        
        return content

class ScriptErrorAnalyzer:
    """Analyzes and reports on script errors across all generated files."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_all_scripts(self, output_dir: Path) -> Dict[str, List[Dict]]:
        """Analyze all scripts and return error report."""
        
        script_errors = {}
        
        for script_file in output_dir.glob("*.py"):
            if script_file.name in ['requirements.txt', 'README.md']:
                continue
            
            try:
                content = script_file.read_text(encoding='utf-8')
                errors = self._collect_file_errors(content)
                
                if errors:
                    script_errors[script_file.name] = errors
                    self.logger.info(f"📋 Found {len(errors)} errors in {script_file.name}")
                else:
                    self.logger.info(f"✅ No errors in {script_file.name}")
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to analyze {script_file.name}: {e}")
        
        return script_errors
    
    def _collect_file_errors(self, content: str) -> List[Dict[str, Any]]:
        """Collect errors from a single file."""
        errors = []
        
        try:
            ast.parse(content)
            return []
        except SyntaxError as e:
            error_info = {
                'type': 'SyntaxError',
                'message': str(e),
                'line': e.lineno,
                'column': e.offset,
                'text': e.text.strip() if e.text else '',
                'severity': 'high'
            }
            errors.append(error_info)
        except Exception as e:
            error_info = {
                'type': type(e).__name__,
                'message': str(e),
                'line': None,
                'column': None,
                'text': '',
                'severity': 'medium'
            }
            errors.append(error_info)
        
        return errors
    
    def generate_error_report(self, script_errors: Dict[str, List[Dict]]) -> str:
        """Generate a comprehensive error report."""
        
        if not script_errors:
            return "🎉 All scripts are syntax-error free!"
        
        report_lines = [
            "📋 SYNTAX ERROR REPORT",
            "=" * 50,
            ""
        ]
        
        total_errors = sum(len(errors) for errors in script_errors.values())
        report_lines.append(f"Total files with errors: {len(script_errors)}")
        report_lines.append(f"Total errors found: {total_errors}")
        report_lines.append("")
        
        for script_name, errors in script_errors.items():
            report_lines.append(f"📄 {script_name}")
            report_lines.append("-" * (len(script_name) + 4))
            
            for i, error in enumerate(errors, 1):
                report_lines.append(f"  {i}. {error['type']}: {error['message']}")
                if error['line']:
                    report_lines.append(f"     Line {error['line']}: {error['text']}")
                report_lines.append("")
        
        return "\n".join(report_lines)

def fix_all_scripts_with_llm(output_dir: Path, api_key: str, endpoint: str) -> Dict[str, Any]:
    """
    Fix all scripts using LLM and return comprehensive report.
    
    Returns:
        Dict with results: {
            'fixed_files': List[str],
            'failed_files': List[str], 
            'error_report': str,
            'success_rate': float
        }
    """
    
    # Step 1: Analyze all errors first
    analyzer = ScriptErrorAnalyzer()
    initial_errors = analyzer.analyze_all_scripts(output_dir)
    
    if not initial_errors:
        return {
            'fixed_files': [],
            'failed_files': [],
            'error_report': "🎉 All scripts were already syntax-error free!",
            'success_rate': 1.0
        }
    
    print(f"📋 Found errors in {len(initial_errors)} files. Starting LLM fixes...")
    
    # Step 2: Fix each file with LLM
    fixer = LLMSyntaxFixer(api_key, endpoint)
    fixed_files = []
    failed_files = []
    
    for script_name in initial_errors.keys():
        script_path = output_dir / script_name
        
        try:
            # Read current content
            original_content = script_path.read_text(encoding='utf-8')
            
            # Fix with LLM
            fixed_content, remaining_errors = fixer.fix_script_with_llm(
                original_content, script_name
            )
            
            # Apply final formatting
            final_content = fixer.apply_final_formatting(fixed_content)
            
            # Write back
            script_path.write_text(final_content, encoding='utf-8')
            
            if not remaining_errors:
                fixed_files.append(script_name)
                print(f"   ✅ Fixed {script_name}")
            else:
                failed_files.append(script_name)
                print(f"   ⚠️ Partially fixed {script_name} ({len(remaining_errors)} errors remain)")
                
        except Exception as e:
            failed_files.append(script_name)
            print(f"   ❌ Failed to fix {script_name}: {e}")
    
    # Step 3: Generate final report
    final_errors = analyzer.analyze_all_scripts(output_dir)
    error_report = analyzer.generate_error_report(final_errors)
    
    success_rate = len(fixed_files) / len(initial_errors) if initial_errors else 1.0
    
    return {
        'fixed_files': fixed_files,
        'failed_files': failed_files,
        'error_report': error_report,
        'success_rate': success_rate,
        'initial_error_count': len(initial_errors),
        'final_error_count': len(final_errors)
    }