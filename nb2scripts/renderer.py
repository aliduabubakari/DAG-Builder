"""
Jinja2-based template rendering for script generation.
"""
import datetime
from pathlib import Path
from typing import Dict, Any
from jinja2 import Environment, FileSystemLoader, select_autoescape
from .schema import Script, Operation

class ScriptRenderer:
    """Renders scripts using Jinja2 templates."""
    
    def __init__(self, template_dir: str = None):
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters
        self.env.filters['datetime'] = self._datetime_filter
        self.env.filters['snake_case'] = self._snake_case_filter
        self.env.filters['clean_name'] = self._clean_name_filter
        # Don't add regex_replace filter - we're using simple string replacement

    def _regex_replace_filter(self, text: str, pattern: str, replacement: str = '') -> str:
        """Custom Jinja2 filter for regex replacement."""
        import re
        return re.sub(pattern, replacement, text)
    
    def render_script(self, script: Script) -> str:
        """
        Render a complete script using templates.
        
        Args:
            script: Script object to render
            
        Returns:
            Complete script content as string
        """
        # Render the base template
        base_template = self.env.get_template("base.j2")
        base_content = base_template.render(
            script=script,
            timestamp=datetime.datetime.now(),
            operations=script.operations
        )
        
        # Render each operation
        operation_blocks = []
        for i, operation in enumerate(script.operations):
            try:
                template = self.env.get_template(f"{operation.op_type}.j2")
                block = template.render(
                    op=operation,
                    operation=operation,
                    index=i + 1,
                    script=script
                )
                operation_blocks.append(block)
            except Exception as e:
                # Fallback to generic template
                print(f"Warning: No template for {operation.op_type}, using generic template")
                template = self.env.get_template("generic.j2")
                block = template.render(
                    op=operation,
                    operation=operation,
                    index=i + 1,
                    script=script
                )
                operation_blocks.append(block)
        
        # Combine base and operations
        full_content = base_content + "\n\n" + "\n\n".join(operation_blocks)
        
        # Render the main function
        main_template = self.env.get_template("main.j2")
        main_content = main_template.render(
            script=script,
            operations=script.operations
        )
        
        return full_content + "\n\n" + main_content
    
    def _datetime_filter(self, format_string: str) -> str:
        """Custom Jinja2 filter for datetime formatting."""
        return datetime.datetime.now().strftime(format_string)
    
    def _snake_case_filter(self, text: str) -> str:
        """Convert text to snake_case."""
        import re
        # Replace spaces and special chars with underscores
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', '_', text)
        return text.lower()
    
    def _clean_name_filter(self, text: str) -> str:
        """Clean text for use as function/variable names."""
        import re
        # Keep only alphanumeric and underscores
        text = re.sub(r'[^\w]', '_', text)
        # Remove multiple underscores
        text = re.sub(r'_+', '_', text)
        # Remove leading/trailing underscores
        return text.strip('_').lower()