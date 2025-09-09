"""
File writing utilities for generated scripts.
"""
import os
import stat
from pathlib import Path
from typing import List
from .schema import Script

class ScriptWriter:
    """Handles writing generated scripts to disk."""
    
    def __init__(self, output_dir: str = "scripts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_script(self, script: Script, content: str) -> Path:
        """
        Write a script to disk.
        
        Args:
            script: Script object
            content: Script content
            
        Returns:
            Path to written file
        """
        file_path = self.output_dir / script.filename
        
        # Write content
        file_path.write_text(content, encoding='utf-8')
        
        # Make executable
        current_permissions = file_path.stat().st_mode
        file_path.chmod(current_permissions | stat.S_IEXEC)
        
        return file_path
    
    def write_all_scripts(self, scripts: List[Script], contents: List[str]) -> List[Path]:
        """
        Write multiple scripts to disk.
        
        Args:
            scripts: List of Script objects
            contents: List of script contents
            
        Returns:
            List of paths to written files
        """
        if len(scripts) != len(contents):
            raise ValueError("Number of scripts and contents must match")
        
        written_files = []
        for script, content in zip(scripts, contents):
            file_path = self.write_script(script, content)
            written_files.append(file_path)
            print(f"✅ Wrote {file_path}")
        
        return written_files
    
    def create_requirements_file(self) -> Path:
        """Create a requirements.txt file for the generated scripts."""
        requirements = [
            "pandas>=1.3.0",
            "semt-py>=0.1.0",  # Adjust version as needed
            "codecarbon>=2.0.0",
            "psutil>=5.8.0"
        ]
        
        req_file = self.output_dir / "requirements.txt"
        req_file.write_text("\n".join(requirements) + "\n")
        return req_file
    
    def create_readme(self, scripts: List[Script]) -> Path:
        """Create a README file explaining the generated scripts."""
        readme_content = f"""# Generated Scripts

This directory contains {len(scripts)} auto-generated scripts from a Jupyter notebook.

## Scripts

"""
        
        for script in sorted(scripts, key=lambda s: s.stage):
            readme_content += f"### {script.filename}\n"
            readme_content += f"- Stage: {script.stage}\n"
            readme_content += f"- Operations: {len(script.operations)}\n"
            for op in script.operations:
                readme_content += f"  - {op.op_type}: {op.name}\n"
            readme_content += "\n"
        
        readme_content += """
## Usage

Each script can be run independently:

```bash
python 01_load_table.py
python 02_reconcile_poi.py
# ... etc
```

## Environment Variables

Set these environment variables before running:

```bash
export API_BASE_URL='http://localhost:3003'
export API_USERNAME='your_username'
export API_PASSWORD='your_password'
export DATASET_ID='5'
export DATA_DIR='/path/to/data'
export RUN_ID='$(date +%Y%m%d_%H%M%S)'
```

## Dependencies

Install dependencies with:

```bash
pip install -r requirements.txt
```
"""
        
        readme_file = self.output_dir / "README.md"
        readme_file.write_text(readme_content)
        return readme_file