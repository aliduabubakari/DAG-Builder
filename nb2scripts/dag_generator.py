"""
LLM-powered DAG generator with intelligent documentation.
"""
import re
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from openai import AzureOpenAI
from jinja2 import Environment, FileSystemLoader

from .schema import Script

class DAGGenerator:
    """Generates Airflow DAGs with LLM-powered documentation."""
    
    def __init__(self, api_key: str, endpoint: str, deployment: str = "gpt-4o-mini"):
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-08-01-preview",
            azure_endpoint=endpoint
        )
        self.deployment = deployment
        self.logger = logging.getLogger(__name__)
        
        # Set up Jinja2 environment
        template_dir = Path(__file__).parent / "templates"
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def generate_dag(self, scripts: List[Script], output_dir: Path, dag_name: str = None) -> Path:
        """
        Generate a complete Airflow DAG from the scripts.
        
        Args:
            scripts: List of generated Script objects
            output_dir: Directory where scripts are located
            dag_name: Optional custom DAG name
            
        Returns:
            Path to generated DAG file
        """
        self.logger.info(f"🏗️ Generating DAG for {len(scripts)} scripts...")
        
        if not dag_name:
            dag_name = "generated_notebook_pipeline"
        
        # Analyze scripts to understand the pipeline
        pipeline_analysis = self._analyze_pipeline(scripts, output_dir)
        
        # Generate LLM-powered documentation
        dag_documentation = self._generate_dag_documentation(pipeline_analysis)
        
        # Prepare script information for DAG template
        dag_scripts = self._prepare_script_data(scripts, pipeline_analysis)
        
        # Generate DAG using template
        dag_content = self._render_dag_template(
            dag_name, dag_documentation, dag_scripts, pipeline_analysis
        )
        
        # Write DAG file
        dag_file_path = output_dir / f"{dag_name}.py"
        dag_file_path.write_text(dag_content, encoding='utf-8')
        
        self.logger.info(f"✅ Generated DAG: {dag_file_path}")
        return dag_file_path
    
    def _analyze_pipeline(self, scripts: List[Script], output_dir: Path) -> Dict[str, Any]:
        """Analyze the generated scripts to understand the pipeline structure."""
        
        analysis = {
            'total_scripts': len(scripts),
            'operation_types': [],
            'script_details': [],
            'pipeline_flow': [],
            'estimated_data_flow': 'CSV → JSON → Enriched JSON',
            'complexity': 'medium'
        }
        
        for script in scripts:
            # Analyze script content
            script_file = output_dir / script.filename
            if script_file.exists():
                content = script_file.read_text(encoding='utf-8')
                script_analysis = self._analyze_script_content(content, script)
            else:
                script_analysis = self._analyze_script_from_operations(script)
            
            analysis['script_details'].append(script_analysis)
            
            # Track operation types
            for op in script.operations:
                if op.op_type not in analysis['operation_types']:
                    analysis['operation_types'].append(op.op_type)
            
            # Build pipeline flow
            stage_name = self._infer_stage_name(script)
            analysis['pipeline_flow'].append(stage_name)
        
        # Determine complexity
        if len(scripts) <= 2:
            analysis['complexity'] = 'simple'
        elif len(scripts) <= 4:
            analysis['complexity'] = 'medium'
        else:
            analysis['complexity'] = 'complex'
        
        return analysis
    
    def _analyze_script_content(self, content: str, script: Script) -> Dict[str, Any]:
        """Analyze actual script content to extract meaningful information."""
        
        analysis = {
            'name': script.name,
            'filename': script.filename,
            'stage': script.stage,
            'operations': len(script.operations),
            'primary_function': 'unknown',
            'input_type': 'unknown',
            'output_type': 'JSON',
            'api_calls': [],
            'data_processing': [],
            'environment_vars': {}
        }
        
        # Extract function patterns
        if 'reconcile' in content.lower():
            analysis['primary_function'] = 'reconciliation'
            analysis['api_calls'].append('reconciliation_manager.reconcile')
        
        if 'extend' in content.lower():
            analysis['primary_function'] = 'extension'
            analysis['api_calls'].append('extension_manager.extend_column')
        
        if 'read_csv' in content:
            analysis['primary_function'] = 'data_loading'
            analysis['input_type'] = 'CSV'
            analysis['data_processing'].append('CSV parsing')
        
        if 'add_table' in content:
            analysis['data_processing'].append('Table creation')
        
        # Extract potential environment variables
        env_var_pattern = r'os\.environ\.get\([\'"]([^\'\"]+)[\'"]'
        env_vars = re.findall(env_var_pattern, content)
        for var in env_vars:
            analysis['environment_vars'][var] = f"${{{var}}}"
        
        return analysis
    
    def _analyze_script_from_operations(self, script: Script) -> Dict[str, Any]:
        """Analyze script based on operations when content is not available."""
        
        analysis = {
            'name': script.name,
            'filename': script.filename,
            'stage': script.stage,
            'operations': len(script.operations),
            'primary_function': 'data_processing',
            'input_type': 'JSON',
            'output_type': 'JSON',
            'api_calls': [],
            'data_processing': [],
            'environment_vars': {}
        }
        
        # Infer from operation types
        op_types = [op.op_type for op in script.operations]
        
        if 'load' in op_types:
            analysis['primary_function'] = 'data_loading'
            analysis['input_type'] = 'CSV'
            analysis['data_processing'].append('Data loading and table creation')
        
        if 'reconcile' in op_types:
            analysis['primary_function'] = 'reconciliation'
            analysis['api_calls'].append('reconciliation_manager.reconcile')
            analysis['data_processing'].append('Entity reconciliation')
        
        if 'extend' in op_types:
            analysis['primary_function'] = 'extension'
            analysis['api_calls'].append('extension_manager.extend_column')
            analysis['data_processing'].append('Data enrichment')
        
        return analysis
    
    def _generate_dag_documentation(self, pipeline_analysis: Dict[str, Any]) -> str:
        """Generate comprehensive DAG documentation using LLM."""
        
        prompt = f"""
Generate comprehensive Airflow DAG documentation for a data processing pipeline.

PIPELINE ANALYSIS:
- Total Scripts: {pipeline_analysis['total_scripts']}
- Operation Types: {', '.join(pipeline_analysis['operation_types'])}
- Pipeline Flow: {' → '.join(pipeline_analysis['pipeline_flow'])}
- Complexity: {pipeline_analysis['complexity']}

SCRIPT DETAILS:
{json.dumps(pipeline_analysis['script_details'], indent=2)}

Generate documentation that includes:

1. **Pipeline Overview** - What this pipeline does
2. **Architecture** - Hybrid approach with XCom and file management  
3. **Data Flow** - How data transforms through each stage
4. **Stages Description** - What each stage accomplishes
5. **Expected Outputs** - What the final result looks like
6. **Monitoring & Metrics** - What gets tracked

Format as markdown suitable for Airflow DAG doc_md. Be professional but concise.
Focus on the actual data transformations happening, not generic descriptions.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert in data engineering and Airflow DAG documentation. Generate clear, accurate pipeline documentation."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            documentation = response.choices[0].message.content.strip()
            self.logger.info("🤖 Generated LLM-powered DAG documentation")
            return documentation
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate LLM documentation: {e}")
            return self._generate_fallback_documentation(pipeline_analysis)
    
    def _generate_fallback_documentation(self, pipeline_analysis: Dict[str, Any]) -> str:
        """Generate basic documentation if LLM fails."""
        
        return f"""
# Generated Notebook Pipeline

## Overview
This DAG implements a {pipeline_analysis['complexity']} data processing pipeline with {pipeline_analysis['total_scripts']} stages.

## Pipeline Stages
{chr(10).join([f"{i+1}. **{stage.title()}**" for i, stage in enumerate(pipeline_analysis['pipeline_flow'])])}

## Architecture
- **Hybrid Approach**: Combines Airflow XCom with file-based state management
- **Docker Execution**: Each stage runs in isolated containers
- **Data Flow**: {pipeline_analysis['estimated_data_flow']}

## Operation Types
- {', '.join(pipeline_analysis['operation_types']).title()}

## Monitoring
Each stage produces metrics and maintains audit trails for complete traceability.
"""
    
    def _prepare_script_data(self, scripts: List[Script], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare script data for DAG template."""
        
        dag_scripts = []
        
        for i, script in enumerate(scripts):
            script_analysis = analysis['script_details'][i]
            
            # Generate environment variables
            env_vars = {
                "DATASET_ID": "2",  # Default
                "TABLE_ID": "",
            }
            
            # Add script-specific env vars based on function
            if script_analysis['primary_function'] == 'reconciliation':
                env_vars.update({
                    "RECONCILIATION_COLUMN_NAME": "City",
                    "RECONCILIATION_RECONCILIATOR_ID": "geocodingHere",
                    "RECONCILIATION_OPTIONAL_COLUMNS": ""
                })
            elif script_analysis['primary_function'] == 'extension':
                env_vars.update({
                    "EXTENSION_COLUMN_NAME": "City", 
                    "EXTENSION_EXTENDER_ID": "meteoPropertiesOpenMeteo",
                    "EXTENSION_PROPERTIES": "temperature_max,temperature_min"
                })
            elif script_analysis['primary_function'] == 'data_loading':
                env_vars.update({
                    "LOAD_DATASET_ID": "2",
                    "LOAD_TABLE_NAME_PATTERN": "pipeline_table_{ds_nodash}"
                })
            
            # Generate documentation for this task
            task_doc = self._generate_task_documentation(script, script_analysis)
            
            dag_scripts.append({
                'name': script.name,
                'filename': script.filename,
                'stage': script.stage,
                'stage_name': self._infer_stage_name(script),
                'env_vars': env_vars,
                'documentation': task_doc
            })
        
        return dag_scripts
    
    def _generate_task_documentation(self, script: Script, analysis: Dict[str, Any]) -> str:
        """Generate documentation for individual DAG tasks."""
        
        function_desc = {
            'reconciliation': 'Reconciles entities using external APIs',
            'extension': 'Enriches data with additional properties',
            'data_loading': 'Loads and processes input data'
        }.get(analysis['primary_function'], 'Processes data')
        
        processing_steps = analysis.get('data_processing', ['Data processing'])
        
        return f"""
## Stage {analysis['stage']}: {analysis['name'].replace('_', ' ').title()}

**Function**: {function_desc}

**Processing**:
{chr(10).join([f"- {step}" for step in processing_steps])}

**Input**: {analysis['input_type']} data from previous stage
**Output**: {analysis['output_type']} data for next stage
"""
    
    def _infer_stage_name(self, script: Script) -> str:
        """Infer stage name from script."""
        name = script.name.lower()
        
        if 'load' in name:
            return 'loaded'
        elif 'reconcile' in name:
            return 'reconciled'
        elif 'extend' in name:
            return 'extended'
        else:
            return 'processed'
    
    def _render_dag_template(self, dag_name: str, documentation: str, scripts: List[Dict], analysis: Dict) -> str:
        """Render the DAG template with all data."""
        
        try:
            template = self.jinja_env.get_template("dag_template.j2")
            
            # Prepare template variables
            template_vars = {
                'dag_name': dag_name,
                'dag_id': dag_name.lower().replace(' ', '_').replace('-', '_'),
                'dag_documentation': documentation,
                'dag_tags': ["generated", "notebook", analysis['complexity']],
                'scripts': scripts,
                'stage_names': [script['stage_name'] for script in scripts]
            }
            
            self.logger.info(f"📝 Rendering DAG template with {len(scripts)} scripts")
            
            # Validate template variables
            for script in scripts:
                if not script.get('name'):
                    raise ValueError(f"Script missing name: {script}")
                if not script.get('stage_name'):
                    script['stage_name'] = f"stage_{script.get('stage', 1)}"
            
            rendered = template.render(**template_vars)
            self.logger.info("✅ DAG template rendered successfully")
            return rendered
            
        except Exception as e:
            self.logger.error(f"❌ Failed to render DAG template: {e}")
            # Return a simple fallback DAG
            return self._generate_fallback_dag(dag_name, scripts)

    def _generate_fallback_dag(self, dag_name: str, scripts: List[Dict]) -> str:
        """Generate a simple fallback DAG if template rendering fails."""
        
        script_tasks = []
        dependencies = []
        
        for i, script in enumerate(scripts):
            task_name = f"{script['name']}_task".replace('-', '_')
            script_tasks.append(f"""
        {task_name} = DockerOperator(
            task_id="{script['name']}",
            image="semt-pipeline:latest",
            command=["python", "/app/scripts/{script['filename']}"],
            environment={{
                "RUN_ID": DAG_RUN_ID_TEMPLATE,
                "STAGE_NAME": "{script.get('stage_name', 'processed')}",
                "STAGE_NUMBER": "{script['stage']}",
                "API_BASE_URL": "http://node-server-api:3003",
                "DATA_DIR": "/app/data",
            }},
            do_xcom_push=True,
            auto_remove=True,
            docker_url="unix://var/run/docker.sock",
            network_mode="semt_pipeline_network",
            mounts=[Mount(source="/app/data", target="/app/data", type="bind")],
            mount_tmp_dir=False,
        )""")
            
            if i == 0:
                dependencies.append(f"find_input_file_task >> {task_name}")
            else:
                prev_task = f"{scripts[i-1]['name']}_task".replace('-', '_')
                dependencies.append(f"{prev_task} >> {task_name}")
        
        return f'''# dags/{dag_name}.py
    from __future__ import annotations
    import os
    import pendulum
    from airflow.models.dag import DAG
    from airflow.providers.docker.operators.docker import DockerOperator
    from airflow.operators.python import PythonOperator
    from docker.types import Mount

    DAG_RUN_ID_TEMPLATE = "{{{{ ds_nodash }}}}_{{{{ ts_nodash }}}}"

    def find_latest_csv_file(**kwargs):
        import glob
        search_path = os.path.join("/app/data", "*.csv")
        files = glob.glob(search_path)
        if not files:
            raise FileNotFoundError("No CSV files found")
        return max(files, key=os.path.getmtime)

    with DAG(
        dag_id="{dag_name.lower().replace(' ', '_')}",
        start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
        catchup=False,
        schedule=None,
        tags=["generated", "fallback"],
    ) as dag:

        find_input_file_task = PythonOperator(
            task_id="find_input_file",
            python_callable=find_latest_csv_file,
        )
        
        {"".join(script_tasks)}

        # Dependencies
        {chr(10).join(dependencies)}
    '''