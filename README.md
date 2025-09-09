
# DAG Builder for Notebook Pipelines

An intelligent tool that converts Jupyter notebooks into production-ready Airflow DAGs with semantic understanding and LLM-powered optimizations.

## Features

- **Automated Pipeline Generation**: Transforms notebook cells into executable pipeline scripts
- **Semantic Operation Classification**: Identifies load, reconcile, and extend operations using both heuristic patterns and LLM analysis
- **Intelligent Script Grouping**: Optimizes script organization based on operation types and semantic relationships
- **Airflow DAG Generation**: Creates complete Airflow DAGs with proper task dependencies
- **LLM-Powered Enhancements**:
  - Syntax error fixing
  - Documentation generation
  - Pipeline analysis and optimization
- **Hybrid Execution Model**: Supports both file-based and XCom data passing between tasks

## Installation

```bash
git clone https://github.com/yourusername/nb2scripts.git
cd nb2scripts/tools
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- Jupyter notebooks to convert
- Optional: Azure OpenAI API credentials for LLM features

## Usage

### Basic Conversion
```bash
python nb2scripts.py notebook.ipynb --output-dir scripts
```

### Full Conversion with LLM Features
```bash
python nb2scripts.py notebook.ipynb \
  --output-dir scripts \
  --use-enhanced-llm \
  --generate-dag \
  --llm-api-key YOUR_API_KEY \
  --llm-endpoint YOUR_ENDPOINT
```

### Command Line Options
| Option | Description |
|--------|-------------|
| `--output-dir` | Output directory for generated scripts (default: scripts) |
| `--use-enhanced-llm` | Use LLM for intelligent operation classification |
| `--generate-dag` | Generate an Airflow DAG file |
| `--dag-name` | Custom name for the generated DAG |
| `--llm-api-key` | Azure OpenAI API key |
| `--llm-endpoint` | Azure OpenAI endpoint |
| `--max-scripts` | Maximum number of scripts to generate |
| `--verbose` | Show detailed output |

## Architecture

### Core Components
1. **Notebook Loader**: Parses Jupyter notebooks into structured cells
2. **Operation Classifier**:
   - Heuristic-based classification using pattern matching
   - Enhanced LLM classification for complex notebooks
3. **Script Grouper**: Intelligently combines operations into optimal scripts
4. **Template Renderer**: Generates Python scripts using Jinja2 templates
5. **DAG Generator**: Creates complete Airflow DAGs with:
   - DockerOperator tasks
   - XCom data passing
   - Hybrid file-based state management
6. **Syntax Fixer**: Corrects generated code using multiple strategies

### LLM Integration
The system leverages Azure OpenAI for:
- Intelligent operation classification
- Syntax error correction
- Pipeline documentation generation
- DAG optimization suggestions

## Generated Output Structure

```
scripts/
├── 01_load_table.py
├── 02_reconcile_poi.py
├── 03_reconcile_place.py
├── 04_extend_properties.py
├── generated_pipeline_dag.py  # Airflow DAG
├── requirements.txt
└── README.md
```

## Airflow DAG Features

- **Task Isolation**: Each script runs in its own Docker container
- **Data Tracking**: Maintains audit trails and execution metadata
- **Hybrid State Management**: Combines XCom with file-based data passing
- **Self-documenting**: Includes LLM-generated pipeline documentation
- **Error Handling**: Built-in retries and failure notifications

## Customization

### Template Variables
The DAG template supports these variables:
- `dag_name`: Name of your DAG
- `scripts`: List of script objects with metadata
- `dag_documentation`: LLM-generated pipeline documentation
- `dag_tags`: Classification tags for the DAG

### Environment Variables
Configure these in your Airflow environment:
```bash
# API Configuration
export API_BASE_URL='http://your-api-server:3003'
export API_USERNAME='api_user'
export API_PASSWORD='api_password'

# Pipeline Configuration
export DATASET_ID='2'
export DATA_DIR='/path/to/data'
```

## Best Practices

1. **Notebook Preparation**:
   - Use clear cell markers (`# %%`) to separate operations
   - Add metadata tags for operation types where possible
   - Keep related operations in contiguous cells

2. **DAG Optimization**:
   - Start with `--max-scripts` equal to your notebook's main sections
   - Review LLM-generated documentation for accuracy
   - Validate task dependencies in the generated DAG

3. **Execution**:
   - Run syntax validation on generated scripts
   - Test individual script components before full DAG execution
   - Monitor resource usage for memory-intensive operations

## Extending the DAG Builder

The system is designed for extensibility in two key ways:

### 1. Adding New Service Templates
To support additional operation types beyond load/reconcile/extend:

1. **Create a new template file** in `tools/nb2scripts/templates/` (e.g., `transform.j2`)
   ```jinja
   # Example transform.j2 template
   def run_transform_{{ index }}(config: Config, file_manager: PipelineFileManager):
       """{{ op.name }} transformation"""
       current_data = file_manager.load_current_state()
       
       # Custom transformation logic
       {% for cell in op.code_cells %}
       {{ cell.source | indent(4, first=true) }}
       {% endfor %}
       
       file_manager.save_current_state(transformed_data)
   ```

2. **Register the operation type** in `OperationClassifier.PATTERNS`:
   ```python
   # In classifier.py
   PATTERNS = {
       ...,
       'transform': [
           r'df\.transform\(',
           r'apply_transformation\(',
           r'#\s*transform:'
       ]
   }
   ```

3. **Update the renderer** to handle the new template:
   ```python
   # In renderer.py
   def render_script(self, script: Script) -> str:
       try:
           template = self.env.get_template(f"{operation.op_type}.j2")
       except TemplateNotFound:
           template = self.env.get_template("generic.j2")
   ```

### 2. Modifying the DAG Template
To customize the Airflow DAG structure:

1. **Edit `dag_template.j2`** for different execution patterns:
   ```jinja
   {# Example modification for parallel tasks #}
   {% if script.allow_parallel %}
   {{ task_var_name }}_task = KubernetesPodOperator(
       task_id="{{ script.name }}_parallel",
       namespace="data-pipelines",
       ...
   )
   {% endif %}
   ```

2. **Key template variables** available for customization:
   ```python
   # Available in dag_template.j2 context
   {
       "scripts": List[Script],  # All script objects
       "dag_config": {  # From DAGGenerator
           "complexity": "medium", 
           "pipeline_flow": ["load", "transform", "extend"]
       },
       "user_vars": {}  # Pass custom variables via --dag-var
   }
   ```

### 3. Integrating New LLM Providers
To add alternative LLM backends:

1. **Extend the LLMClient** class:
   ```python
   # In llm_client.py
   class CustomLLMClient(LLMClient):
       def __init__(self, model: str, **kwargs):
           self.client = CustomLLMSDK(api_key=kwargs['api_key'])
       
       def classify_chunk(self, chunk: Chunk) -> LLMClassificationResult:
           response = self.client.generate(
               prompt=self._build_prompt(chunk),
               temperature=0.2
           )
           return self._parse_response(response)
   ```

2. **Update the classifier initialization**:
   ```python
   # In llm_classifier.py
   def __init__(self, llm_type="azure"):
       if llm_type == "custom":
           self.client = CustomLLMClient()
       else:
           self.client = AzureLLMClient()
   ```

## Extension Best Practices

1. **Template Development**:
   - Maintain consistent variable naming (`{{ op.name }}`, `{{ index }}`)
   - Include error handling blocks in all templates
   ```jinja
   try:
       {{ operation_code }}
   except Exception as e:
       logger.error(f"Failed {{ op.name }}: {str(e)}")
       raise
   ```

2. **DAG Customization**:
   - Preserve these required template blocks:
   ```jinja
   {# Required in all DAG templates #}
   {{ dag_configuration_block }}
   {{ task_definition_blocks }}
   {{ dependency_definition }}
   ```

3. **LLM Integration**:
   - Implement caching for all LLM clients
   - Maintain consistent response format:
   ```python
   {
       "op_type": str,
       "confidence": float,
       "meta": dict
   }
   ```

## Example Extension: Adding a Validation Service

1. Create `validate.j2`:
   ```jinja
   def run_validation_{{ index }}(data: dict) -> dict:
       """{{ op.name }} validation"""
       validator = Validator(config={{ op.meta | tojson }})
       return {
           'valid': validator.check(data),
           'errors': validator.get_errors()
       }
   ```

2. Register the operation type:
   ```python
   PATTERNS['validate'] = [
       r'validate_',
       r'check_quality',
       r'DataValidator'
   ]
   ```

3. Update DAG template to handle validation results:
   ```jinja
   {% if "validate" in script.operations %}
   validate_task = PythonOperator(
       task_id="validate_results",
       python_callable=validate_output,
       op_args=["{{ "{{ ti.xcom_pull(task_ids='" + last_task + "') }}" }}"]
   )
   {% endif %}
   ```


## Troubleshooting

### Common Issues
| Issue | Solution |
|-------|----------|
| Syntax errors in generated scripts | Use `--llm-fix-syntax` flag or run the syntax fixer manually |
| Missing dependencies | Ensure all packages in requirements.txt are installed |
| DAG not appearing in Airflow | Check Airflow logs for parsing errors |
| Task failures | Verify environment variables and input file paths |


## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

## License

[MIT License](LICENSE)

---

**Note**: This tool requires careful review of generated code before production use. Always validate the output against your specific requirements and security policies.