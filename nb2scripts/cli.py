"""
Command-line interface for nb2scripts.
"""
import argparse
import sys
import os
import ast
import logging
from pathlib import Path
from typing import List

try:
    import black
    BLACK_AVAILABLE = True
except ImportError:
    BLACK_AVAILABLE = False
    print("⚠️ Black not available. Install with: pip install black")

try:
    import autopep8
    AUTOPEP8_AVAILABLE = True
except ImportError:
    AUTOPEP8_AVAILABLE = False
    print("⚠️ autopep8 not available. Install with: pip install autopep8")

from .loader import NotebookLoader
from .classifier import OperationClassifier
from .renderer import ScriptRenderer
from .writer import ScriptWriter
from .schema import Script
from .syntax_fixer import AdvancedSyntaxFixer, fix_generated_scripts
from .llm_syntax_fixer import fix_all_scripts_with_llm, ScriptErrorAnalyzer
from .script_grouper import SemanticScriptGrouper
from .dag_generator import DAGGenerator


# Try to import enhanced LLM classifier
try:
    from .llm_classifier import EnhancedLLMClassifier
    LLM_AVAILABLE = True
except ImportError as e:
    LLM_AVAILABLE = False
    print(f"⚠️ Enhanced LLM classifier not available: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Jupyter notebooks to production-ready Python scripts"
    )
    parser.add_argument(
        "notebook",
        help="Path to the Jupyter notebook (.ipynb file)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="scripts",
        help="Output directory for generated scripts (default: scripts)"
    )
    parser.add_argument(
        "--use-enhanced-llm",
        action="store_true",
        help="Use enhanced LLM-based intelligent chunking"
    )
    parser.add_argument(
        "--llm-api-key",
        help="Azure OpenAI API key (or set AZURE_OPENAI_KEY env var)"
    )
    parser.add_argument(
        "--llm-endpoint",
        help="Azure OpenAI endpoint (or set AZURE_OPENAI_ENDPOINT env var)"
    )
    parser.add_argument(
        "--max-scripts",
        type=int,
        help="Maximum number of scripts to generate (optional - if not provided, LLM determines optimal number)"
    )
    parser.add_argument(
        "--llm-fix-syntax",
        action="store_true",
        default=True,
        help="Use LLM to fix syntax errors (default: enabled)"
    )
    parser.add_argument(
        "--no-llm-fix",
        action="store_true",
        help="Disable LLM syntax fixing"
    )
    parser.add_argument(
        "--generate-dag",
        action="store_true",
        help="Generate an Airflow DAG for the scripts"
    )
    parser.add_argument(
        "--dag-name",
        help="Custom name for the generated DAG (default: generated_notebook_pipeline)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Handle LLM fix logic
    if args.no_llm_fix:
        args.llm_fix_syntax = False
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    try:
        # Check if notebook exists
        if not os.path.exists(args.notebook):
            print(f"❌ Notebook file not found: {args.notebook}")
            sys.exit(1)
        
        # Get LLM credentials if needed
        api_key = args.llm_api_key or os.environ.get('AZURE_OPENAI_KEY')
        endpoint = args.llm_endpoint or os.environ.get('AZURE_OPENAI_ENDPOINT')
        
        if (args.use_enhanced_llm or args.llm_fix_syntax or args.generate_dag) and (not api_key or not endpoint):
            print("❌ LLM API key and endpoint required for LLM features")
            print("Set --llm-api-key and --llm-endpoint or environment variables")
            sys.exit(1)
        
        # Load notebook
        print(f"🔄 Loading notebook: {args.notebook}")
        loader = NotebookLoader()
        cells = loader.load_notebook(args.notebook)
        
        if not loader.validate_notebook(cells):
            print("❌ Invalid notebook structure")
            sys.exit(1)
        
        print(f"📓 Loaded {len(cells)} cells")
        
        # Choose classification method
        if args.use_enhanced_llm:
            if not LLM_AVAILABLE:
                print("❌ Enhanced LLM classifier not available")
                print("Install required packages: pip install openai")
                sys.exit(1)
            
            print("🤖 Using enhanced LLM classification...")
            classifier = EnhancedLLMClassifier(api_key, endpoint)
            operations = classifier.classify(cells)
        else:
            print("🔍 Using heuristic classification...")
            classifier = OperationClassifier()
            operations = classifier.classify(cells)
        
        if not operations:
            print("❌ No operations found in notebook")
            sys.exit(1)
        
        print(f"📋 Found {len(operations)} operations:")
        for op in operations:
            confidence = op.meta.get('confidence', 'N/A')
            print(f"   - {op.op_type}: {op.name} (confidence: {confidence})")
        
        # Group into scripts with semantic naming
        print("📦 Grouping operations into scripts...")
        grouper = SemanticScriptGrouper()
        
        if args.max_scripts:
            print(f"   📏 Limiting to maximum {args.max_scripts} scripts")
            print(f"   🔢 Have {len(operations)} operations to distribute")
            
            # Show operation breakdown
            from collections import Counter
            op_types = Counter(op.op_type for op in operations)
            print("   📊 Operation types:")
            for op_type, count in op_types.items():
                print(f"      - {op_type}: {count}")
            
            scripts = grouper.group_operations_intelligently(operations, args.max_scripts)
        else:
            print("   🧠 Using intelligent semantic grouping (no limit)")
            scripts = grouper.group_operations_intelligently(operations)
        
        print(f"📝 Will generate {len(scripts)} scripts:")
        for script in scripts:
            op_types = [op.op_type for op in script.operations]
            op_summary = f"{len(script.operations)} operations: {', '.join(set(op_types))}"
            print(f"   - {script.filename} ({op_summary})")
            
            # Show detailed breakdown for each script
            if args.verbose:
                for op in script.operations:
                    subject = "unknown"
                    if hasattr(grouper, '_identify_operation_subject'):
                        try:
                            subject = grouper._identify_operation_subject(op, op.op_type)
                        except:
                            pass
                    print(f"      └─ {op.op_type} ({subject}): {op.name[:50]}...")
        
        # Render scripts
        print("🏗️  Rendering scripts...")
        renderer = ScriptRenderer()
        rendered_contents = []
        
        for script in scripts:
            content = renderer.render_script(script)
            rendered_contents.append(content)
        
        # Write scripts
        print("💾 Writing scripts to disk...")
        writer = ScriptWriter(args.output_dir)
        written_files = writer.write_all_scripts(scripts, rendered_contents)
        
        # Initial error analysis
        print("🔍 Analyzing generated scripts for syntax errors...")
        analyzer = ScriptErrorAnalyzer()
        initial_errors = analyzer.analyze_all_scripts(Path(args.output_dir))
        
        if initial_errors:
            print(f"⚠️ Found syntax errors in {len(initial_errors)} files")
            
            if args.llm_fix_syntax:
                print("🤖 Applying LLM-powered syntax fixes...")
                fix_results = fix_all_scripts_with_llm(
                    Path(args.output_dir), api_key, endpoint
                )
                
                print(f"\n📊 LLM Fix Results:")
                print(f"   ✅ Successfully fixed: {len(fix_results['fixed_files'])} files")
                print(f"   ⚠️ Partially fixed: {len(fix_results['failed_files'])} files")
                print(f"   📈 Success rate: {fix_results['success_rate']:.1%}")
                
                if fix_results['failed_files']:
                    print(f"\n⚠️ Files that still have issues:")
                    for failed_file in fix_results['failed_files']:
                        print(f"   - {failed_file}")
                
                # Show final error report
                if fix_results['error_report']:
                    print(f"\n{fix_results['error_report']}")
            else:
                # Just show the error report
                error_report = analyzer.generate_error_report(initial_errors)
                print(f"\n{error_report}")
                print("\n💡 Tip: Use --llm-fix-syntax to automatically fix these errors")
        else:
            print("✅ All generated scripts are syntax-error free!")
        
        # Generate DAG if requested
        if args.generate_dag:
            if not LLM_AVAILABLE or not api_key or not endpoint:
                print("⚠️ DAG generation requires LLM features to be enabled")
                print("Provide --llm-api-key and --llm-endpoint for DAG generation")
            else:
                try:
                    print("🏗️ Generating Airflow DAG...")
                    dag_generator = DAGGenerator(api_key, endpoint)
                    dag_file = dag_generator.generate_dag(
                        scripts, 
                        Path(args.output_dir),
                        args.dag_name
                    )
                    print(f"✅ Generated DAG: {dag_file}")
                    
                    # Post-process DAG for syntax errors
                    print("🔍 Checking DAG syntax...")
                    from .dag_syntax_fixer import fix_generated_dag
                    
                    dag_fixed = fix_generated_dag(dag_file, api_key, endpoint)
                    
                    if dag_fixed:
                        print("✅ DAG syntax validation passed")
                        
                        # Final validation
                        try:
                            import ast
                            with open(dag_file, 'r') as f:
                                dag_content = f.read()
                            ast.parse(dag_content)
                            print("✅ Final DAG syntax validation passed")
                        except SyntaxError as e:
                            print(f"⚠️ Final validation failed: {e}")
                    else:
                        print("⚠️ DAG syntax issues detected but may still be usable")
                        
                except Exception as e:
                    print(f"❌ DAG generation failed: {e}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()
        
        # Create additional files
        writer.create_requirements_file()
        writer.create_readme(scripts)
        
        print(f"\n🎉 Successfully generated {len(written_files)} scripts!")
        print(f"📁 Output directory: {Path(args.output_dir).absolute()}")
        
        print("\n📋 Generated files:")
        for file_path in written_files:
            print(f"   ✅ {file_path.name}")
        
        # Show DAG file if generated
        if args.generate_dag and 'dag_file' in locals():
            print(f"   🔄 {Path(dag_file).name}")
        
        print("\n🔧 Next steps:")
        print("1. Set environment variables (see README.md)")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Run scripts: python <script_name>.py")
        
        if args.generate_dag:
            print("4. Deploy DAG to Airflow: copy the .py DAG file to your Airflow DAGs folder")
        
        # Show grouping summary
        if args.max_scripts:
            print(f"\n📊 Script Grouping Summary (Limited to {args.max_scripts}):")
        else:
            print(f"\n📊 Script Grouping Summary (Semantic):")
        
        for script in scripts:
            print(f"   📄 {script.name}:")
            for op in script.operations:
                print(f"      - {op.op_type}: {op.meta.get('description', op.name)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()