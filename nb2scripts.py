#!/usr/bin/env python3
"""
Main entry point for nb2scripts CLI.
"""
import sys
import os
from pathlib import Path

# Add the tools directory to the path so we can import nb2scripts package
tools_dir = Path(__file__).parent
sys.path.insert(0, str(tools_dir))

# Now import from the nb2scripts package
try:
    from nb2scripts.cli import main
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required files are in place:")
    print("- tools/nb2scripts/__init__.py")
    print("- tools/nb2scripts/cli.py")
    print("- tools/nb2scripts/loader.py")
    print("- tools/nb2scripts/classifier.py")
    print("- tools/nb2scripts/renderer.py")
    print("- tools/nb2scripts/writer.py")
    print("- tools/nb2scripts/schema.py")
    sys.exit(1)

if __name__ == "__main__":
    main()