"""
Zomato AI Unified Analytics Dashboard
Entry point from workspace root.
"""
import sys
from pathlib import Path
import runpy

target_script = Path(__file__).resolve().parent / "ai" / "dashboard.py"
runpy.run_path(str(target_script), run_name="__main__")
