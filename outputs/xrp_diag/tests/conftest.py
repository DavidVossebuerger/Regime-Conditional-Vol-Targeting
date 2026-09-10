import sys
from pathlib import Path
# Ensure repository root is on sys.path so tests can import the package in editable mode
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
