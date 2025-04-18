import sys
from pathlib import Path

# Add src directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))