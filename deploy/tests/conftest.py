# deploy/tests/conftest.py
import sys
from pathlib import Path

DEPLOY = Path(__file__).resolve().parents[1]          # /home/mk/CarCV-metrics/deploy
sys.path.insert(0, str(DEPLOY / "evaluation"))         # import evaluate / metrics / visualize
sys.path.insert(0, str(DEPLOY / "scripts"))            # import run_local / prep_*
