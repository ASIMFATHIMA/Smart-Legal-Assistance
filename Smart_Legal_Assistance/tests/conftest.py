import os
import sys

# Ensure `from config.config import ...` resolves to `Smart_Legal_Assistance/config/config.py`
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
