from __future__ import annotations

import os
import runpy
import sys


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_APP = os.path.join(BASE_DIR, "app.py")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Keep the legacy entrypoint working by delegating to the latest Streamlit app.
runpy.run_path(ROOT_APP, run_name="__main__")
