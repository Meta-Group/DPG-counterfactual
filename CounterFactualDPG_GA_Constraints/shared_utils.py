"""
Shared utilities for CounterFactual DPG notebooks.

This module contains common functions and classes used across multiple notebooks
in the CounterFactualDPG_GA_Constraints directory.
"""

import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional


# ============================================================================
# Path Setup
# ============================================================================

def setup_paths():
    """
    Set up paths for accessing constraints directory.
    
    Returns:
        tuple: (notebook_dir, constraints_dir, PATH)
            - notebook_dir: Current working directory
            - constraints_dir: Absolute path to constraints directory
            - PATH: Path to constraints directory with trailing slash
    """
    notebook_dir = os.getcwd()
    constraints_dir = os.path.abspath(os.path.join(notebook_dir, '..', 'constraints'))
    PATH = constraints_dir + '/'
    return notebook_dir, constraints_dir, PATH
