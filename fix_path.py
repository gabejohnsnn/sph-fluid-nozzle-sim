"""
Add this module to the beginning of a script to add the project root to Python's path
Usage:
    import fix_path
    # Now you can import fluid_sim modules
"""

import os
import sys

# Get the directory containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the project root directory to Python's path
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
