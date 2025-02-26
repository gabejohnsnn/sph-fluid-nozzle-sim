#!/usr/bin/env python

"""
Helper script to run examples with the correct Python path
Usage: python run_example.py <example-script-name> [arguments]
Example: python run_example.py examples/cross_section_visualization.py --fluid oil
"""

import sys
import os
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_example.py <example-script-name> [arguments]")
        print("Example: python run_example.py examples/cross_section_visualization.py --fluid oil")
        return
    
    # Get the example script and arguments
    example_script = sys.argv[1]
    args = sys.argv[2:]
    
    # Ensure the example script exists
    if not os.path.exists(example_script):
        print(f"Error: Example script '{example_script}' not found")
        return
    
    # Create a command to run the example with correct Python path
    cmd = [sys.executable, "-m", example_script.replace("/", ".").replace(".py", "")]
    cmd.extend(args)
    
    # Print command for debugging
    print(f"Running: {' '.join(cmd)}")
    
    # Run the command
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
