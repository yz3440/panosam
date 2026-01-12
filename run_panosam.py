#!/usr/bin/env python3
"""
Standalone script to run PanoSAM segmentation on panoramic images.

This script provides a simple way to run SAM3 segmentation without installing
the package. For production use, install the package and use the `panosam` CLI.

Usage:
    python run_panosam.py --image <path> --prompt <text>
"""

import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from panosam.cli import main

if __name__ == "__main__":
    main()
