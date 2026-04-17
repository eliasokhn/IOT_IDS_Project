"""Shared pytest configuration and fixtures."""
import sys
import os

# Ensure src/ is always on the path when running tests from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
