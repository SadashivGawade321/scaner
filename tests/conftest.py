# tests/conftest.py
# Shared pytest configuration for AAROGYA tests.
import os, sys

# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
