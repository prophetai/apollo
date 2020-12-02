import os
import sys

# https://github.com/pytest-dev/pytest/issues/2421#issuecomment-403724503
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
