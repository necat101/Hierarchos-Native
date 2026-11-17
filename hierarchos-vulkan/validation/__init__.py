"""Validator package.

The checks are normally executed as scripts, which puts this directory on
``sys.path`` automatically, so their sibling imports resolve. Running the
release test by module path
(``python -m unittest hierarchos-vulkan.validation.test_unified_release``)
does not do that, and the test module imports ``check_unified_release``
directly. The package bootstrap adds this directory once, so every sibling
import inside the checks resolves exactly as it does in script mode.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
