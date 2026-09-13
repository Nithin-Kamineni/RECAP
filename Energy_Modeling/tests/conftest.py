"""Shared plumbing for the new test root.

The environment fixture is IMPORTED, not copied: `eccenergy/tests/conftest.py`
explains at length why every test needs `os.environ` restored around it, and two
copies of that would be the beginning of a drift. pytest discovers a fixture by
name in a conftest's namespace, so the import is the whole registration.
"""
from eccenergy.tests.conftest import _ecc_env  # noqa: F401  (autouse fixture)
