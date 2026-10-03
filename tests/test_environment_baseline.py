"""
Tests for VoiceTTSr environment baseline and preflight verification (VT-REM-1 / ENV-01).
Verifies interpreter requirements, runtime and development dependencies,
and dynamic pytest collection health.
"""

import sys
import os
import subprocess
import json
import pytest

from tools.verify_env import (
    REQUIRED_PYTHON_MIN,
    REQUIRED_RUNTIME_PACKAGES,
    REQUIRED_DEV_PACKAGES,
    check_python_version,
    check_packages,
    check_pytest_collection,
    run_preflight,
)


def test_python_version_meets_minimum():
    """Verify current interpreter meets or exceeds Python 3.10."""
    res = check_python_version()
    assert res["status"] == "PASS"
    assert sys.version_info >= REQUIRED_PYTHON_MIN


def test_runtime_packages_installed():
    """Verify all production runtime dependencies are present."""
    res = check_packages(REQUIRED_RUNTIME_PACKAGES)
    assert res["passed"] is True, f"Missing runtime packages: {res['details']}"
    for pkg in REQUIRED_RUNTIME_PACKAGES:
        assert pkg in res["details"]
        assert res["details"][pkg]["status"] == "PASS"


def test_dev_packages_installed():
    """Verify development and test runner dependencies are present."""
    res = check_packages(REQUIRED_DEV_PACKAGES)
    assert res["passed"] is True, f"Missing dev packages: {res['details']}"
    for pkg in REQUIRED_DEV_PACKAGES:
        assert pkg in res["details"]
        assert res["details"][pkg]["status"] == "PASS"


def test_dynamic_pytest_collection_discovers_tests():
    """Verify pytest collection runs cleanly and dynamically discovers test cases."""
    col = check_pytest_collection()
    assert col["status"] == "PASS"
    assert col["returncode"] == 0
    assert col["collected_count"] is not None
    # Test suite should discover at least 48 tests dynamically
    assert col["collected_count"] >= 48


def test_run_preflight_end_to_end():
    """Verify the combined preflight check returns PASS."""
    report = run_preflight(check_dev=True, run_collection=True)
    assert report["overall_status"] == "PASS"
    assert report["python"]["status"] == "PASS"
    assert report["runtime_packages"]["passed"] is True
    assert report["dev_packages"]["passed"] is True
    assert report["pytest_collection"]["status"] == "PASS"


def test_verify_env_cli_json():
    """Verify CLI execution of tools/verify_env.py --json emits valid JSON with zero exit code."""
    res = subprocess.run(
        [sys.executable, os.path.join("tools", "verify_env.py"), "--json"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["overall_status"] == "PASS"
    assert "python" in data
    assert "runtime_packages" in data
    assert "dev_packages" in data
    assert "pytest_collection" in data
    assert data["pytest_collection"]["collected_count"] >= 48
