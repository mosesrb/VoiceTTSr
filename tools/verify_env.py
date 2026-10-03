"""
VoiceTTSr Preflight Environment Verification Tool
Verifies Python interpreter version, required runtime/development packages,
and pytest collection health before running tests or launching the studio.
"""

import sys
import importlib
import subprocess
import argparse
import json
import os

# Suppress pygame welcome banner
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

REQUIRED_PYTHON_MIN = (3, 10)
REQUIRED_PYTHON_MAX = (3, 12)  # Recommended upper bound for legacy Coqui dependencies

REQUIRED_RUNTIME_PACKAGES = [
    "pydub",
    "pygame",
    "numpy",
    "requests",
    "safetensors",
    "send2trash",
    "transformers",
    "torch",
    "soundfile",
]

REQUIRED_DEV_PACKAGES = [
    "pytest",
]


def check_python_version() -> dict:
    ver = sys.version_info
    passed = ver >= REQUIRED_PYTHON_MIN
    return {
        "status": "PASS" if passed else "FAIL",
        "current_version": f"{ver.major}.{ver.minor}.{ver.micro}",
        "required_min": f"{REQUIRED_PYTHON_MIN[0]}.{REQUIRED_PYTHON_MIN[1]}",
        "interpreter": sys.executable,
    }


def check_packages(packages: list) -> dict:
    results = {}
    all_ok = True
    for pkg in packages:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "installed")
            results[pkg] = {"status": "PASS", "version": str(version)}
        except ImportError as e:
            results[pkg] = {"status": "FAIL", "error": str(e)}
            all_ok = False
    return {"passed": all_ok, "details": results}


def check_pytest_collection() -> dict:
    try:
        res = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = res.stdout.strip()
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        # Last line usually contains count like '48 tests collected'
        count = None
        for line in reversed(lines):
            if "collected" in line:
                parts = line.split()
                for p in parts:
                    if p.isdigit():
                        count = int(p)
                        break
                if count is not None:
                    break

        return {
            "status": "PASS" if res.returncode == 0 else "FAIL",
            "returncode": res.returncode,
            "collected_count": count,
            "summary": lines[-1] if lines else "",
        }
    except Exception as e:
        return {"status": "FAIL", "error": str(e), "collected_count": 0}


def run_preflight(check_dev: bool = True, run_collection: bool = True) -> dict:
    py_res = check_python_version()
    rt_res = check_packages(REQUIRED_RUNTIME_PACKAGES)
    dev_res = check_packages(REQUIRED_DEV_PACKAGES) if check_dev else {"passed": True, "details": {}}
    col_res = check_pytest_collection() if (run_collection and dev_res["passed"]) else {"status": "SKIPPED"}

    overall_pass = py_res["status"] == "PASS" and rt_res["passed"] and dev_res["passed"]
    if run_collection and col_res.get("status") == "FAIL":
        overall_pass = False

    return {
        "overall_status": "PASS" if overall_pass else "FAIL",
        "python": py_res,
        "runtime_packages": rt_res,
        "dev_packages": dev_res,
        "pytest_collection": col_res,
    }


def main():
    parser = argparse.ArgumentParser(description="VoiceTTSr Environment Preflight Verifier")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument("--no-dev", action="store_true", help="Skip dev/test packages")
    parser.add_argument("--no-collect", action="store_true", help="Skip pytest collection check")
    args = parser.parse_args()

    report = run_preflight(check_dev=not args.no_dev, run_collection=not args.no_collect)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("=" * 60)
        print(" VoiceTTSr Environment Preflight Verification")
        print("=" * 60)
        py = report["python"]
        print(f"Python Interpreter: {py['interpreter']}")
        print(f"Python Version:     {py['current_version']} (Requires >= {py['required_min']}) -> [{py['status']}]")
        print("-" * 60)
        print("Runtime Dependencies:")
        for pkg, det in report["runtime_packages"]["details"].items():
            ver = det.get("version", det.get("error"))
            print(f"  - {pkg:<18} [{det['status']}] {ver}")
        print("-" * 60)
        if not args.no_dev:
            print("Dev / Testing Dependencies:")
            for pkg, det in report["dev_packages"]["details"].items():
                ver = det.get("version", det.get("error"))
                print(f"  - {pkg:<18} [{det['status']}] {ver}")
            print("-" * 60)
        if not args.no_collect:
            col = report["pytest_collection"]
            print(f"Pytest Test Discovery: [{col['status']}] Collected: {col.get('collected_count')} tests")
            print("=" * 60)

        print(f"OVERALL STATUS: [{report['overall_status']}]")

    sys.exit(0 if report["overall_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
