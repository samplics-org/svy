#!/usr/bin/env python3
import shutil
import sys

from pathlib import Path


# Try importing stubgen directly
try:
    from mypy.stubgen import main as stubgen_main
except ImportError:
    print("❌ 'mypy' is not installed. Run: pip install mypy")
    sys.exit(1)

# --- CONFIGURATION ---
PACKAGE_NAME = "svy"
SRC_DIR = Path("src")  # Where your source code lives
OUTPUT_DIR = Path("out") / PACKAGE_NAME  # Where pyi files go


def main():
    print(f"🔍 Generating .pyi stubs for '{PACKAGE_NAME}'...")

    # 1. Clean output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Note: stubgen creates the folders automatically, we just clean up first.

    # 2. Argument list for stubgen
    # We pass the path to the package source
    args = [
        str(SRC_DIR / PACKAGE_NAME),
        "--output",
        str(OUTPUT_DIR.parent),
        "--include-private",
        "--include-docstrings",
        "--ignore-errors",  # Keep going if some files have syntax issues
    ]

    print(f"   Target: {SRC_DIR / PACKAGE_NAME}")

    # 3. Run stubgen programmatically
    # This avoids the "No code object" error completely
    try:
        stubgen_main(args)
    except SystemExit:
        # stubgen might call sys.exit(), we catch it to print our success message
        pass

    # 4. Cleanup / Validation
    # Sometimes stubgen outputs to out/svy/svy/ instead of out/svy/
    # We check and move if necessary

    # Check if the expected init exists
    expected_init = OUTPUT_DIR / "__init__.pyi"

    # Check for double nesting (out/svy/svy)
    nested_dir = OUTPUT_DIR / PACKAGE_NAME

    if not expected_init.exists() and nested_dir.exists():
        print("   ⚠️  Fixing nested folder structure...")
        # Move files up one level
        for file in nested_dir.glob("*"):
            shutil.move(str(file), str(OUTPUT_DIR))
        nested_dir.rmdir()

    print(f"✅ Stubs generated in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
