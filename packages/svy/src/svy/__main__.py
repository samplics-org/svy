# svy/src/svy/__main__.py
import sys

from importlib.metadata import PackageNotFoundError, version


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--version":
        # first try distribution metadata
        try:
            print(f"svy version {version('svy')}")
            return
        except PackageNotFoundError:
            pass
        # fallback: import the package and read __version__
        try:
            import svy

            print(f"svy version {getattr(svy, '__version__', 'unknown')}")
        except Exception:
            print("svy version unknown")
        return

    print("Welcome to svy! Use `svy --help` for commands (coming soon).")


if __name__ == "__main__":
    main()
