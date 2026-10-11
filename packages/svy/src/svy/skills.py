# src/svy/skills.py
"""Install the agent skills shipped by svy and its extensions into a project.

- Packages register skill folders under the ``svy.skills`` entry-point group.
- Skills are linked when the package sits inside the project, else copied.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

from dataclasses import dataclass
from importlib import metadata, resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Literal


ENTRY_POINT_GROUP = "svy.skills"
AGENT_DIRS = {"claude": Path(".claude") / "skills"}
AGENT_NAMES = {"claude": "Claude Code"}
# Marks a copied skill as ours, so reruns may replace or prune it.
MARKER = ".svy-skills"

Action = Literal["linked", "copied", "updated", "unchanged", "removed", "skipped"]


@dataclass(frozen=True)
class SkillChange:
    name: str
    action: Action
    path: Path
    source: str  # "package==version"


@dataclass(frozen=True)
class _Source:
    name: str
    files: Traversable
    origin: str


def _find_root(start: Path) -> Path:
    for folder in (start, *start.parents):
        if (folder / "pyproject.toml").exists() or (folder / ".git").exists():
            return folder
    return start


def _sources() -> dict[str, _Source]:
    found: dict[str, _Source] = {}
    for ep in metadata.entry_points(group=ENTRY_POINT_GROUP):
        origin = f"{ep.dist.name}=={ep.dist.version}" if ep.dist else ep.value
        for folder in resources.files(ep.value).iterdir():
            if not (folder.is_dir() and folder.joinpath("SKILL.md").is_file()):
                continue
            if folder.name in found:
                raise RuntimeError(
                    f"Skill {folder.name!r} is shipped by both {found[folder.name].origin} "
                    f"and {origin}."
                )
            found[folder.name] = _Source(folder.name, folder, origin)
    return found


def _is_ours(dest: Path) -> bool:
    if dest.is_symlink():
        return "_skills" in Path(os.readlink(dest)).parts
    return (dest / MARKER).is_file()


def _copy_tree(src: Traversable, dest: Path) -> None:
    dest.mkdir(parents=True)
    for item in src.iterdir():
        if item.name == "__pycache__":
            continue
        if item.is_dir():
            _copy_tree(item, dest / item.name)
        else:
            (dest / item.name).write_bytes(item.read_bytes())


def _remove(dest: Path) -> None:
    if dest.is_symlink():
        dest.unlink()
    else:
        shutil.rmtree(dest)


def _install_one(src: _Source, dest: Path, root: Path, copy: bool) -> Action:
    existed = dest.is_symlink() or dest.exists()
    if existed and not _is_ours(dest):
        return "skipped"

    src_path = Path(str(src.files)).resolve() if isinstance(src.files, Path) else None
    link_target = None
    if not copy and src_path is not None and src_path.is_relative_to(root.resolve()):
        link_target = Path(os.path.relpath(src_path, dest.parent.resolve()))

    if link_target is not None:
        if dest.is_symlink() and Path(os.readlink(dest)) == link_target:
            return "unchanged"
        if existed:
            _remove(dest)
        try:
            dest.symlink_to(link_target, target_is_directory=True)
            return "updated" if existed else "linked"
        except OSError:
            pass  # no symlink privilege (Windows): fall through to a copy

    if (
        not dest.is_symlink()
        and (dest / MARKER).is_file()
        and (dest / MARKER).read_text().strip() == src.origin
    ):
        return "unchanged"
    if dest.is_symlink() or dest.exists():
        _remove(dest)
    _copy_tree(src.files, dest)
    (dest / MARKER).write_text(src.origin + "\n")
    return "updated" if existed else "copied"


def install(
    root: str | Path | None = None,
    *,
    agent: Literal["claude"] = "claude",
    copy: bool = False,
) -> list[SkillChange]:
    """Install every registered svy skill into the agent's skills folder of a project."""
    if agent not in AGENT_DIRS:
        raise ValueError(f"Unknown agent {agent!r}; expected one of {sorted(AGENT_DIRS)}.")
    root = Path(root) if root is not None else _find_root(Path.cwd())
    skills_dir = root / AGENT_DIRS[agent]
    skills_dir.mkdir(parents=True, exist_ok=True)

    sources = _sources()
    changes = [
        SkillChange(
            name, _install_one(src, skills_dir / name, root, copy), skills_dir / name, src.origin
        )
        for name, src in sorted(sources.items())
    ]
    for dest in sorted(skills_dir.iterdir()):
        if dest.name not in sources and _is_ours(dest):
            _remove(dest)
            changes.append(SkillChange(dest.name, "removed", dest, ""))
    return changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="svy-skills",
        description="Install the agent skills shipped with svy and its extensions into a project.",
    )
    parser.add_argument(
        "--root", type=Path, help="Project root (default: nearest pyproject.toml or .git)."
    )
    parser.add_argument("--agent", choices=sorted(AGENT_DIRS), default="claude")
    parser.add_argument(
        "--copy", action="store_true", help="Copy instead of linking, e.g. to commit the skills."
    )
    args = parser.parse_args(argv)

    try:
        changes = install(args.root, agent=args.agent, copy=args.copy)
    except (OSError, RuntimeError, ValueError) as err:
        print(f"svy-skills: {err}", file=sys.stderr)
        return 1

    cwd = Path.cwd()
    for change in changes:
        shown = os.path.relpath(change.path, cwd)
        line = f"{change.action:>9}  {shown}"
        if change.action == "skipped":
            line += "  (exists and was not installed by svy-skills; left as is)"
        elif change.source:
            line += f"  ({change.source})"
        print(line)
    if not changes:
        print("No svy skills found in this environment.")
    else:
        print(f"Skills are read by {AGENT_NAMES[args.agent]} when a new session starts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
