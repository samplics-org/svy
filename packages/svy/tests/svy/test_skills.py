import os
import re

from importlib import resources
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import svy

from svy import skills


SKILLS_ROOT = resources.files("svy._skills")
SKILL_FILES = sorted((p for p in Path(str(SKILLS_ROOT)).rglob("*.md")), key=lambda p: p.as_posix())
_FENCE = re.compile(r"^```python\n(.*?)^```", re.S | re.M)


def _make_source(base: Path, name: str, body: str = "x") -> skills._Source:
    folder = base / "pkg" / "_skills" / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(f"---\nname: {name}\n---\n{body}\n")
    (folder / "topic.md").write_text("topic\n")
    return skills._Source(name, folder, f"{name}-pkg==1.0")


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("")
    monkeypatch.chdir(root)
    return root


def _use_sources(monkeypatch, *sources):
    monkeypatch.setattr(skills, "_sources", lambda: {s.name: s for s in sources})


def test_svy_registers_its_skill():
    sources = skills._sources()
    assert "svy" in sources
    assert sources["svy"].files.joinpath("SKILL.md").is_file()


def test_skill_md_frontmatter_has_name_and_description():
    text = SKILLS_ROOT.joinpath("svy", "SKILL.md").read_text()
    front = text.split("---")[1]
    assert re.search(r"^name: svy$", front, re.M)
    description = re.search(r"^description: (.+)$", front, re.M)
    assert description and len(description.group(1)) <= 1024


def test_skill_md_links_resolve():
    folder = Path(str(SKILLS_ROOT)) / "svy"
    for md in folder.glob("*.md"):
        for target in re.findall(r"\]\(([^)#:]+\.md)\)", md.read_text()):
            assert (folder / target).is_file(), f"{md.name} links to missing {target}"


# Stand in for the user's files the skills describe; deliberately not svy datasets.
def _households(rng) -> pl.DataFrame:
    rows = [
        (region, f"{region[0]}{s}", f"{region[0]}{s}-{c:02d}", hh)
        for region in ("North", "South", "East", "West")
        for s in (1, 2)
        for c in range(6)
        for hh in range(1, 6)
    ]
    hh = pl.DataFrame(rows, schema=["region", "stratum", "cluster", "hh_id"], orient="row")
    return hh.with_columns(weight=pl.Series(rng.uniform(80, 160, hh.height)))


def _persons(rng, households: pl.DataFrame) -> pl.DataFrame:
    keys = households.select("cluster", "hh_id")
    persons = pl.concat([keys, keys]).sort("cluster", "hh_id")
    n = persons.height
    age = rng.integers(15, 90, n)
    employed = (rng.random(n) < np.where(age < 65, 0.7, 0.2)).astype(int)
    return persons.with_columns(
        person_id=pl.int_range(pl.len()).over("cluster", "hh_id") + 1,
        sex=pl.Series(rng.choice(["Female", "Male"], n)),
        age=pl.Series(age),
        educ=pl.Series(rng.choice(["None", "Primary", "Secondary", "Tertiary"], n)),
        income=pl.Series(rng.lognormal(7.5, 0.6, n) * (1 + employed)),
        employed=pl.Series(employed),
        hours=pl.Series(rng.integers(10, 60, n)).scatter(np.flatnonzero(employed == 0), None),
        resp=pl.Series(
            rng.choice(["respondent", "nonrespondent", "ineligible"], n, p=[0.85, 0.12, 0.03])
        ),
    )


def _frame(rng) -> tuple[pl.DataFrame, pl.DataFrame]:
    rows = [
        (region, f"{region[0]}-{c:02d}", int(rng.integers(40, 200)))
        for region in ("North", "South", "East", "West")
        for c in range(15)
    ]
    frame = pl.DataFrame(rows, schema=["region", "cluster", "n_households"], orient="row")
    listing = pl.DataFrame(
        [(c, h) for _, c, n in rows for h in range(1, n + 1)],
        schema=["cluster", "hh_id"],
        orient="row",
    )
    return frame, listing


@pytest.fixture
def example_files(tmp_path, monkeypatch):
    rng = np.random.default_rng(2026)
    households = _households(rng)
    persons = _persons(rng, households)
    data = persons.join(households, on=["cluster", "hh_id"])
    households.write_csv(tmp_path / "households.csv")
    persons.write_csv(tmp_path / "persons.csv")
    data.write_csv(tmp_path / "survey.csv")
    frame, listing = _frame(rng)
    monkeypatch.chdir(tmp_path)
    return {"data": data, "frame": frame, "listing": listing}


@pytest.mark.parametrize("md", SKILL_FILES, ids=lambda p: p.name)
def test_skill_code_blocks_run(md, example_files, capsys):
    namespace = dict(example_files)
    for i, block in enumerate(_FENCE.findall(md.read_text())):
        try:
            exec(compile(block, f"{md.name}[block {i}]", "exec"), namespace)
        except Exception as err:
            pytest.fail(f"{md.name} block {i} failed: {type(err).__name__}: {err}\n{block}")


@pytest.mark.parametrize(
    "namespace", ["wrangling", "estimation", "categorical", "glm", "weighting", "sampling"]
)
def test_every_public_method_is_in_its_file(namespace):
    sample = svy.Sample(pl.DataFrame({"y": [1.0, 2.0]}))
    accessor = type(getattr(sample, namespace))
    methods = {
        n for n in dir(accessor) if not n.startswith("_") and callable(getattr(accessor, n))
    }
    methods -= {"set_default_print_width", "set_print_width"}
    text = (Path(str(SKILLS_ROOT)) / "svy" / f"{namespace}.md").read_text()
    missing = sorted(m for m in methods if not re.search(rf"\b{m}\b", text))
    assert not missing, f"{namespace}.md does not mention {missing}"


def test_links_skills_inside_the_project(project, monkeypatch):
    src = _make_source(project / ".venv", "alpha")
    _use_sources(monkeypatch, src)

    changes = skills.install()

    dest = project / ".claude" / "skills" / "alpha"
    assert [(c.name, c.action) for c in changes] == [("alpha", "linked")]
    assert dest.is_symlink()
    assert not Path(os.readlink(dest)).is_absolute()
    assert (dest / "topic.md").read_text() == "topic\n"


def test_copies_skills_outside_the_project(project, tmp_path, monkeypatch):
    src = _make_source(tmp_path / "cache", "alpha")
    _use_sources(monkeypatch, src)

    changes = skills.install()

    dest = project / ".claude" / "skills" / "alpha"
    assert [c.action for c in changes] == ["copied"]
    assert not dest.is_symlink()
    assert (dest / skills.MARKER).read_text().strip() == "alpha-pkg==1.0"


def test_copy_flag_copies_inside_the_project(project, monkeypatch):
    _use_sources(monkeypatch, _make_source(project / ".venv", "alpha"))

    assert [c.action for c in skills.install(copy=True)] == ["copied"]
    assert not (project / ".claude" / "skills" / "alpha").is_symlink()


@pytest.mark.parametrize("inside", [True, False])
def test_rerun_is_unchanged(project, tmp_path, monkeypatch, inside):
    base = project / ".venv" if inside else tmp_path / "cache"
    _use_sources(monkeypatch, _make_source(base, "alpha"))
    skills.install()

    assert [c.action for c in skills.install()] == ["unchanged"]


def test_copy_is_updated_when_the_version_changes(project, tmp_path, monkeypatch):
    src = _make_source(tmp_path / "cache", "alpha", body="old")
    _use_sources(monkeypatch, src)
    skills.install()

    (src.files / "SKILL.md").write_text("new\n")
    _use_sources(monkeypatch, skills._Source("alpha", src.files, "alpha-pkg==2.0"))

    assert [c.action for c in skills.install()] == ["updated"]
    assert (project / ".claude" / "skills" / "alpha" / "SKILL.md").read_text() == "new\n"


def test_relinks_after_the_venv_moves(project, monkeypatch):
    _use_sources(monkeypatch, _make_source(project / ".venv", "alpha"))
    skills.install()
    _use_sources(monkeypatch, _make_source(project / ".venv2", "alpha"))

    assert [c.action for c in skills.install()] == ["updated"]
    assert (project / ".claude" / "skills" / "alpha" / "SKILL.md").is_file()


def test_prunes_skills_no_longer_shipped(project, monkeypatch):
    _use_sources(
        monkeypatch,
        _make_source(project / ".venv", "alpha"),
        _make_source(project / ".venv2", "beta"),
    )
    skills.install()
    _use_sources(monkeypatch, skills._sources()["alpha"])

    changes = skills.install()

    assert [(c.name, c.action) for c in changes] == [("alpha", "unchanged"), ("beta", "removed")]
    assert not (project / ".claude" / "skills" / "beta").is_symlink()


def test_leaves_user_skills_alone(project, monkeypatch):
    own = project / ".claude" / "skills" / "alpha"
    own.mkdir(parents=True)
    (own / "SKILL.md").write_text("mine\n")
    other = project / ".claude" / "skills" / "mine"
    other.mkdir()
    _use_sources(monkeypatch, _make_source(project / ".venv", "alpha"))

    changes = skills.install()

    assert [c.action for c in changes] == ["skipped"]
    assert (own / "SKILL.md").read_text() == "mine\n"
    assert other.is_dir()


def test_root_is_found_from_a_subfolder(project, monkeypatch):
    _use_sources(monkeypatch, _make_source(project / ".venv", "alpha"))
    sub = project / "notebooks"
    sub.mkdir()
    monkeypatch.chdir(sub)

    skills.install()

    assert (project / ".claude" / "skills" / "alpha").is_symlink()


def test_unknown_agent_raises(project):
    with pytest.raises(ValueError, match="Unknown agent"):
        skills.install(agent="other")  # type: ignore[arg-type]


def test_cli_installs_svy_skill(project, capsys):
    assert skills.main(["--copy"]) == 0
    out = capsys.readouterr().out
    assert "copied" in out and "svy" in out
    assert (project / ".claude" / "skills" / "svy" / "SKILL.md").is_file()
