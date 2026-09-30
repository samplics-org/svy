# tests/svy/core/test_singleton_design_part.py
"""The singleton rule as a design part.

``svy.Singleton`` is declared on the design (``update_design(singleton=...)``)
and never changes because the data changed. What it does to the data (the
variance columns, ``last_result``) is derived from rule + data whenever either
changed: the rule is applied to the singletons found now. An explicit collapse
mapping that no longer fits the data raises at the next Taylor analysis;
entries for strata that are no longer singletons are ignored (INFO).

``BASELINE_SES`` were computed before the rule was declarative (commit
3c38a29, the tip of feat/design-serialization) with the same data and calls,
and pin the numbers. Where an independent reference exists (collapse, pool and
self_representing are strata recodes) it is checked too.
"""

from __future__ import annotations

import datetime as dt
import gc
import warnings

import polars as pl
import pytest

import svy

from svy.core.design import Singleton
from svy.core.enumerations import SingletonMethod
from svy.core.singleton import _VAR_EXCLUDE_COL, _VAR_PSU_COL, _VAR_STRATUM_COL
from svy.errors import SerializationError
from svy.errors.singleton_errors import SingletonError
from svy.serialize import from_json, to_design, to_json


def _result(sample):
    """What the design's singleton rule did to the current data (internal)."""
    sample._sync_parts()
    return sample._singleton_result


def _detected(sample):
    """The singleton strata of the current data (internal)."""
    from svy.core.singleton import _Engine

    sample._sync_parts()
    return _Engine(sample, _sync=False).detected()


def _keys(sample):
    """svy's keys of the singleton strata of the current data (internal)."""
    return [s.stratum_key for s in _detected(sample)]


def _declare(sample, method, **kw):
    """A fork of ``sample`` with the singleton rule declared on its design."""
    new = sample._fork()
    new.update_design(singleton=Singleton(method, **kw))
    return new


def resolved(h: svy.Sample):
    """What the rule did to the data, in the strata's values: the collapse
    mapping, or the singleton strata handled."""
    from svy.core.singleton import _Engine

    lr, f = _result(h), _Engine(h, _sync=False)
    if lr is None:
        return None
    if isinstance(lr.applied, dict) and lr.method == SingletonMethod.COLLAPSE:
        keys = list(lr.applied)
        return dict(zip(f._key_values(keys), f._key_values([lr.applied[k] for k in keys])))
    return f._key_values([s.stratum_key for s in lr.detected])


def _problem(h: svy.Sample) -> str:
    """The code of the error a Taylor analysis raises for the rule."""
    with pytest.raises(SingletonError) as err:
        h.estimation.mean("y")
    return err.value.code


pytestmark = pytest.mark.filterwarnings("error")

# ---------------------------------------------------------------------------
# Data: strata a, b, c with 3 PSUs, d and e with one; 3 rows per PSU
# ---------------------------------------------------------------------------

STRATA = {"a": 3, "b": 3, "c": 3, "d": 1, "e": 1}
REGION = {"a": "N", "b": "N", "c": "S", "d": "N", "e": "S"}
CODE = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}


def make_frame(strata: dict[str, int] = STRATA) -> pl.DataFrame:
    rows = []
    i = 0
    for st, n_psu in strata.items():
        for p in range(n_psu):
            for r in range(3):
                i += 1
                rows.append(
                    {
                        "id": i,
                        "st": st,
                        "reg": REGION.get(st, "N"),
                        "code": CODE.get(st, 9),
                        "psu": f"{st}{p}",
                        "hh": f"{st}{p}h{r % 2}",
                        "y": float((i * 37) % 17 + 3 * p + CODE.get(st, 9)),
                        "x": int((i * 7) % 3 == 0),
                        "dom": "u" if i % 2 else "v",
                        "w": 1.0 + (i % 4) * 0.5,
                    }
                )
    return pl.DataFrame(rows)


DATA = make_frame()

DESIGNS = {
    "single": dict(stratum="st", psu="psu", wgt="w"),
    "tuple": dict(stratum=("reg", "code"), psu="psu", wgt="w"),
    "ssu": dict(stratum="st", psu="psu", ssu="hh", wgt="w"),
}


def sample(kind: str = "single", data: pl.DataFrame = DATA) -> svy.Sample:
    return svy.Sample(data, svy.Design(**DESIGNS[kind]))


def _key(kind: str, st: str) -> str:
    return f"{REGION[st]}__by__{CODE[st]}" if kind == "tuple" else st


def _last(singleton, candidates):
    return sorted(c.stratum_key for c in candidates)[-1]


def rules(kind: str) -> dict:
    k = lambda s: _key(kind, s)  # noqa: E731
    return {
        "certainty": lambda s: _declare(s, "self_representing"),
        "skip": lambda s: _declare(s, "skip"),
        "scale": lambda s: _declare(s, "scale"),
        "center": lambda s: _declare(s, "center"),
        "pool": lambda s: _declare(s, "pool"),
        "collapse_dict": lambda s: _declare(s, "collapse", using={k("d"): k("a"), k("e"): k("c")}),
        "collapse_smallest": lambda s: _declare(s, "collapse"),
        "collapse_callable": lambda s: _declare(s, "collapse", using=_last),
        "collapse_rstate": lambda s: _declare(s, "collapse", rstate=3),
        "collapse_largest_rstate": lambda s: _declare(s, "collapse", using="largest", rstate=11),
    }


RULES = list(rules("single"))


def estimates(s: svy.Sample, *, by: str = "dom") -> list:
    out = []
    for e in (
        s.estimation.mean("y"),
        s.estimation.total("y"),
        s.estimation.mean("y", by=by),
        s.estimation.prop("x"),
    ):
        out.append(sorted([str(p.by_level), str(p.y_level), p.est, p.se] for p in e.estimates))
    return out


def ses(s: svy.Sample) -> list[float]:
    return [r[3] for e in estimates(s) for r in e]


def se(s: svy.Sample) -> float:
    return s.estimation.mean("y").estimates[0].se


# Computed before the change; see the module docstring. Tuple strata give the
# single-column numbers exactly.
BASELINE_SES = {
    "single/certainty": [1.017349576500428, 65.38921929492659, 1.5533085841827028, 1.0379737781096448, 0.05308376278020675, 0.05308376278020675],
    "single/skip": [0.9665528987379829, 62.853003110432205, 1.4973277889710257, 0.8750482239797549, 0.0327528934148409, 0.0327528934148409],
    "single/scale": [1.2478144266802371, 81.1428781019423, 1.933041863499452, 1.1296823995339895, 0.04228380357859513, 0.04228380357859513],
    "single/center": [1.0087553086714105, 66.90986289080344, 1.553020307019563, 0.8849808295510357, 0.033870678872476426, 0.033870678872476426],
    "single/pool": [1.049260574153878, 69.64553108419807, 1.6060894411650188, 0.8921043729246919, 0.03478671792937147, 0.03478671792937147],
    "single/collapse_dict": [0.9135734860304381, 60.543372882587235, 1.4531647810170631, 0.9056067676100659, 0.03209233833375578, 0.03209233833375578],
    "single/collapse_smallest": [0.9605878543305235, 62.791984626489814, 1.4890640399754769, 0.90300700437403, 0.03206931789905218, 0.03206931789905218],
    "single/collapse_callable": [0.9897367489718996, 67.54998149518622, 1.4964190469875422, 0.9274733886069583, 0.033850627369727396, 0.033850627369727396],
    "single/collapse_rstate": [1.013526041618512, 68.88033100965761, 1.4938125793648291, 0.9364598997157967, 0.03351075977128117, 0.03351075977128117],
    "single/collapse_largest_rstate": [1.0246477697678817, 67.19747019047666, 1.5351333985999913, 0.9703168332789266, 0.033785113373656035, 0.033785113373656035],
    "ssu/certainty": [0.9856814888905344, 82.56209784156408, 1.553020307019563, 0.8849808295510357, 0.04238790599069629, 0.04238790599069629],
    "wgt/collapse_dict/ps": [0.8584438249918355, 98.72103987406109, 1.4531647810170631, 0.9056067676100659, 0.03123418936266654, 0.03123418936266655],
    "wgt/collapse_dict/rake": [0.8557909311347657, 98.41595708049803, 1.4735997335465494, 0.8661299491821834, 0.030869038500611987, 0.030869038500612],
    "wgt/collapse_dict/trim": [0.8656787343187885, 52.49526835751856, 1.444471247363955, 0.9052102477435969, 0.018498848625519232, 0.018498848625519232],
    "wgt/skip/ps": [0.9090799332492724, 104.54419232366632, 1.4973277889710257, 0.875048223979755, 0.030877296824938266, 0.030877296824938272],
    "wgt/skip/rake": [0.9051536131853717, 104.09266551631774, 1.5119393146881301, 0.8496110778496276, 0.030487836121857872, 0.03048783612185788],
    "wgt/skip/trim": [0.913940657743033, 55.249520039322036, 1.4602619777808374, 0.8716584757488833, 0.018223001614190253, 0.018223001614190253],
    "wgt/certainty/ps": [0.9654172954885231, 111.02298898118016, 1.5533085841827028, 1.0379737781096448, 0.052879845278146966, 0.05287984527814697],
    "wgt/certainty/rake": [0.9584973314939131, 110.2271931218, 1.569507399217594, 1.0062066877398943, 0.052995493808961375, 0.05299549380896138],
    "wgt/certainty/trim": [0.9852784307879222, 57.84299374483537, 1.5471860173362506, 1.0434765590453687, 0.047473972219589766, 0.047473972219589766],
    "wgt/scale/ps": [1.1736171472819275, 134.96597193742164, 1.933041863499452, 1.1296823995339895, 0.03986241879296302, 0.039862418792963025],
    "wgt/scale/rake": [1.1685482898754715, 134.38305333567922, 1.9519052620877944, 1.0968431850883136, 0.03935962718728549, 0.039359627187285494],
    "wgt/scale/trim": [1.1798923156202066, 71.32682366608239, 1.8851901070150294, 1.1253062533853222, 0.02352579392322324, 0.02352579392322324],
    "wgt/center/ps": [0.9492200269500519, 109.16030309925597, 1.553020307019563, 0.8849808295510359, 0.03344165625770566, 0.03344165625770567],
    "wgt/center/rake": [0.9140001008393405, 105.11001159652413, 1.536084602997663, 0.8502276943253662, 0.03315029798514637, 0.03315029798514638],
    "wgt/center/trim": [0.9700347930955088, 58.429836306259254, 1.5480353796663653, 0.881707538275757, 0.018993617646328342, 0.018993617646328342],
    "wgt/pool/ps": [0.9876930212816073, 113.58469744738484, 1.6060894411650188, 0.892104372924692, 0.035323173296679194, 0.0353231732966792],
    "wgt/pool/rake": [0.9216909460055349, 105.9944587906365, 1.5577345581304207, 0.8508334868378064, 0.035172958991434716, 0.03517295899143473],
    "wgt/pool/trim": [1.0226575132611402, 61.22589480521978, 1.6303763172441395, 0.8827699356586985, 0.01862366203249392, 0.01862366203249392],
}  # fmt: skip


def _pin(kind: str, rule: str) -> list[float]:
    return BASELINE_SES[
        "ssu/certainty" if (kind, rule) == ("ssu", "certainty") else f"single/{rule}"
    ]


def _saved(s: svy.Sample) -> svy.Design:
    return to_design(from_json(to_json(s.design)))


def _coded(s: svy.Sample) -> svy.Design:
    return eval(s.design._to_code(), {"svy": svy, "datetime": dt})


SAVABLE = [r for r in RULES if r != "collapse_callable"]


# ---------------------------------------------------------------------------
# Every method: the pinned numbers, save/restore, code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(DESIGNS))
@pytest.mark.parametrize("rule", RULES)
def test_estimates_are_pinned(kind, rule):
    h = rules(kind)[rule](sample(kind))
    assert ses(h) == pytest.approx(_pin(kind, rule), rel=1e-12)


@pytest.mark.parametrize("kind", list(DESIGNS))
@pytest.mark.parametrize("rule", SAVABLE)
def test_estimates_survive_save_restore_and_code(kind, rule):
    h = rules(kind)[rule](sample(kind))
    live = estimates(h)
    for design in (_saved(h), _coded(h)):
        assert design == h.design
        assert design.singleton == h.design.singleton
        # With the sample's frame and the original.
        assert estimates(svy.Sample(h.data, design)) == live
        assert estimates(svy.Sample(DATA, design)) == live


@pytest.mark.parametrize("kind", list(DESIGNS))
@pytest.mark.parametrize("rule", RULES)
def test_the_rule_is_on_the_design_and_in_the_history(kind, rule):
    s = sample(kind)
    h = rules(kind)[rule](s)
    rule_ = h.design.singleton
    assert isinstance(rule_, Singleton)
    assert h.design_history[-2].singleton is None
    assert h.design_history[-1].singleton == rule_
    assert s.design.singleton is None
    assert len(s.design_history) == 1


def test_the_rule_is_stored_and_what_it_did_derived():
    got = {rule: rules("single")[rule](sample()) for rule in RULES}
    assert got["certainty"].design.singleton == Singleton("self_representing")
    assert got["collapse_smallest"].design.singleton == Singleton("collapse")
    assert got["collapse_rstate"].design.singleton == Singleton("collapse", rstate=3)
    for rule in ("certainty", "skip", "scale", "center", "pool"):
        assert resolved(got[rule]) == ["d", "e"]
    assert resolved(got["collapse_dict"]) == {"d": "a", "e": "c"}
    assert resolved(got["collapse_smallest"]) == {"d": "a", "e": "b"}
    assert resolved(got["collapse_callable"]) == {"d": "c", "e": "c"}
    assert resolved(got["collapse_rstate"]) == {"d": "c", "e": "b"}
    assert resolved(got["collapse_largest_rstate"]) == {"d": "a", "e": "a"}
    tup = rules("tuple")["collapse_rstate"](sample("tuple"))
    assert resolved(tup) == {("N", 4): ("S", 3), ("S", 5): ("N", 2)}


def test_an_int_rstate_resolves_the_same_on_every_refresh():
    h = _declare(sample(), "collapse", rstate=3)
    r = h.wrangling.filter_records(svy.col("id") != 1)
    assert resolved(r) == resolved(h) == {"d": "c", "e": "b"}
    assert resolved(svy.Sample(DATA, _saved(h))) == {"d": "c", "e": "b"}


def test_a_callable_is_called_on_each_resolution_and_cannot_be_saved():
    calls = []

    def pick(singleton, candidates):
        calls.append(singleton.stratum_key)
        return sorted(c.stratum_key for c in candidates)[0]

    h = _declare(sample(), "collapse", using=pick)
    assert calls == ["d", "e"]
    h.wrangling.filter_records(svy.col("id") != 1).estimation.mean("y")
    assert calls == ["d", "e", "d", "e"]
    with pytest.raises(SerializationError, match="callable"):
        _saved(h)


# ---------------------------------------------------------------------------
# Independent references: collapse, pool and self_representing are recodes
# ---------------------------------------------------------------------------

_LONE = pl.col("st").is_in(["d", "e"])


def _plain(data: pl.DataFrame, **kw) -> svy.Sample:
    return svy.Sample(data, svy.Design(**{**DESIGNS["single"], **kw}))


REFERENCES = {
    "collapse": lambda df: df.with_columns(pl.col("st").replace({"d": "a", "e": "c"})),
    "pool": lambda df: df.with_columns(
        pl.col("st").replace({"d": "__pooled__", "e": "__pooled__"})
    ),
    "self_representing": lambda df: df.with_columns(
        pl.when(_LONE).then(pl.col("psu")).otherwise(pl.col("st")).alias("st"),
        pl.when(_LONE).then(pl.col("id").cast(pl.Utf8)).otherwise(pl.col("psu")).alias("psu"),
    ),
}


def test_collapse_is_the_recoded_strata():
    h = _declare(sample(), "collapse", using={"d": "a", "e": "c"})
    assert ses(h) == pytest.approx(ses(_plain(REFERENCES["collapse"](DATA))), rel=1e-12)


def test_pool_is_the_recoded_strata():
    ref = _plain(REFERENCES["pool"](DATA))
    assert ses(_declare(sample(), "pool")) == pytest.approx(ses(ref), rel=1e-12)


def test_self_representing_makes_rows_the_psus():
    ref = _plain(REFERENCES["self_representing"](DATA))
    assert ses(_declare(sample(), "self_representing")) == pytest.approx(ses(ref), rel=1e-12)


def test_self_representing_makes_ssus_the_psus():
    ref = _plain(
        DATA.with_columns(
            pl.when(_LONE).then(pl.col("psu")).otherwise(pl.col("st")).alias("st"),
            pl.when(_LONE).then(pl.col("hh")).otherwise(pl.col("psu")).alias("psu"),
        )
    )
    assert ses(_declare(sample("ssu"), "self_representing")) == pytest.approx(ses(ref), rel=1e-12)


# ---------------------------------------------------------------------------
# The stale-cache bug
# ---------------------------------------------------------------------------

STALE = make_frame({"a": 3, "b": 3, "c": 3, "d": 1})


def test_collapse_then_recode_strata_gives_the_fresh_sample():
    s = svy.Sample(STALE, svy.Design(stratum="st", psu="psu", wgt="w"))
    h = _declare(s, "collapse", using={"d": "a"})
    before = se(h)
    r = h.wrangling.recode("st", {"c": ["b", "c"]}, replace=True)
    fresh = svy.Sample(
        STALE.with_columns(pl.col("st").replace({"d": "a", "b": "c"})),
        svy.Design(stratum="st", psu="psu", wgt="w"),
    )
    assert se(r) == pytest.approx(se(fresh), rel=1e-12)
    assert se(r) != pytest.approx(before)
    assert r.design.singleton == Singleton("collapse", using={"d": "a"})
    assert ses(r) == pytest.approx(ses(fresh), rel=1e-12)


def test_collapse_then_recode_inplace():
    s = svy.Sample(STALE, svy.Design(stratum="st", psu="psu", wgt="w"))
    h = _declare(s, "collapse", using={"d": "a"})
    h.wrangling.recode("st", {"c": ["b", "c"]}, replace=True, inplace=True)
    fresh = svy.Sample(
        STALE.with_columns(pl.col("st").replace({"d": "a", "b": "c"})),
        svy.Design(stratum="st", psu="psu", wgt="w"),
    )
    assert se(h) == pytest.approx(se(fresh), rel=1e-12)


# ---------------------------------------------------------------------------
# Changes to the data: the rule stays and is applied again
# ---------------------------------------------------------------------------

RULE = Singleton("collapse", using={"d": "a", "e": "c"})


def handled() -> svy.Sample:
    return _declare(sample(), "collapse", using={"d": "a", "e": "c"})


def _fresh(r: svy.Sample, rule: Singleton = RULE) -> svy.Sample:
    """The rule declared afresh on the changed sample's data and design."""
    return svy.Sample(r.data, r.design.update(singleton=rule))


def _split_d(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when((pl.col("st") == "d") & (pl.col("id") % 2 == 0))
        .then(pl.lit("d1"))
        .otherwise(pl.col("psu"))
        .alias("psu")
    )


SAME_SINGLETONS = {
    "recode_other": lambda s, ip: s.wrangling.recode(
        "dom", {"U": ["u"]}, replace=True, inplace=ip
    ),
    "recode_strata_same_singletons": lambda s, ip: s.wrangling.recode(
        "st", {"a": ["a", "b"]}, replace=True, inplace=ip
    ),
    "filter_rows_not_psus": lambda s, ip: s.wrangling.filter_records(
        svy.col("id") != 1, inplace=ip
    ),
    "filter_whole_psu": lambda s, ip: s.wrangling.filter_records(
        svy.col("psu") != "b0", inplace=ip
    ),
    "cast_strata": lambda s, ip: s.wrangling.cast("st", pl.Categorical, inplace=ip),
    "fill_null_strata": lambda s, ip: s.wrangling.fill_null("st", "z", inplace=ip),
    "mutate_other": lambda s, ip: s.wrangling.mutate({"y2": pl.col("y") * 2}, inplace=ip),
    "join": lambda s, ip: s.wrangling.join(
        pl.DataFrame({"id": DATA["id"], "extra": DATA["y"] + 1}), on="id", inplace=ip
    ),
    "order_by": lambda s, ip: s.wrangling.order_by("y", inplace=ip),
    "with_row_index": lambda s, ip: s.wrangling.with_row_index("rix", inplace=ip),
    "remove_other": lambda s, ip: s.wrangling.remove_columns("hh", inplace=ip),
}


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("trigger", list(SAME_SINGLETONS))
def test_changes_that_keep_the_singletons_keep_the_numbers(trigger, inplace):
    h = handled()
    r = SAME_SINGLETONS[trigger](h, inplace)
    assert (r is h) is inplace
    assert r.design.singleton == RULE
    assert ses(r) == pytest.approx(ses(_fresh(r)), rel=1e-12)
    assert _result(r).config.stratum_mapping == {"d": "a", "e": "c"}


# change -> the singletons the rule finds now, or the code the next Taylor
# analysis raises, and the mapping entries it ignores
OTHER_SINGLETONS = {
    "split_the_singleton_psu": (
        lambda s, ip: s.wrangling.mutate(
            {
                "psu": pl.when((pl.col("st") == "d") & (pl.col("id") % 2 == 0))
                .then(pl.lit("d1"))
                .otherwise(pl.col("psu"))
            },
            inplace=ip,
        ),
        {"e": "c"},
        ["d"],
    ),
    "recode_away_the_target": (
        lambda s, ip: s.wrangling.recode("st", {"b": ["b", "c"]}, replace=True, inplace=ip),
        "SINGLETON_TARGET_MISSING",
        [],
    ),
    "recode_psu_values": (
        lambda s, ip: s.wrangling.recode(
            "psu", {"b0": ["b0", "b1", "b2"]}, replace=True, inplace=ip
        ),
        "SINGLETON_UNMAPPED",
        [],
    ),
    "filter_away_a_handled_singleton": (
        lambda s, ip: s.wrangling.filter_records(svy.col("st") != "e", inplace=ip),
        {"d": "a"},
        ["e"],
    ),
    "filter_creates_a_singleton": (
        lambda s, ip: s.wrangling.filter_records(
            svy.col("psu").is_in(["b0", "b1"]), negate=True, inplace=ip
        ),
        "SINGLETON_UNMAPPED",
        [],
    ),
    "filter_all_singletons_away": (
        lambda s, ip: s.wrangling.filter_records(
            svy.col("st").is_in(["d", "e"]), negate=True, inplace=ip
        ),
        None,
        [],
    ),
    "mutate_merges_the_singletons": (
        lambda s, ip: s.wrangling.mutate(
            {"st": pl.when(pl.col("st") == "e").then(pl.lit("d")).otherwise(pl.col("st"))},
            inplace=ip,
        ),
        None,
        [],
    ),
}


def _unused(r: svy.Sample) -> list[str]:
    return [
        v for w in r.warnings if w.code == "SINGLETON_MAPPING_UNUSED" for v in w.extra["strata"]
    ]


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("trigger", list(OTHER_SINGLETONS))
def test_changes_to_the_singletons_apply_the_rule_again(trigger, inplace):
    change, outcome, unused = OTHER_SINGLETONS[trigger]
    h = handled()
    r = change(h, inplace)
    # Silent: no Python warning (pytestmark turns one into an error).
    assert r.design.singleton == RULE
    assert [d.singleton for d in r.design_history] == [None, RULE]
    if isinstance(outcome, str):
        assert _problem(r) == outcome
        assert _result(r) is None
        # The same rule on a fresh sample fails the same way.
        assert _problem(_fresh(r)) == outcome
        return
    assert resolved(r) == outcome
    assert ses(r) == pytest.approx(ses(_fresh(r)), rel=1e-12)
    assert _unused(r) == unused


def test_the_error_is_raised_at_every_taylor_analysis_until_the_rule_changes():
    r = handled().wrangling.filter_records(svy.col("psu").is_in(["b0", "b1"]), negate=True)
    for _ in range(3):
        assert _problem(r) == "SINGLETON_UNMAPPED"
        r.design
        r.design_history
    with pytest.raises(SingletonError, match="not in the collapse mapping: 'b'") as err:
        r.categorical.tabulate("x")
    assert err.value.where == "Sample.categorical.tabulate"
    assert "sample.singletons lists the singletons" in err.value.hint
    # Inspection does not raise.
    assert _keys(r) == ["b", "d", "e"]
    r.update_design(singleton=Singleton("collapse", using={"b": "a", "d": "a", "e": "c"}))
    assert resolved(r) == {"b": "a", "d": "a", "e": "c"}
    r.estimation.mean("y")


def test_a_collapse_strategy_records_a_changed_mapping():
    h = _declare(sample(), "collapse")
    assert resolved(h) == {"d": "a", "e": "b"}
    r = h.wrangling.filter_records(svy.col("st") != "b")
    assert resolved(r) == {"d": "a", "e": "c"}
    (w,) = [w for w in r.warnings if w.code == "SINGLETON_COLLAPSE_CHANGED"]
    assert "'d' -> 'a', 'e' -> 'c'" in w.detail
    assert w.extra["mapping"] == {"d": "a", "e": "c"}
    # Unchanged mapping: nothing recorded.
    q = h.wrangling.filter_records(svy.col("id") != 1)
    assert not [w for w in q.warnings if w.code == "SINGLETON_COLLAPSE_CHANGED"]


@pytest.mark.parametrize("inplace", [False, True])
def test_a_cast_that_merges_strata_leaves_the_rule_idle(inplace):
    df = DATA.with_columns(
        pl.col("st").replace_strict({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 4.4}).alias("g")
    )
    h = _declare(svy.Sample(df, svy.Design(stratum="g", psu="psu", wgt="w")), "skip")
    assert resolved(h) == [4.0, 4.4]
    r = h.wrangling.cast("g", pl.Int64, strict=False, inplace=inplace)
    # 4.0 and 4.4 are both 4 now: one stratum with two PSUs.
    assert r.design.singleton == Singleton("skip")
    assert _result(r) is None


@pytest.mark.parametrize("inplace", [False, True])
def test_a_fill_null_that_merges_strata_leaves_the_rule_idle(inplace):
    df = DATA.with_columns(
        pl.when(pl.col("st") == "e").then(None).otherwise(pl.col("st")).alias("st")
    )
    h = _declare(svy.Sample(df, svy.Design(**DESIGNS["single"])), "skip")
    assert resolved(h) == [None, "d"]
    r = h.wrangling.fill_null("st", "d", inplace=inplace)
    assert r.design.singleton == Singleton("skip") and _result(r) is None


@pytest.mark.parametrize("inplace", [False, True])
def test_a_fill_null_elsewhere_keeps_the_rule(inplace):
    df = DATA.with_columns(pl.when(pl.col("id") == 1).then(None).otherwise(pl.col("x")).alias("x"))
    h = _declare(svy.Sample(df, svy.Design(**DESIGNS["single"])), "skip")
    r = h.wrangling.fill_null("x", 0, inplace=inplace)
    assert r.design.singleton == Singleton("skip") and resolved(r) == ["d", "e"]


@pytest.mark.parametrize("method", ["self_representing", "skip", "scale", "center", "pool"])
def test_every_rule_handles_a_new_singleton(method):
    h = _declare(sample(), method)
    r = h.wrangling.filter_records(svy.col("psu").is_in(["b0", "b1"]), negate=True)
    assert r.design.singleton == Singleton(method)
    assert resolved(r) == ["b", "d", "e"]
    assert ses(r) == pytest.approx(ses(_fresh(r, Singleton(method))), rel=1e-12)


@pytest.mark.parametrize("method", ["self_representing", "skip", "scale", "center", "pool"])
def test_every_rule_is_kept_by_an_unrelated_filter(method):
    h = _declare(sample(), method)
    r = h.wrangling.filter_records(svy.col("psu") != "a2")
    assert r.design.singleton == h.design.singleton
    assert ses(r) == pytest.approx(ses(_fresh(r, Singleton(method))), rel=1e-12)


def test_declared_before_the_singletons_exist():
    clean = make_frame({"a": 3, "b": 2, "c": 2})
    s = svy.Sample(clean, svy.Design(**DESIGNS["single"], singleton="center"))
    assert _result(s) is None
    assert ses(s) == pytest.approx(ses(_plain(clean)), rel=1e-12)
    r = s.wrangling.filter_records(svy.col("psu") != "b1")
    assert resolved(r) == ["b"]
    assert ses(r) == pytest.approx(ses(_fresh(r, Singleton("center"))), rel=1e-12)


# set_data / update_data / clone


@pytest.mark.parametrize("setter", ["set_data", "update_data"])
def test_set_data_with_other_singletons_applies_the_rule_to_them(setter):
    h = handled()
    getattr(h, setter)(_split_d(h.data))
    assert h.design.singleton == RULE
    assert resolved(h) == {"e": "c"}
    assert _unused(h) == ["d"]


@pytest.mark.parametrize("setter", ["set_data", "update_data"])
def test_set_data_with_the_same_singletons_keeps_the_numbers(setter):
    h = handled()
    getattr(h, setter)(h.data.with_columns((pl.col("y") + 1).alias("y")))
    assert h.design.singleton == RULE
    assert se(h) == pytest.approx(BASELINE_SES["single/collapse_dict"][0], rel=1e-12)


def test_clone_with_other_singletons():
    h = handled()
    c = h.clone(data=_split_d(h.data))
    assert c.design.singleton == RULE and resolved(c) == {"e": "c"}
    assert resolved(h) == {"d": "a", "e": "c"}


# Design edits: the rule is column-agnostic intent


@pytest.mark.parametrize(
    "edit, outcome",
    [
        ({"stratum": "reg"}, None),
        ({"stratum": ("reg", "st")}, "SINGLETON_UNMAPPED"),
        ({"psu": "hh"}, None),
        ({"psu": None}, None),
        ({"ssu": "hh"}, {"d": "a", "e": "c"}),
    ],
)
def test_a_design_edit_keeps_the_rule_and_applies_it_again(edit, outcome):
    h = handled()
    h.update_design(**edit)
    assert h.design.singleton == RULE
    assert h.design_history[-2].singleton == RULE
    if isinstance(outcome, str):
        assert _problem(h) == outcome
    else:
        assert resolved(h) == outcome


def test_a_new_rule_with_the_edit():
    h = handled()
    rule = Singleton("collapse", using={("N", "d"): ("N", "a"), ("S", "e"): ("S", "c")})
    h.update_design(stratum=("reg", "st"), singleton=rule)
    assert resolved(h) == {("N", "d"): ("N", "a"), ("S", "e"): ("S", "c")}
    assert ses(h) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)


def test_a_tuple_and_back_brings_the_numbers_back():
    h = handled()
    h.update_design(stratum=("st", "reg"))
    h.update_design(stratum="st")
    assert h.design.singleton == RULE
    assert ses(h) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)


def test_other_design_edits_keep_the_rule():
    h = handled()
    h.update_design(prob="w", mos="y")
    assert h.design.singleton == RULE
    h.update_design(wgt="y")
    assert h.design.singleton == RULE


def test_removing_the_stratum_drops_the_rule_with_one_warning():
    h = handled()
    with pytest.warns(UserWarning, match=r"singleton rule \(collapse\) removed") as rec:
        h.update_design(stratum=None)
    assert len(rec) == 1 and rec[0].filename == __file__
    assert h.design.singleton is None


def test_forced_removal_of_the_ssu_keeps_the_rule():
    h = _declare(sample("ssu"), "self_representing")
    with pytest.warns(UserWarning) as rec:
        r = h.wrangling.remove_columns("hh", force=True)
    assert len(rec) == 1 and "ssu='hh'" in str(rec[0].message)
    assert "singleton" not in str(rec[0].message)
    assert r.design.singleton == Singleton("self_representing")
    # Without an SSU the singletons' rows are the PSUs.
    assert ses(r) == pytest.approx(BASELINE_SES["single/certainty"], rel=1e-12)


def test_forced_removal_of_the_stratum_drops_the_rule_in_the_one_warning():
    h = handled()
    with pytest.warns(UserWarning) as rec:
        r = h.wrangling.remove_columns("st", force=True)
    assert len(rec) == 1
    assert "stratum='st'" in str(rec[0].message)
    assert "singleton rule (collapse)" in str(rec[0].message)
    assert r.design.singleton is None


def test_a_within_column_is_protected_and_follows_renames():
    h = _declare(sample(), "collapse", within="reg")
    assert resolved(h) == {"d": "a", "e": "c"}
    with pytest.raises(svy.MethodError):
        h.wrangling.remove_columns("reg")
    r = h.wrangling.rename_columns({"reg": "region"})
    assert r.design.singleton == Singleton("collapse", within="region")
    assert resolved(r) == {"d": "a", "e": "c"}
    with pytest.warns(UserWarning, match=r"singleton rule \(collapse; reads region\)"):
        q = r.wrangling.remove_columns("region", force=True)
    assert q.design.singleton is None


def test_within_constrains_the_targets_to_the_same_region():
    # Without within, e (region S) goes to the smallest stratum, a (region N).
    assert resolved(_declare(sample(), "collapse")) == {"d": "a", "e": "b"}
    assert resolved(_declare(sample(), "collapse", within="reg")) == {"d": "a", "e": "c"}
    ref = _plain(DATA.with_columns(pl.col("st").replace({"d": "a", "e": "c"})))
    h = _declare(sample(), "collapse", within="reg")
    assert ses(h) == pytest.approx(ses(ref), rel=1e-12)


def test_within_must_be_constant_within_a_stratum():
    h = _declare(sample(), "collapse", within="dom")
    assert _problem(h) == "SINGLETON_WITHIN_INVALID"


# Renames carry the rule: strata are values, within/order_by columns follow


@pytest.mark.parametrize("inplace", [False, True])
def test_renaming_the_strata_and_psu_carries_the_rule(inplace):
    h = handled()
    r = h.wrangling.rename_columns({"st": "stratum2", "psu": "cluster"}, inplace=inplace)
    assert r.design.stratum == "stratum2" and r.design.psu == "cluster"
    assert r.design.singleton == RULE
    assert r.design_history[-1].singleton == RULE
    assert ses(r) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)


def test_clean_names_carries_the_rule():
    h = _declare(
        svy.Sample(
            DATA.rename({"st": "Strata Col"}),
            svy.Design(stratum="Strata Col", psu="psu", wgt="w"),
        ),
        "collapse",
        using={"d": "a", "e": "c"},
    )
    r = h.wrangling.clean_names()
    assert r.design.stratum == "strata_col"
    assert r.design.singleton == RULE
    assert ses(r) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)


def test_rename_rep_wgts_has_no_effect_on_the_rule():
    h = handled().weighting.create_bs_wgts(n_reps=4, rstate=1)
    prefix = h.design.rep_wgts.prefix
    r = h.wrangling.rename_rep_wgts({prefix: "boot"})
    assert r.design.rep_wgts.prefix == "boot"
    assert r.design.singleton == RULE
    assert ses(r) == pytest.approx(ses(h), rel=1e-12)


# ---------------------------------------------------------------------------
# Forks, inplace, caching
# ---------------------------------------------------------------------------


def test_forks_resolve_on_their_own_data():
    h = handled()
    f = h.wrangling.filter_records(svy.col("st") != "e")
    assert resolved(f) == {"d": "a"}
    assert resolved(h) == {"d": "a", "e": "c"}
    assert ses(h) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)
    # And the other way round: the parent changes inplace, the fork is intact.
    g = h.wrangling.filter_records(svy.col("id") != 1)
    h.wrangling.filter_records(svy.col("psu").is_in(["b0", "b1"]), negate=True, inplace=True)
    assert _problem(h) == "SINGLETON_UNMAPPED"
    assert g.design.singleton == RULE
    assert ses(g) == pytest.approx(ses(_fresh(g)), rel=1e-12)


def test_forks_do_not_share_derived_state():
    h = handled()
    h.estimation.mean("y")
    f = h.wrangling.filter_records(svy.col("psu") != "a2")
    assert f._data.height != h._data.height
    f.estimation.mean("y")
    assert h._data.height == DATA.height
    assert _result(h).n_psus_before == 11
    assert _result(f).n_psus_before == 10


def test_the_resolution_is_cached_per_version():
    h = handled()
    h.estimation.mean("y")
    stamp = h.__dict__["_parts_stamp"]
    data = h._data
    h.estimation.total("y")
    h.design
    assert h.__dict__["_parts_stamp"] is stamp
    assert h._data is data


def test_version_keying_survives_many_short_lived_samples():
    """Designs and samples freed and reallocated (ids reused) never carry
    another sample's derived state: the stamp holds the design itself and a
    process-wide data version."""
    small = make_frame({"a": 2, "b": 2, "d": 1})
    base = _declare(
        svy.Sample(small, svy.Design(**DESIGNS["single"])), "collapse", using={"d": "a"}
    )
    for _ in range(15):
        f = base.wrangling.filter_records(svy.col("st") != "d")
        assert _result(f) is None
        del f
        gc.collect()
        g = svy.Sample(small, base.design)
        assert _result(g).applied == {"d": "a"}
        del g
    assert _result(base).applied == {"d": "a"}


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def test_history_entries_carry_their_own_rule():
    h = _declare(_declare(sample(), "collapse", using={"d": "a", "e": "c"}), "skip")
    assert [d.singleton for d in h.design_history] == [None, RULE, Singleton("skip")]


def test_restoring_a_design_from_before_the_rule():
    h = handled()
    h.set_design(h.design_history[0])
    assert h.design.singleton is None
    assert _problem(h) == "SINGLETON_ERROR"
    h.set_design(h.design_history[1])
    assert h.design.singleton == RULE
    assert ses(h) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)


def test_restoring_a_design_on_other_singletons_applies_its_rule():
    h = handled()
    f = h.wrangling.filter_records(svy.col("st") != "e")
    f.set_design(h.design)
    assert f.design.singleton == RULE and resolved(f) == {"d": "a"}


def test_switching_weights_keeps_the_rule():
    h = handled().weighting.poststratify(controls={"u": 60.0, "v": 55.0}, cells="dom")
    assert h.design.singleton == RULE
    h.update_design(wgt="w")
    assert h.design.singleton == RULE
    assert h.design.wgt_adjustment is None
    h.update_design(wgt="ps_wgt")
    assert h.design.wgt_adjustment is not None
    assert h.design.singleton == RULE
    assert ses(h) == pytest.approx(BASELINE_SES["wgt/collapse_dict/ps"], rel=1e-12)


def test_use_weight_keeps_the_rule():
    h = handled().weighting.poststratify(controls={"u": 60.0, "v": 55.0}, cells="dom")
    u = h.use_weight("w")
    assert u.design.singleton == RULE
    assert ses(u) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)


# ---------------------------------------------------------------------------
# Save/restore against other data
# ---------------------------------------------------------------------------


def test_a_saved_rule_on_other_data_applies_to_its_singletons():
    s = svy.Sample(_split_d(DATA), _saved(handled()))
    assert s.design.singleton == RULE
    assert resolved(s) == {"e": "c"} and _unused(s) == ["d"]


def test_a_saved_rule_on_its_data():
    s = svy.Sample(DATA, _saved(handled()))
    assert s.design.singleton == RULE
    assert _result(s).method == SingletonMethod.COLLAPSE


def test_design_with_a_mapping_on_other_strata_values():
    design = svy.Design(**DESIGNS["single"], singleton=Singleton("collapse", using={"x": "y"}))
    s = svy.Sample(DATA, design)
    assert _unused(s) == ["x"]
    assert _problem(s) == "SINGLETON_UNMAPPED"


# ---------------------------------------------------------------------------
# The report and the derived columns
# ---------------------------------------------------------------------------


def test_last_result_is_derived_from_the_rule():
    h = _declare(sample(), "scale")
    lr = _result(h)
    assert lr.method == SingletonMethod.SCALE
    assert lr.config.singleton_fraction == pytest.approx(2 / 5)
    assert lr.n_singletons_detected == 2
    r = h.wrangling.filter_records(svy.col("psu") != "a2")
    assert _result(r).config.singleton_fraction == pytest.approx(2 / 5)
    c = _declare(sample(), "collapse", using={"d": "a", "e": "c"})
    assert _result(c).applied == {"d": "a", "e": "c"}
    assert _result(c).n_strata_after == 3
    assert c.design.singleton == RULE


def test_derived_columns_follow_the_data():
    h = handled()
    assert set(h._data.columns) >= {_VAR_STRATUM_COL, _VAR_PSU_COL, _VAR_EXCLUDE_COL}
    r = h.wrangling.recode("st", {"a": ["a", "b"]}, replace=True)
    r.design
    assert r._data.filter(pl.col("st") == "d")[_VAR_STRATUM_COL].unique().to_list() == ["a"]
    assert set(r._data[_VAR_STRATUM_COL].unique().to_list()) == {"a", "c"}


def test_derived_columns_go_when_the_rule_is_idle():
    r = handled().wrangling.filter_records(svy.col("st").is_in(["d", "e"]), negate=True)
    r.design
    assert _VAR_STRATUM_COL not in r._data.columns
    assert _result(r) is None


def test_declaring_the_same_rule_again():
    h = _declare(sample(), "self_representing")
    again = _declare(h, "self_representing")
    assert _result(again) == _result(h)
    assert ses(again) == ses(h)


def test_a_rule_without_singletons_is_idle():
    clean = make_frame({"a": 3, "b": 2})
    s = svy.Sample(clean, svy.Design(**DESIGNS["single"]))
    for rule in RULES:
        h = rules("single")[rule](s)
        assert isinstance(h.design.singleton, Singleton)
        assert _result(h) is None
        assert ses(h) == pytest.approx(ses(s), rel=1e-12)
    assert s.design.singleton is None


# ---------------------------------------------------------------------------
# Weighting and replicate weights with a rule
# ---------------------------------------------------------------------------

WEIGHTINGS = {
    "ps": lambda h: h.weighting.poststratify(controls={"u": 60.0, "v": 55.0}, cells="dom"),
    "rake": lambda h: h.weighting.rake(
        controls={"dom": {"u": 60.0, "v": 55.0}, "reg": {"N": 70.0, "S": 45.0}}
    ),
    "trim": lambda h: h.weighting.trim(upper=2.0),
}


@pytest.mark.parametrize("wname", list(WEIGHTINGS))
@pytest.mark.parametrize("rule", ["collapse_dict", "skip", "certainty", "scale", "center", "pool"])
def test_the_rule_survives_weighting(rule, wname):
    h = rules("single")[rule](sample())
    with warnings.catch_warnings():
        # Trimming's own convergence notes are not what this test is about.
        warnings.simplefilter("ignore")
        r = WEIGHTINGS[wname](h)
    assert r.design.singleton == h.design.singleton
    assert ses(r) == pytest.approx(BASELINE_SES[f"wgt/{rule}/{wname}"], rel=1e-12)
    assert estimates(svy.Sample(r.data, _saved(r))) == estimates(r)
    assert estimates(svy.Sample(r.data, _coded(r))) == estimates(r)


REPLICATES = {
    "jk": lambda s: s.weighting.create_jk_wgts(),
    "bs": lambda s: s.weighting.create_bs_wgts(n_reps=20, rstate=7),
}


@pytest.mark.parametrize("make", list(REPLICATES))
@pytest.mark.parametrize("method", ["collapse", "pool", "self_representing"])
def test_replicates_are_built_on_the_rules_units(method, make):
    """The replicates see the strata the rule gives the Taylor variance: the
    same as replicates on the strata recoded by hand."""
    rule = RULE if method == "collapse" else Singleton(method)
    h = _fresh(sample(), rule)
    r = REPLICATES[make](h)
    ref = REPLICATES[make](_plain(REFERENCES[method](DATA)))
    assert r.design.singleton == rule
    assert r.design.rep_wgts.n_reps == ref.design.rep_wgts.n_reps
    # svy's variance units are not the data's: none recorded.
    assert (r.design.rep_wgts.stratum, r.design.rep_wgts.psu) == (None, None)
    rep = lambda s: s.estimation.mean("y", method="replication").estimates[0].se  # noqa: E731
    assert rep(r) == pytest.approx(rep(ref), rel=1e-12)
    assert estimates(svy.Sample(r.data, _saved(r))) == estimates(r)


@pytest.mark.parametrize("make", list(REPLICATES))
def test_replicates_under_skip_leave_the_singletons_out(make):
    r = REPLICATES[make](_declare(sample(), "skip"))
    assert (r.design.rep_wgts.stratum, r.design.rep_wgts.psu) == ("st", "psu")
    (w,) = [w for w in r.warnings if w.code == "SINGLETON_SKIPPED_IN_REPLICATES"]
    assert "Singleton('skip')" in w.detail


@pytest.mark.parametrize("make", list(REPLICATES))
@pytest.mark.parametrize("method", [None, "center", "scale"])
def test_replicates_without_an_analogue_raise(method, make):
    s = sample() if method is None else _declare(sample(), method)
    with pytest.raises(SingletonError) as err:
        REPLICATES[make](s)
    assert err.value.code == "SINGLETON_REPLICATES"
    assert 'svy.Singleton("collapse")' in err.value.detail


def test_replicates_on_explicit_units_ignore_the_rule():
    r = _declare(sample(), "center").weighting.create_bs_wgts(
        n_reps=20, rstate=7, stratum="st", psu="psu"
    )
    assert (r.design.rep_wgts.stratum, r.design.rep_wgts.psu) == ("st", "psu")


def test_brr_on_the_rules_units_still_refuses_odd_strata():
    from svy.errors import DimensionError

    with pytest.raises(DimensionError, match="BRR needs 2 PSUs"):
        handled().weighting.create_brr_wgts()


# ---------------------------------------------------------------------------
# Strata shapes
# ---------------------------------------------------------------------------


def _rule_round_trips(s: svy.Sample, rule) -> svy.Sample:
    h = rule(s)
    live = estimates(h)
    assert estimates(svy.Sample(h.data, _saved(h))) == live
    assert estimates(svy.Sample(h.data, _coded(h))) == live
    return h


def test_three_column_tuple_strata():
    df = DATA.with_columns(pl.lit("k").alias("k3"))
    s = svy.Sample(df, svy.Design(stratum=("reg", "code", "k3"), psu="psu", wgt="w"))
    h = _rule_round_trips(s, lambda s: _declare(s, "collapse", using="smallest"))
    assert resolved(h) == {("N", 4, "k"): ("N", 1, "k"), ("S", 5, "k"): ("N", 2, "k")}
    assert ses(h) == pytest.approx(BASELINE_SES["single/collapse_smallest"], rel=1e-12)


def test_integer_strata():
    s = svy.Sample(DATA, svy.Design(stratum="code", psu="psu", wgt="w"))
    h = _rule_round_trips(s, lambda s: _declare(s, "collapse", using={4: 1, 5: 3}))
    assert resolved(h) == {4: 1, 5: 3}
    assert ses(h) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)


def test_bool_and_int_tuple_strata():
    df = DATA.with_columns((pl.col("code") % 2 == 0).alias("even"))
    s = svy.Sample(df, svy.Design(stratum=("even", "code"), psu="psu", wgt="w"))
    h = _rule_round_trips(s, lambda s: _declare(s, "skip"))
    # In the order of svy's stratum keys ("false__by__5" < "true__by__4").
    assert resolved(h) == [(False, 5), (True, 4)]
    assert ses(h) == pytest.approx(BASELINE_SES["single/skip"], rel=1e-12)


def test_categorical_strata():
    s = svy.Sample(
        DATA.with_columns(pl.col("st").cast(pl.Categorical)), svy.Design(**DESIGNS["single"])
    )
    h = _rule_round_trips(s, lambda s: _declare(s, "pool"))
    assert resolved(h) == ["d", "e"]
    assert ses(h) == pytest.approx(BASELINE_SES["single/pool"], rel=1e-12)


def _null_round_trip(s: svy.Sample, rule) -> svy.Sample:
    """Null strata need drop_nulls=True at estimation (as before this change)."""
    h = rule(s)

    def ests(x: svy.Sample) -> list:
        return [
            (p.est, p.se)
            for e in (
                x.estimation.mean("y", drop_nulls=True),
                x.estimation.total("y", drop_nulls=True),
            )
            for p in e.estimates
        ]

    live = ests(h)
    for design in (_saved(h), _coded(h)):
        r = svy.Sample(h.data, design)
        assert ests(r) == live
        assert _result(r).applied == _result(h).applied
        assert r._data[_VAR_STRATUM_COL].to_list() == h._data[_VAR_STRATUM_COL].to_list()
    return h


def test_strata_with_a_null():
    df = DATA.with_columns(
        pl.when(pl.col("st") == "e").then(None).otherwise(pl.col("st")).alias("st")
    )
    s = svy.Sample(df, svy.Design(**DESIGNS["single"]))
    h = _null_round_trip(s, lambda s: _declare(s, "self_representing"))
    assert resolved(h) == [None, "d"]
    assert _result(h).applied == ("__Null__", "d")


def test_tuple_strata_with_a_null():
    df = DATA.with_columns(
        pl.when(pl.col("st") == "e").then(None).otherwise(pl.col("code")).alias("code")
    )
    s = svy.Sample(df, svy.Design(**DESIGNS["tuple"]))
    h = _null_round_trip(
        s, lambda s: _declare(s, "collapse", using={("S", None): ("S", 3), ("N", 4): ("N", 1)})
    )
    assert resolved(h) == {("N", 4): ("N", 1), ("S", None): ("S", 3)}


def test_date_strata():
    days = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    df = DATA.with_columns(
        pl.col("st").replace_strict({k: dt.date(2020, 1, v) for k, v in days.items()}).alias("day")
    )
    s = svy.Sample(df, svy.Design(stratum="day", psu="psu", wgt="w"))
    mapping = {dt.date(2020, 1, 4): dt.date(2020, 1, 1), dt.date(2020, 1, 5): dt.date(2020, 1, 3)}
    h = _rule_round_trips(s, lambda s: _declare(s, "collapse", using=mapping))
    assert resolved(h) == mapping
    assert ses(h) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)


def test_integer_strata_cast_to_text_keep_the_rule():
    s = svy.Sample(DATA, svy.Design(**DESIGNS["tuple"]))
    h = _declare(s, "skip")
    r = h.wrangling.cast("code", pl.Utf8)
    assert r.design.singleton == Singleton("skip")
    assert resolved(r) == [("N", "4"), ("S", "5")]
    assert ses(r) == pytest.approx(BASELINE_SES["single/skip"], rel=1e-12)


def test_psu_only_design_takes_no_rule():
    s = svy.Sample(DATA, svy.Design(psu="psu", wgt="w"))
    with pytest.raises(ValueError, match="no stratum"):
        s.update_design(singleton="skip")
    assert s.design.singleton is None


def test_strata_without_psus():
    df = make_frame({"a": 3, "b": 3}).vstack(make_frame({"d": 1}).with_columns(pl.col("id") + 100))
    df = df.filter(~((pl.col("st") == "d") & (pl.col("id") != 101)))
    s = svy.Sample(df, svy.Design(stratum="st", wgt="w"))
    h = _rule_round_trips(s, lambda s: _declare(s, "collapse", using={"d": "a"}))
    ref = svy.Sample(
        df.with_columns(pl.col("st").replace({"d": "a"})), svy.Design(stratum="st", wgt="w")
    )
    assert ses(h) == pytest.approx(ses(ref), rel=1e-12)


def test_strata_as_a_domain_variable():
    df = DATA.with_columns(pl.col("st").alias("st_by"))
    h = _declare(
        svy.Sample(df, svy.Design(**DESIGNS["single"])), "collapse", using={"d": "a", "e": "c"}
    )
    ref = _plain(df.with_columns(pl.col("st").replace({"d": "a", "e": "c"})))
    flat = lambda x: [v for e in estimates(x, by="st_by") for r in e for v in r[2:]]  # noqa: E731
    assert flat(h) == pytest.approx(flat(ref), rel=1e-12)
    m = h.estimation.mean("y", where=svy.col("st_by") == "a")
    assert m.estimates[0].se == pytest.approx(
        ref.estimation.mean("y", where=svy.col("st_by") == "a").estimates[0].se, rel=1e-12
    )


def test_the_strata_column_as_the_ssu():
    """A design that names the stratum column in another role."""
    s = svy.Sample(DATA, svy.Design(stratum="st", psu="psu", ssu="st", wgt="w"))
    h = _rule_round_trips(s, lambda s: _declare(s, "collapse", using={"d": "a", "e": "c"}))
    assert h.design.singleton == RULE


# ---------------------------------------------------------------------------
# Singleton patterns
# ---------------------------------------------------------------------------


def test_every_stratum_but_one_a_singleton():
    df = make_frame({"a": 3, "b": 1, "c": 1, "d": 1})
    s = svy.Sample(df, svy.Design(**DESIGNS["single"]))
    h = _rule_round_trips(s, lambda s: _declare(s, "collapse"))
    assert resolved(h) == {"b": "a", "c": "a", "d": "a"}
    ref = _plain(df.with_columns(pl.lit("a").alias("st")))
    assert ses(h) == pytest.approx(ses(ref), rel=1e-12)


def test_all_strata_singletons_collapse_has_no_target():
    s = svy.Sample(make_frame({"d": 1, "e": 1}), svy.Design(**DESIGNS["single"]))
    h = _declare(s, "collapse")
    assert _problem(h) == "NO_MERGE_TARGETS"
    h = _rule_round_trips(s, lambda s: _declare(s, "pool"))
    assert resolved(h) == ["d", "e"]


def test_a_singleton_whose_psu_has_zero_weight():
    df = DATA.with_columns(
        pl.when(pl.col("st") == "d").then(0.0).otherwise(pl.col("w")).alias("w")
    )
    s = svy.Sample(df, svy.Design(**DESIGNS["single"]))
    assert _keys(s) == ["d", "e"]
    h = _rule_round_trips(s, lambda s: _declare(s, "collapse", using={"d": "a", "e": "c"}))
    ref = _plain(df.with_columns(pl.col("st").replace({"d": "a", "e": "c"})))
    assert ses(h) == pytest.approx(ses(ref), rel=1e-12)


def test_a_domain_singleton_is_not_a_design_singleton():
    s = svy.Sample(make_frame({"a": 3, "b": 3}), svy.Design(**DESIGNS["single"]))
    assert not (s.n_singletons > 0)
    m = s.estimation.mean("y", where=svy.col("psu").is_in(["a0", "b0", "b1"]))
    assert m.estimates[0].se > 0
    assert s.design.singleton is None


def test_several_singletons():
    df = make_frame({"a": 3, "b": 3, "c": 3, "d": 1, "e": 1, "f": 1})
    h = _declare(svy.Sample(df, svy.Design(**DESIGNS["single"])), "skip")
    assert resolved(h) == ["d", "e", "f"]
    r = h.wrangling.filter_records(svy.col("psu").is_in(["a0", "a1"]), negate=True)
    assert resolved(r) == ["a", "d", "e", "f"]


# ---------------------------------------------------------------------------
# Strata named by their own values (or svy's key strings)
# ---------------------------------------------------------------------------


def _int_sample() -> svy.Sample:
    return svy.Sample(DATA, svy.Design(stratum="code", psu="psu", wgt="w"))


def test_collapse_takes_native_int_strata():
    h = _declare(_int_sample(), "collapse", using={4: 1, 5: 3})
    assert resolved(h) == {4: 1, 5: 3}
    assert ses(h) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)


def test_collapse_still_takes_key_strings():
    old = _declare(_int_sample(), "collapse", using={"4": "1", "5": "3"})
    new = _declare(_int_sample(), "collapse", using={4: 1, 5: 3})
    mixed = _declare(_int_sample(), "collapse", using={4: "1", "5": 3})
    assert resolved(old) == resolved(new) == resolved(mixed) == {4: 1, 5: 3}
    assert ses(old) == ses(new) == ses(mixed)


def test_collapse_takes_native_bool_and_tuple_strata():
    df = DATA.with_columns((pl.col("code") % 2 == 0).alias("even"))
    s = svy.Sample(df, svy.Design(stratum=("even", "code"), psu="psu", wgt="w"))
    h = _declare(s, "collapse", using={(True, 4): (False, 1), (False, 5): (False, 3)})
    assert resolved(h) == {(False, 5): (False, 3), (True, 4): (False, 1)}
    assert ses(h) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)
    # Lists for tuples, and the key strings, still work.
    k = _declare(s, "collapse", using={"true__by__4": "false__by__1", "false__by__5": [False, 3]})
    assert resolved(k) == resolved(h)


def test_collapse_takes_native_date_strata():
    days = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    df = DATA.with_columns(
        pl.col("st").replace_strict({k: dt.date(2020, 1, v) for k, v in days.items()}).alias("day")
    )
    s = svy.Sample(df, svy.Design(stratum="day", psu="psu", wgt="w"))
    h = _declare(s, "collapse", using={"2020-01-04": "2020-01-01", "2020-01-05": "2020-01-03"})
    assert ses(h) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)


def test_collapse_callable_may_return_a_value_or_a_candidate():
    s = _int_sample()
    by_value = _declare(s, "collapse", using=lambda single, cands: 1)
    by_info = _declare(
        s, "collapse", using=lambda single, cands: next(c for c in cands if c.stratum_key == "1")
    )
    assert resolved(by_value) == resolved(by_info) == {4: 1, 5: 1}


def test_an_incomplete_or_wrong_mapping_raises_at_estimation():
    h = _declare(_int_sample(), "collapse", using={4: 1})
    with pytest.raises(SingletonError, match="not in the collapse mapping: 5"):
        h.estimation.mean("y")
    h = _declare(_int_sample(), "collapse", using={4: 99, 5: 1})
    with pytest.raises(SingletonError, match="no longer has: 99"):
        h.estimation.mean("y")
    # 5 is a singleton itself: refused when the rule is built.
    with pytest.raises(svy.MethodError, match="must not be singleton strata"):
        Singleton("collapse", using={4: 5, 5: 1})
    # A target that became a singleton in the data: raised at estimation.
    h = _declare(_int_sample(), "collapse", using={4: 1, 5: 3})
    r = h.wrangling.filter_records(svy.col("psu").is_in(["c1", "c2"]), negate=True)
    assert _problem(r) == "SINGLETON_TARGET_SINGLETON"


def test_an_ambiguous_string_is_an_error():
    from svy.core.singleton import _StrataIndex

    # Stratum 1 has key "k1" and stratum 2 has key "1": the string "1" names
    # both (one by its value's str form, one by key), so it resolves to neither.
    frame = pl.DataFrame({"s": [1, 2], "key": ["k1", "1"]})
    index = _StrataIndex(frame, ["s"], "key")
    assert index.key(1) == "k1"
    assert index.key(2) == "1"
    with pytest.raises(ValueError, match="matches several strata"):
        index.key("1")
    assert index.key("1", strict=False) is None


# ---------------------------------------------------------------------------
# Re-derivation only when the strata can have moved
# ---------------------------------------------------------------------------


@pytest.fixture
def rederive_calls(monkeypatch):
    import svy.core.singleton as mod

    calls = []
    real = mod._rederive

    def counting(sample):
        calls.append(sample)
        return real(sample)

    monkeypatch.setattr(mod, "_rederive", counting)
    return calls


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.wrangling.mutate({"z": pl.col("y") * 2}),
        lambda s: s.wrangling.recode("dom", {"U": ["u"]}, replace=True),
        lambda s: s.wrangling.join(pl.DataFrame({"id": DATA["id"], "k": DATA["y"]}), on="id"),
        lambda s: s.weighting.poststratify(controls={"u": 60.0, "v": 55.0}, cells="dom"),
        lambda s: s.wrangling.filter_records(svy.col("y") > -1000.0),
    ],
)
def test_changes_elsewhere_skip_the_rederivation(change, rederive_calls):
    h = handled()
    h.estimation.mean("y")
    rederive_calls.clear()
    r = change(h)
    r.estimation.mean("y")
    assert rederive_calls == []
    assert r.design.singleton == RULE


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.wrangling.recode("st", {"a": ["a", "b"]}, replace=True),
        lambda s: s.wrangling.filter_records(svy.col("id") != 1),
        lambda s: s.wrangling.order_by("y"),
        lambda s: s.wrangling.mutate({"psu": pl.col("psu") + "x"}),
        lambda s: s.wrangling.cast("st", pl.Categorical),
    ],
)
def test_changes_to_rows_or_strata_rederive(change, rederive_calls):
    h = handled()
    h.estimation.mean("y")
    rederive_calls.clear()
    r = change(h)
    r.estimation.mean("y")
    assert len(rederive_calls) == 1


def test_a_change_to_a_within_column_rederives(rederive_calls):
    h = _declare(sample(), "collapse", within="reg")
    h.estimation.mean("y")
    rederive_calls.clear()
    r = h.wrangling.mutate({"reg": pl.lit("N")})
    assert resolved(r) == {"d": "a", "e": "b"}
    assert len(rederive_calls) == 1


def test_a_skipped_rederivation_keeps_the_derived_columns():
    h = handled()
    r = h.wrangling.mutate({"z": pl.col("y") * 2})
    assert ses(r) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)
    assert r._data[_VAR_STRATUM_COL].to_list() == h._data[_VAR_STRATUM_COL].to_list()


# ---------------------------------------------------------------------------
# Singletons re-detected on samples without a rule
# ---------------------------------------------------------------------------

NO_SINGLETONS = make_frame({"a": 3, "b": 2, "c": 2})


def test_a_filter_creating_a_singleton_is_seen_as_on_a_fresh_sample():
    s = svy.Sample(NO_SINGLETONS, svy.Design(**DESIGNS["single"]))
    s.estimation.mean("y")
    f = s.wrangling.filter_records(svy.col("psu") != "b1")
    fresh = svy.Sample(f.data, svy.Design(**DESIGNS["single"]))
    with pytest.raises(SingletonError):
        fresh.estimation.mean("y")
    with pytest.raises(SingletonError) as exc:
        f.estimation.mean("y")
    assert "st=b" in str(exc.value)
    assert _keys(f) == ["b"]


def test_a_filter_removing_the_singletons_is_seen_too():
    s = svy.Sample(DATA, svy.Design(**DESIGNS["single"]))
    with pytest.raises(SingletonError):
        s.estimation.mean("y")
    f = s.wrangling.filter_records(svy.col("st").is_in(["d", "e"]), negate=True)
    fresh = svy.Sample(f.data, svy.Design(**DESIGNS["single"]))
    assert se(f) == pytest.approx(se(fresh), rel=1e-12)


@pytest.mark.parametrize("inplace", [False, True])
def test_a_recode_creating_a_singleton_is_seen(inplace):
    s = svy.Sample(NO_SINGLETONS, svy.Design(**DESIGNS["single"]))
    r = s.wrangling.recode("psu", {"c0": ["c0", "c1"]}, replace=True, inplace=inplace)
    with pytest.raises(SingletonError):
        r.estimation.mean("y")


def test_filter_records_reports_singletons_a_rule_handles_at_info():
    s = _declare(svy.Sample(NO_SINGLETONS, svy.Design(**DESIGNS["single"])), "center")
    f = s.wrangling.filter_records(
        svy.col("psu") != "b1", check_singletons=True, on_singletons="error"
    )
    (w,) = [w for w in f.warnings if w.code == "SINGLETONS_DETECTED"]
    assert "handled by the design's rule Singleton('center')" in w.detail
    assert w.level.name == "INFO"
    plain = svy.Sample(NO_SINGLETONS, svy.Design(**DESIGNS["single"]))
    with pytest.raises(svy.MethodError, match="Singletons detected"):
        plain.wrangling.filter_records(
            svy.col("psu") != "b1", check_singletons=True, on_singletons="error"
        )


# ---------------------------------------------------------------------------
# Every rule x every trigger
# ---------------------------------------------------------------------------

METHODS = ["self_representing", "skip", "scale", "center", "pool", "collapse"]


def _rule(method: str) -> Singleton:
    return RULE if method == "collapse" else Singleton(method)


def _apply(method: str, s: svy.Sample) -> svy.Sample:
    return s._fork().update_design(singleton=_rule(method))


TRIGGERS = {
    "join": lambda s: s.wrangling.join(pl.DataFrame({"id": DATA["id"], "k": DATA["y"]}), on="id"),
    "rename_strata": lambda s: s.wrangling.rename_columns({"st": "st9"}),
    "recode_other": lambda s: s.wrangling.recode("dom", {"U": ["u"]}, replace=True),
    "filter_rows": lambda s: s.wrangling.filter_records(svy.col("id") != 2),
    "update_design_wgt": lambda s: s.update_design(wgt="y"),
    "set_data_same": lambda s: s.set_data(s.data),
    "update_data_same": lambda s: s.update_data(s.data),
    "filter_handled_away": lambda s: s.wrangling.filter_records(svy.col("st") != "e"),
    "filter_new_singleton": lambda s: s.wrangling.filter_records(
        svy.col("psu") != "b0"
    ).wrangling.filter_records(svy.col("psu") != "b1"),
    "mutate_split_psu": lambda s: s.wrangling.mutate(
        {
            "psu": pl.when((pl.col("st") == "d") & (pl.col("id") % 2 == 0))
            .then(pl.lit("d1"))
            .otherwise(pl.col("psu"))
        }
    ),
    "set_data_split": lambda s: s.set_data(_split_d(s.data)),
    "update_data_split": lambda s: s.update_data(_split_d(s.data)),
    "update_design_psu": lambda s: s.update_design(psu="hh"),
    "update_design_stratum_tuple": lambda s: s.update_design(stratum=("st", "reg")),
}


@pytest.mark.parametrize("trigger", list(TRIGGERS))
@pytest.mark.parametrize("method", METHODS)
def test_rule_by_trigger(method, trigger):
    """Whatever the change, the rule stays, silently, and the sample gives what
    the rule gives on a fresh sample of the changed data and design."""
    h = _apply(method, sample())
    r = TRIGGERS[trigger](h)
    assert r.design.singleton == _rule(method)
    fresh = _fresh(r, _rule(method))
    try:
        expected = ses(fresh)
    except SingletonError as err:
        assert _problem(r) == err.code
        return
    assert ses(r) == pytest.approx(expected, rel=1e-12, nan_ok=True)


# ---------------------------------------------------------------------------
# More edge cases
# ---------------------------------------------------------------------------


def test_a_where_domain_singleton_does_not_touch_the_rule():
    h = handled()
    m = h.estimation.mean("y", where=svy.col("psu").is_in(["a0", "b0", "b1", "d0"]))
    ref = _plain(DATA.with_columns(pl.col("st").replace({"d": "a", "e": "c"})))
    r = ref.estimation.mean("y", where=svy.col("psu").is_in(["a0", "b0", "b1", "d0"]))
    assert m.estimates[0].se == pytest.approx(r.estimates[0].se, rel=1e-12)
    assert h.design.singleton == RULE


def test_the_strata_column_inside_the_psu():
    s = svy.Sample(DATA, svy.Design(stratum="st", psu=("st", "psu"), wgt="w"))
    h = _rule_round_trips(s, lambda s: _declare(s, "collapse", using={"d": "a", "e": "c"}))
    assert h.design.singleton == RULE
    assert ses(h) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)


@pytest.mark.parametrize("wname", list(WEIGHTINGS))
@pytest.mark.parametrize("method", METHODS)
def test_weighting_with_a_rule_equals_a_fresh_sample_with_it(method, wname):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = WEIGHTINGS[wname](_apply(method, sample()))
    assert ses(r) == pytest.approx(ses(_fresh(r, _rule(method))), rel=1e-12)


def test_zero_weight_singleton_psu_with_every_rule():
    df = DATA.with_columns(
        pl.when(pl.col("st") == "d").then(0.0).otherwise(pl.col("w")).alias("w")
    )
    for method in METHODS:
        h = _apply(method, svy.Sample(df, svy.Design(**DESIGNS["single"])))
        assert h.design.singleton is not None
        assert estimates(svy.Sample(h.data, _saved(h))) == estimates(h)


def test_all_strata_singletons_with_every_rule():
    s = svy.Sample(make_frame({"d": 1, "e": 1}), svy.Design(**DESIGNS["single"]))
    for method in ["self_representing", "skip", "scale", "center", "pool"]:
        h = _apply(method, s)
        assert resolved(h) == ["d", "e"]
        # scale has no reference stratum left: NaN SEs, as before.
        restored = ses(svy.Sample(h.data, _saved(h)))
        assert restored == pytest.approx(ses(h), rel=0, nan_ok=True)
