# tests/svy/core/test_singleton_design_part.py
"""Singleton handling as a design part.

``sample.singleton.*`` resolves a rule into a ``SingletonSpec`` on the design;
the variance columns are derived from design + data whenever either changed.
The spec is kept while the singleton strata detected in the data are exactly
the ones it handles (and a collapse target still exists), and cleared with one
warning otherwise.

``BASELINE_SES`` were computed on this branch before the change (commit
3c38a29, the tip of feat/design-serialization) with the same data and calls,
and pin today's numbers. Where an independent reference exists (collapse,
pool and certainty are strata recodes) it is checked too.
"""

from __future__ import annotations

import datetime as dt
import gc
import warnings

import polars as pl
import pytest

import svy

from svy.core.design import SingletonSpec
from svy.core.enumerations import SingletonHandling
from svy.core.singleton import _VAR_EXCLUDE_COL, _VAR_PSU_COL, _VAR_STRATUM_COL
from svy.errors.singleton_errors import SingletonError
from svy.serialize import from_json, to_design, to_json


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
        "certainty": lambda s: s.singleton.certainty(),
        "skip": lambda s: s.singleton.skip(),
        "scale": lambda s: s.singleton.scale(),
        "center": lambda s: s.singleton.center(),
        "pool": lambda s: s.singleton.pool(),
        "collapse_dict": lambda s: s.singleton.collapse(using={k("d"): k("a"), k("e"): k("c")}),
        "collapse_smallest": lambda s: s.singleton.collapse(),
        "collapse_callable": lambda s: s.singleton.collapse(using=_last),
        "collapse_rstate": lambda s: s.singleton.collapse(rstate=3),
        "collapse_largest_rstate": lambda s: s.singleton.collapse(using="largest", rstate=11),
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


# ---------------------------------------------------------------------------
# Every method: today's numbers, save/restore, code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(DESIGNS))
@pytest.mark.parametrize("rule", RULES)
def test_estimates_are_todays(kind, rule):
    h = rules(kind)[rule](sample(kind))
    assert ses(h) == pytest.approx(_pin(kind, rule), rel=1e-12)


@pytest.mark.parametrize("kind", list(DESIGNS))
@pytest.mark.parametrize("rule", RULES)
def test_estimates_survive_save_restore_and_code(kind, rule):
    h = rules(kind)[rule](sample(kind))
    live = estimates(h)
    for design in (_saved(h), _coded(h)):
        assert design == h.design
        assert design.singleton == h.design.singleton
        # With the sample's frame (derived columns included) and the original.
        assert estimates(svy.Sample(h.data, design)) == live
        assert estimates(svy.Sample(DATA, design)) == live


@pytest.mark.parametrize("kind", list(DESIGNS))
@pytest.mark.parametrize("rule", RULES)
def test_the_rule_is_on_the_design_and_in_the_history(kind, rule):
    s = sample(kind)
    h = rules(kind)[rule](s)
    spec = h.design.singleton
    assert isinstance(spec, SingletonSpec)
    assert h.design_history[-2].singleton is None
    assert h.design_history[-1].singleton == spec
    assert s.design.singleton is None
    assert len(s.design_history) == 1
    # Strata stored as the columns' own values, never svy's key strings.
    for v in (*spec.strata, *(x for pair in spec.mapping for x in pair)):
        if kind == "tuple":
            assert isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], int)
        else:
            assert isinstance(v, str) and "__by__" not in v


def test_resolved_decisions_are_stored_not_replayed():
    specs = {rule: rules("single")[rule](sample()).design.singleton for rule in RULES}
    assert specs["certainty"] == SingletonSpec.certainty(["d", "e"])
    assert specs["skip"] == SingletonSpec.skip(["d", "e"])
    assert specs["scale"] == SingletonSpec.scale(["d", "e"])
    assert specs["center"] == SingletonSpec.center(["d", "e"])
    assert specs["pool"] == SingletonSpec.pool(["d", "e"])
    assert specs["collapse_dict"] == SingletonSpec.collapse({"d": "a", "e": "c"})
    assert specs["collapse_smallest"] == SingletonSpec.collapse({"d": "a", "e": "b"})
    assert specs["collapse_callable"] == SingletonSpec.collapse({"d": "c", "e": "c"})
    assert specs["collapse_rstate"] == SingletonSpec.collapse({"d": "c", "e": "b"})
    assert specs["collapse_largest_rstate"] == SingletonSpec.collapse({"d": "a", "e": "a"})
    tup = rules("tuple")["collapse_rstate"](sample("tuple")).design.singleton
    assert tup == SingletonSpec.collapse({("N", 4): ("S", 3), ("S", 5): ("N", 2)})


def test_a_callable_is_called_once_and_its_answer_stored():
    calls = []

    def pick(singleton, candidates):
        calls.append(singleton.stratum_key)
        return sorted(c.stratum_key for c in candidates)[0]

    h = sample().singleton.collapse(using=pick)
    assert calls == ["d", "e"]
    h.wrangling.filter_records(svy.col("id") != 1).estimation.mean("y")
    svy.Sample(DATA, _saved(h)).estimation.mean("y")
    assert calls == ["d", "e"]


# ---------------------------------------------------------------------------
# Independent references: collapse, pool and certainty are strata recodes
# ---------------------------------------------------------------------------

_LONE = pl.col("st").is_in(["d", "e"])


def _plain(data: pl.DataFrame, **kw) -> svy.Sample:
    return svy.Sample(data, svy.Design(**{**DESIGNS["single"], **kw}))


def test_collapse_is_the_recoded_strata():
    ref = _plain(DATA.with_columns(pl.col("st").replace({"d": "a", "e": "c"})))
    h = sample().singleton.collapse(using={"d": "a", "e": "c"})
    assert ses(h) == pytest.approx(ses(ref), rel=1e-12)


def test_pool_is_the_recoded_strata():
    ref = _plain(DATA.with_columns(pl.col("st").replace({"d": "__pooled__", "e": "__pooled__"})))
    assert ses(sample().singleton.pool()) == pytest.approx(ses(ref), rel=1e-12)


def test_certainty_makes_rows_the_psus():
    ref = _plain(
        DATA.with_columns(
            pl.when(_LONE).then(pl.col("psu")).otherwise(pl.col("st")).alias("st"),
            pl.when(_LONE).then(pl.col("id").cast(pl.Utf8)).otherwise(pl.col("psu")).alias("psu"),
        )
    )
    assert ses(sample().singleton.certainty()) == pytest.approx(ses(ref), rel=1e-12)


def test_certainty_makes_ssus_the_psus():
    ref = _plain(
        DATA.with_columns(
            pl.when(_LONE).then(pl.col("psu")).otherwise(pl.col("st")).alias("st"),
            pl.when(_LONE).then(pl.col("hh")).otherwise(pl.col("psu")).alias("psu"),
        )
    )
    assert ses(sample("ssu").singleton.certainty()) == pytest.approx(ses(ref), rel=1e-12)


# ---------------------------------------------------------------------------
# The stale-cache bug
# ---------------------------------------------------------------------------

STALE = make_frame({"a": 3, "b": 3, "c": 3, "d": 1})


def test_collapse_then_recode_strata_gives_the_fresh_sample():
    s = svy.Sample(STALE, svy.Design(stratum="st", psu="psu", wgt="w"))
    h = s.singleton.collapse(using={"d": "a"})
    before = se(h)
    r = h.wrangling.recode("st", {"c": ["b", "c"]}, replace=True)
    fresh = svy.Sample(
        STALE.with_columns(pl.col("st").replace({"d": "a", "b": "c"})),
        svy.Design(stratum="st", psu="psu", wgt="w"),
    )
    assert se(r) == pytest.approx(se(fresh), rel=1e-12)
    assert se(r) != pytest.approx(before)
    # d is still the only singleton and a exists: kept, silently.
    assert r.design.singleton == SingletonSpec.collapse({"d": "a"})
    assert ses(r) == pytest.approx(ses(fresh), rel=1e-12)


def test_collapse_then_recode_inplace():
    s = svy.Sample(STALE, svy.Design(stratum="st", psu="psu", wgt="w"))
    h = s.singleton.collapse(using={"d": "a"})
    h.wrangling.recode("st", {"c": ["b", "c"]}, replace=True, inplace=True)
    fresh = svy.Sample(
        STALE.with_columns(pl.col("st").replace({"d": "a", "b": "c"})),
        svy.Design(stratum="st", psu="psu", wgt="w"),
    )
    assert se(h) == pytest.approx(se(fresh), rel=1e-12)


# ---------------------------------------------------------------------------
# Triggers: kept or cleared, lazily at the next use or at the call
# ---------------------------------------------------------------------------

SPEC = SingletonSpec.collapse({"d": "a", "e": "c"})


def handled() -> svy.Sample:
    return sample().singleton.collapse(using={"d": "a", "e": "c"})


def _split_d(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when((pl.col("st") == "d") & (pl.col("id") % 2 == 0))
        .then(pl.lit("d1"))
        .otherwise(pl.col("psu"))
        .alias("psu")
    )


KEPT = {
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
@pytest.mark.parametrize("trigger", list(KEPT))
def test_changes_that_keep_the_singletons_keep_the_rule_silently(trigger, inplace):
    h = handled()
    r = KEPT[trigger](h, inplace)
    assert (r is h) is inplace
    assert r.design.singleton == SPEC
    # Re-applied to the current data: the same as the rule on a fresh sample.
    fresh = svy.Sample(r.data, svy.Design(**DESIGNS["single"])).singleton.collapse(
        using={"d": "a", "e": "c"}
    )
    assert ses(r) == pytest.approx(ses(fresh), rel=1e-12)
    assert r.singleton.last_result.config.stratum_mapping == {"d": "a", "e": "c"}


CLEARED = {
    "split_the_singleton_psu": (
        lambda s, ip: s.wrangling.mutate(
            {
                "psu": pl.when((pl.col("st") == "d") & (pl.col("id") % 2 == 0))
                .then(pl.lit("d1"))
                .otherwise(pl.col("psu"))
            },
            inplace=ip,
        ),
        "the singleton strata changed",
        "'e'",
    ),
    "recode_away_the_target": (
        lambda s, ip: s.wrangling.recode("st", {"b": ["b", "c"]}, replace=True, inplace=ip),
        "collapse targets no longer in the data: 'c'",
        "'d', 'e'",
    ),
    "recode_psu_values": (
        lambda s, ip: s.wrangling.recode(
            "psu", {"b0": ["b0", "b1", "b2"]}, replace=True, inplace=ip
        ),
        "the singleton strata changed",
        "'b', 'd', 'e'",
    ),
    "filter_away_a_handled_singleton": (
        lambda s, ip: s.wrangling.filter_records(svy.col("st") != "e", inplace=ip),
        "handled strata no longer in the data: 'e'",
        "'d'",
    ),
    "filter_creates_a_singleton": (
        lambda s, ip: s.wrangling.filter_records(
            svy.col("psu").is_in(["b0", "b1"]), negate=True, inplace=ip
        ),
        "the singleton strata changed",
        "'b', 'd', 'e'",
    ),
    "filter_all_singletons_away": (
        lambda s, ip: s.wrangling.filter_records(
            svy.col("st").is_in(["d", "e"]), negate=True, inplace=ip
        ),
        "handled strata no longer in the data: 'd', 'e'",
        "none",
    ),
    "mutate_merges_the_singletons": (
        lambda s, ip: s.wrangling.mutate(
            {"st": pl.when(pl.col("st") == "e").then(pl.lit("d")).otherwise(pl.col("st"))},
            inplace=ip,
        ),
        "handled strata no longer in the data: 'e'",
        "none",
    ),
}


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("trigger", list(CLEARED))
def test_changes_to_the_singletons_clear_the_rule_at_the_next_use(trigger, inplace):
    change, reason, now = CLEARED[trigger]
    h = handled()
    handled_design = h._design
    r = change(h, inplace)
    with pytest.warns(UserWarning) as rec:
        design = r.design
    assert len(rec) == 1
    msg = str(rec[0].message)
    assert msg.startswith(
        "[SINGLETON_RULE_CLEARED] Singleton handling cleared: "
        "singleton handling (collapse) cleared: "
    )
    assert reason in msg
    assert f"singletons now: {now}." in msg
    assert rec[0].filename == __file__
    assert design.singleton is None
    # Cleared once: nothing more on later uses.
    r.design
    r.singleton.last_result
    assert r.singleton.last_result is None
    # The handled design the clearing replaced stays in the history.
    assert [d.singleton for d in r.design_history] == [None, SPEC, None]
    assert r.design_history[-2] == handled_design
    # Restoring it brings the rule back only if it matches the data again.
    with pytest.warns(UserWarning, match="cleared"):
        r.set_design(r.design_history[-2])


@pytest.mark.parametrize("inplace", [False, True])
def test_a_cast_that_merges_strata_clears_the_rule(inplace):
    df = DATA.with_columns(
        pl.col("st").replace_strict({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 4.4}).alias("g")
    )
    h = svy.Sample(df, svy.Design(stratum="g", psu="psu", wgt="w")).singleton.skip()
    assert h.design.singleton == SingletonSpec.skip([4.0, 4.4])
    r = h.wrangling.cast("g", pl.Int64, strict=False, inplace=inplace)
    # 4.0 is still there as the integer 4; 4.4 is gone.
    with pytest.warns(
        UserWarning, match=r"no longer in the data: 4\.4; singletons now: none\."
    ) as rec:
        assert r.design.singleton is None
    assert len(rec) == 1


@pytest.mark.parametrize("inplace", [False, True])
def test_a_fill_null_that_merges_strata_clears_the_rule(inplace):
    df = DATA.with_columns(
        pl.when(pl.col("st") == "e").then(None).otherwise(pl.col("st")).alias("st")
    )
    h = svy.Sample(df, svy.Design(**DESIGNS["single"])).singleton.skip()
    assert h.design.singleton == SingletonSpec.skip([None, "d"])
    r = h.wrangling.fill_null("st", "d", inplace=inplace)
    with pytest.warns(
        UserWarning, match=r"no longer in the data: None; singletons now: none\."
    ) as rec:
        assert r.design.singleton is None
    assert len(rec) == 1


@pytest.mark.parametrize("inplace", [False, True])
def test_a_fill_null_elsewhere_keeps_the_rule(inplace):
    df = DATA.with_columns(pl.when(pl.col("id") == 1).then(None).otherwise(pl.col("x")).alias("x"))
    h = svy.Sample(df, svy.Design(**DESIGNS["single"])).singleton.skip()
    r = h.wrangling.fill_null("x", 0, inplace=inplace)
    assert r.design.singleton == SingletonSpec.skip(["d", "e"])


@pytest.mark.parametrize("trigger", list(CLEARED))
def test_after_clearing_estimation_is_as_unhandled(trigger):
    change, _, now = CLEARED[trigger]
    r = change(handled(), False)
    with pytest.warns(UserWarning, match="cleared"):
        r.design
    if now == "none":
        r.estimation.mean("y")
    else:
        with pytest.raises(SingletonError):
            r.estimation.mean("y")


def test_the_warning_fires_at_the_estimation_line():
    r = handled().wrangling.filter_records(svy.col("st") != "e")
    with pytest.warns(UserWarning, match="cleared") as rec:
        with pytest.raises(SingletonError):
            r.estimation.mean("y")
    assert len(rec) == 1 and rec[0].filename == __file__


def test_the_warning_fires_at_sample_singleton():
    r = handled().wrangling.filter_records(svy.col("st") != "e")
    with pytest.warns(UserWarning, match="cleared") as rec:
        r.singleton.last_result
    assert len(rec) == 1 and rec[0].filename == __file__


def test_cleared_then_reapplied():
    r = handled().wrangling.filter_records(svy.col("st") != "e")
    with pytest.warns(UserWarning, match="cleared"):
        r.design
    again = r.singleton.collapse(using={"d": "a"})
    assert again.design.singleton == SingletonSpec.collapse({"d": "a"})
    ref = svy.Sample(
        r.data.with_columns(pl.col("st").replace({"d": "a"})), svy.Design(**DESIGNS["single"])
    )
    assert ses(again) == pytest.approx(ses(ref), rel=1e-12)


@pytest.mark.parametrize("method", ["certainty", "skip", "scale", "center", "pool"])
def test_every_rule_is_cleared_by_a_new_singleton(method):
    h = getattr(sample().singleton, method)()
    r = h.wrangling.filter_records(svy.col("psu").is_in(["b0", "b1"]), negate=True)
    with pytest.warns(UserWarning, match=rf"\({method}\) cleared: the singleton strata changed"):
        assert r.design.singleton is None


@pytest.mark.parametrize("method", ["certainty", "skip", "scale", "center", "pool"])
def test_every_rule_is_kept_by_an_unrelated_filter(method):
    h = getattr(sample().singleton, method)()
    r = h.wrangling.filter_records(svy.col("psu") != "a2")
    assert r.design.singleton == h.design.singleton
    fresh = getattr(svy.Sample(r.data, svy.Design(**DESIGNS["single"])).singleton, method)()
    assert ses(r) == pytest.approx(ses(fresh), rel=1e-12)


# set_data / update_data / clone: checked at the call


@pytest.mark.parametrize("setter", ["set_data", "update_data"])
def test_set_data_with_other_singletons_clears_at_the_call(setter):
    h = handled()
    with pytest.warns(UserWarning, match=r"singletons now: 'e'\.") as rec:
        getattr(h, setter)(_split_d(h.data))
    assert len(rec) == 1 and rec[0].filename == __file__
    assert h.design.singleton is None


@pytest.mark.parametrize("setter", ["set_data", "update_data"])
def test_set_data_with_the_same_singletons_keeps_the_rule(setter):
    h = handled()
    getattr(h, setter)(h.data.with_columns((pl.col("y") + 1).alias("y")))
    assert h.design.singleton == SPEC
    assert se(h) == pytest.approx(BASELINE_SES["single/collapse_dict"][0], rel=1e-12)


def test_clone_with_other_singletons_clears_at_the_call():
    h = handled()
    with pytest.warns(UserWarning, match="cleared") as rec:
        c = h.clone(data=_split_d(h.data))
    assert len(rec) == 1 and rec[0].filename == __file__
    assert c.design.singleton is None
    assert h.design.singleton == SPEC


# Design edits


@pytest.mark.parametrize(
    "edit, now",
    [
        ({"stratum": "reg"}, "none"),
        ({"stratum": ("reg", "st")}, "('N', 'd'), ('S', 'e')"),
        ({"psu": "hh"}, "none"),
        ({"psu": None}, "none"),
        ({"ssu": "hh"}, "'d', 'e'"),
    ],
)
def test_a_design_edit_clears_with_the_singletons_of_the_new_design(edit, now):
    h = handled()
    with pytest.warns(UserWarning) as rec:
        h.update_design(**edit)
    assert len(rec) == 1
    msg = str(rec[0].message)
    assert "(collapse) cleared: the stratum/psu/ssu columns changed" in msg
    assert f"singletons now: {now}." in msg
    assert rec[0].filename == __file__
    assert h.design.singleton is None
    # A design change: in the history like any update_design.
    assert h.design_history[-2].singleton == SPEC


def test_a_tuple_and_back_does_not_restore_the_rule():
    h = handled()
    with pytest.warns(UserWarning):
        h.update_design(stratum=("st", "reg"))
    h.update_design(stratum="st")
    assert h.design.singleton is None


def test_other_design_edits_keep_the_rule():
    h = handled()
    h.update_design(prob="w", mos="y")
    assert h.design.singleton == SPEC
    h.update_design(wgt="y")
    assert h.design.singleton == SPEC


def test_forced_removal_of_a_strata_column_drops_the_rule_in_the_one_warning():
    h = sample("ssu").singleton.certainty()
    with pytest.warns(UserWarning) as rec:
        r = h.wrangling.remove_columns("hh", force=True)
    assert len(rec) == 1
    assert "ssu='hh'" in str(rec[0].message)
    assert "singleton handling (certainty)" in str(rec[0].message)
    assert r.design.singleton is None
    assert h.design.singleton == SingletonSpec.certainty(["d", "e"])


# Renames carry the rule: values are stored, not column names


@pytest.mark.parametrize("inplace", [False, True])
def test_renaming_the_strata_and_psu_carries_the_rule(inplace):
    h = handled()
    r = h.wrangling.rename_columns({"st": "stratum2", "psu": "cluster"}, inplace=inplace)
    assert r.design.stratum == "stratum2" and r.design.psu == "cluster"
    assert r.design.singleton == SPEC
    assert r.design_history[-1].singleton == SPEC
    assert ses(r) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)


def test_clean_names_carries_the_rule():
    h = svy.Sample(
        DATA.rename({"st": "Strata Col"}),
        svy.Design(stratum="Strata Col", psu="psu", wgt="w"),
    ).singleton.collapse(using={"d": "a", "e": "c"})
    r = h.wrangling.clean_names()
    assert r.design.stratum == "strata_col"
    assert r.design.singleton == SPEC
    assert ses(r) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)


def test_rename_rep_wgts_has_no_effect_on_the_rule():
    h = handled().weighting.create_bs_wgts(n_reps=4, rstate=1)
    prefix = h.design.rep_wgts.prefix
    r = h.wrangling.rename_rep_wgts({prefix: "boot"})
    assert r.design.rep_wgts.prefix == "boot"
    assert r.design.singleton == SPEC
    assert ses(r) == pytest.approx(ses(h), rel=1e-12)


# ---------------------------------------------------------------------------
# Forks, inplace, caching
# ---------------------------------------------------------------------------


def test_a_fork_clears_while_its_parent_keeps():
    h = handled()
    f = h.wrangling.filter_records(svy.col("st") != "e")
    with pytest.warns(UserWarning, match="cleared"):
        f.design
    assert h.design.singleton == SPEC
    assert ses(h) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)
    with pytest.raises(SingletonError):
        f.estimation.mean("y")
    # And the other way round: the parent changes inplace, the fork is intact.
    g = h.wrangling.filter_records(svy.col("id") != 1)
    h.wrangling.filter_records(svy.col("st") != "d", inplace=True)
    with pytest.warns(UserWarning, match="cleared"):
        h.design
    assert g.design.singleton == SPEC
    assert ses(g) == pytest.approx(
        ses(
            svy.Sample(g.data, svy.Design(**DESIGNS["single"])).singleton.collapse(
                using={"d": "a", "e": "c"}
            )
        ),
        rel=1e-12,
    )


def test_forks_do_not_share_derived_state():
    h = handled()
    h.estimation.mean("y")
    f = h.wrangling.filter_records(svy.col("psu") != "a2")
    assert f._data.height != h._data.height
    f.estimation.mean("y")
    assert h._data.height == DATA.height
    assert h.singleton.last_result.n_psus_before == 11
    assert f.singleton.last_result.n_psus_before == 10


def test_repeated_estimates_do_not_rewarn():
    r = handled().wrangling.filter_records(svy.col("st") != "e")
    with pytest.warns(UserWarning):
        r.design
    for _ in range(3):
        with pytest.raises(SingletonError):
            r.estimation.mean("y")
        r.design
        r.design_history


def test_the_validation_is_cached_per_version():
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
    another sample's validated state: the stamp holds the design itself and a
    process-wide data version."""
    small = make_frame({"a": 2, "b": 2, "d": 1})
    base = svy.Sample(small, svy.Design(**DESIGNS["single"])).singleton.collapse(using={"d": "a"})
    ids: set[int] = set()
    for i in range(15):
        f = base.wrangling.filter_records(svy.col("st") != "d")
        ids.add(id(f._design))
        with pytest.warns(UserWarning, match="cleared"):
            assert f.design.singleton is None
        del f
        gc.collect()
        g = svy.Sample(small, base.design)
        assert g.design.singleton == SingletonSpec.collapse({"d": "a"})
        assert g.singleton.last_result.applied == {"d": "a"}
        del g
    assert base.design.singleton == SingletonSpec.collapse({"d": "a"})


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def test_history_entries_carry_their_own_spec():
    s = sample()
    h = s.singleton.collapse(using={"d": "a", "e": "c"}).singleton.skip()
    assert [d.singleton for d in h.design_history] == [
        None,
        SPEC,
        SingletonSpec.skip(["d", "e"]),
    ]


def test_restoring_a_design_from_before_the_handling():
    h = handled()
    h.set_design(h.design_history[0])
    assert h.design.singleton is None
    with pytest.raises(SingletonError):
        h.estimation.mean("y")
    h.set_design(h.design_history[1])
    assert h.design.singleton == SPEC
    assert ses(h) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)


def test_restoring_a_design_whose_spec_no_longer_matches_clears_it():
    h = handled()
    f = h.wrangling.filter_records(svy.col("st") != "e")
    with pytest.warns(UserWarning):
        f.design
    with pytest.warns(UserWarning, match="handled strata no longer in the data: 'e'") as rec:
        f.set_design(h.design)
    assert len(rec) == 1 and rec[0].filename == __file__
    assert f.design.singleton is None


def test_switching_weights_keeps_the_rule():
    h = handled().weighting.poststratify(controls={"u": 60.0, "v": 55.0}, cells="dom")
    assert h.design.singleton == SPEC
    h.update_design(wgt="w")
    assert h.design.singleton == SPEC
    assert h.design.wgt_adjustment is None
    h.update_design(wgt="ps_wgt")
    assert h.design.wgt_adjustment is not None
    assert h.design.singleton == SPEC
    assert ses(h) == pytest.approx(BASELINE_SES["wgt/collapse_dict/ps"], rel=1e-12)


def test_use_weight_keeps_the_rule():
    h = handled().weighting.poststratify(controls={"u": 60.0, "v": 55.0}, cells="dom")
    u = h.use_weight("w")
    assert u.design.singleton == SPEC
    assert ses(u) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)


# ---------------------------------------------------------------------------
# Save/restore against other data
# ---------------------------------------------------------------------------


def test_a_saved_rule_that_no_longer_matches_is_cleared_at_construction():
    design = _saved(handled())
    with pytest.warns(UserWarning, match=r"cleared: .*singletons now: 'e'\.") as rec:
        s = svy.Sample(_split_d(DATA), design)
    assert len(rec) == 1 and rec[0].filename == __file__
    assert s.design.singleton is None


def test_a_saved_rule_on_its_data_is_silent():
    design = _saved(handled())
    s = svy.Sample(DATA, design)
    assert s.design.singleton == SPEC
    assert s.singleton.last_result.method == SingletonHandling.COLLAPSE


def test_design_with_a_spec_on_other_strata_values():
    design = svy.Design(**DESIGNS["single"], singleton=SingletonSpec.skip(["x", "y"]))
    with pytest.warns(UserWarning, match="handled strata no longer in the data: 'x', 'y'"):
        svy.Sample(DATA, design)


# ---------------------------------------------------------------------------
# The report and the derived columns
# ---------------------------------------------------------------------------


def test_last_result_is_derived_from_the_spec():
    h = sample().singleton.scale()
    lr = h.singleton.last_result
    assert lr.method == SingletonHandling.SCALE
    assert lr.config.singleton_fraction == pytest.approx(2 / 5)
    assert lr.n_singletons_detected == 2
    r = h.wrangling.filter_records(svy.col("psu") != "a2")
    assert r.singleton.last_result.config.singleton_fraction == pytest.approx(2 / 5)
    c = sample().singleton.collapse(using={"d": "a", "e": "c"})
    assert c.singleton.last_result.applied == {"d": "a", "e": "c"}
    assert c.singleton.last_result.n_strata_after == 3


def test_derived_columns_follow_the_data():
    h = handled()
    assert set(h._data.columns) >= {_VAR_STRATUM_COL, _VAR_PSU_COL, _VAR_EXCLUDE_COL}
    r = h.wrangling.recode("st", {"a": ["a", "b"]}, replace=True)
    r.design
    assert r._data.filter(pl.col("st") == "d")[_VAR_STRATUM_COL].unique().to_list() == ["a"]
    assert set(r._data[_VAR_STRATUM_COL].unique().to_list()) == {"a", "c"}


def test_derived_columns_go_when_the_rule_goes():
    r = handled().wrangling.filter_records(svy.col("st") != "e")
    with pytest.warns(UserWarning):
        r.design
    assert _VAR_STRATUM_COL not in r._data.columns


def test_certainty_short_circuits_on_a_certainty_sample():
    h = sample().singleton.certainty()
    assert h.singleton.certainty() is h


def test_handling_without_singletons_returns_the_sample():
    s = svy.Sample(make_frame({"a": 3, "b": 2}), svy.Design(**DESIGNS["single"]))
    for rule in RULES:
        assert rules("single")[rule](s) is s
    assert s.design.singleton is None


# ---------------------------------------------------------------------------
# combine: a data recode, no spec
# ---------------------------------------------------------------------------


def test_combine_recodes_the_data_and_stores_no_spec():
    c = sample().singleton.combine({"st": {"d": "a", "e": "c"}})
    assert c.design.singleton is None
    assert c.singleton.last_result.method == SingletonHandling.COMBINE
    assert c.singleton.last_result.config is None
    ref = _plain(DATA.with_columns(pl.col("st").replace({"d": "a", "e": "c"})))
    assert ses(c) == pytest.approx(ses(ref), rel=1e-12)
    assert ses(c) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)
    # The report stays through later wrangling, as before.
    r = c.wrangling.filter_records(svy.col("id") != 1)
    assert r.singleton.last_result.method == SingletonHandling.COMBINE


def test_combine_after_a_rule_leaves_no_stale_spec():
    h = sample().singleton.skip()
    c = h.singleton.combine({"st": {"d": "a", "e": "c"}})
    assert c.design.singleton is None
    assert _VAR_EXCLUDE_COL not in c._data.columns
    assert ses(c) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)


# ---------------------------------------------------------------------------
# Weighting and replicate weights after handling
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


def test_replicates_after_handling_are_built_on_the_design_strata():
    """Pinned: create_*_wgts use the design's strata and PSUs, as before."""
    h = handled()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jk = h.weighting.create_jk_wgts()
        bs = h.weighting.create_bs_wgts(n_reps=20, rstate=7)
    for r, n_reps, rep_se in ((jk, 11, 0.9737759626249513), (bs, 20, 0.8420318353528821)):
        assert r.design.singleton == SPEC
        assert r.design.rep_wgts.n_reps == n_reps
        assert (r.design.rep_wgts.stratum, r.design.rep_wgts.psu) == ("st", "psu")
        assert se(r) == pytest.approx(0.9135734860304381, rel=1e-12)
        m = r.estimation.mean("y", method="replication").estimates[0]
        assert m.se == pytest.approx(rep_se, rel=1e-12)
        assert estimates(svy.Sample(r.data, _saved(r))) == estimates(r)


def test_brr_after_handling_still_refuses_odd_strata():
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
    h = _rule_round_trips(s, lambda s: s.singleton.collapse(using="smallest"))
    assert h.design.singleton == SingletonSpec.collapse(
        {("N", 4, "k"): ("N", 1, "k"), ("S", 5, "k"): ("N", 2, "k")}
    )
    assert ses(h) == pytest.approx(BASELINE_SES["single/collapse_smallest"], rel=1e-12)


def test_integer_strata():
    s = svy.Sample(DATA, svy.Design(stratum="code", psu="psu", wgt="w"))
    h = _rule_round_trips(s, lambda s: s.singleton.collapse(using={"4": "1", "5": "3"}))
    assert h.design.singleton == SingletonSpec.collapse({4: 1, 5: 3})
    assert ses(h) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)


def test_bool_and_int_tuple_strata():
    df = DATA.with_columns((pl.col("code") % 2 == 0).alias("even"))
    s = svy.Sample(df, svy.Design(stratum=("even", "code"), psu="psu", wgt="w"))
    h = _rule_round_trips(s, lambda s: s.singleton.skip())
    # In the order of svy's stratum keys ("false__by__5" < "true__by__4").
    assert h.design.singleton == SingletonSpec.skip([(False, 5), (True, 4)])
    assert ses(h) == pytest.approx(BASELINE_SES["single/skip"], rel=1e-12)


def test_categorical_strata():
    s = svy.Sample(
        DATA.with_columns(pl.col("st").cast(pl.Categorical)), svy.Design(**DESIGNS["single"])
    )
    h = _rule_round_trips(s, lambda s: s.singleton.pool())
    assert h.design.singleton == SingletonSpec.pool(["d", "e"])
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
        assert r.singleton.last_result.applied == h.singleton.last_result.applied
        assert r._data[_VAR_STRATUM_COL].to_list() == h._data[_VAR_STRATUM_COL].to_list()
    return h


def test_strata_with_a_null():
    df = DATA.with_columns(
        pl.when(pl.col("st") == "e").then(None).otherwise(pl.col("st")).alias("st")
    )
    s = svy.Sample(df, svy.Design(**DESIGNS["single"]))
    h = _null_round_trip(s, lambda s: s.singleton.certainty())
    assert h.design.singleton == SingletonSpec.certainty([None, "d"])
    assert h.singleton.last_result.applied == ("__Null__", "d")


def test_tuple_strata_with_a_null():
    df = DATA.with_columns(
        pl.when(pl.col("st") == "e").then(None).otherwise(pl.col("code")).alias("code")
    )
    s = svy.Sample(df, svy.Design(**DESIGNS["tuple"]))
    h = _null_round_trip(
        s,
        lambda s: s.singleton.collapse(
            using={"S__by____Null__": "S__by__3", "N__by__4": "N__by__1"}
        ),
    )
    assert h.design.singleton == SingletonSpec.collapse(
        {("N", 4): ("N", 1), ("S", None): ("S", 3)}
    )


def test_date_strata():
    days = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    df = DATA.with_columns(
        pl.col("st").replace_strict({k: dt.date(2020, 1, v) for k, v in days.items()}).alias("day")
    )
    s = svy.Sample(df, svy.Design(stratum="day", psu="psu", wgt="w"))
    h = _rule_round_trips(
        s,
        lambda s: s.singleton.collapse(
            using={"2020-01-04": "2020-01-01", "2020-01-05": "2020-01-03"}
        ),
    )
    assert h.design.singleton == SingletonSpec.collapse(
        {dt.date(2020, 1, 4): dt.date(2020, 1, 1), dt.date(2020, 1, 5): dt.date(2020, 1, 3)}
    )
    assert ses(h) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)


def test_integer_strata_cast_to_text_keep_the_rule():
    s = svy.Sample(DATA, svy.Design(**DESIGNS["tuple"]))
    h = s.singleton.skip()
    r = h.wrangling.cast("code", pl.Utf8)
    assert r.design.singleton == SingletonSpec.skip([("N", 4), ("S", 5)])
    assert ses(r) == pytest.approx(BASELINE_SES["single/skip"], rel=1e-12)


def test_psu_only_design_has_no_singleton_rule():
    s = svy.Sample(DATA, svy.Design(psu="psu", wgt="w"))
    assert s.singleton.collapse() is s
    assert s.singleton.certainty() is s
    with pytest.raises(ValueError, match="no stratum"):
        s.update_design(singleton=SingletonSpec.skip(["d"]))
    assert s.design.singleton is None


def test_strata_without_psus():
    df = make_frame({"a": 3, "b": 3}).vstack(make_frame({"d": 1}).with_columns(pl.col("id") + 100))
    df = df.filter(~((pl.col("st") == "d") & (pl.col("id") != 101)))
    s = svy.Sample(df, svy.Design(stratum="st", wgt="w"))
    h = _rule_round_trips(s, lambda s: s.singleton.collapse(using={"d": "a"}))
    ref = svy.Sample(
        df.with_columns(pl.col("st").replace({"d": "a"})), svy.Design(stratum="st", wgt="w")
    )
    assert ses(h) == pytest.approx(ses(ref), rel=1e-12)


def test_strata_as_a_domain_variable():
    df = DATA.with_columns(pl.col("st").alias("st_by"))
    h = svy.Sample(df, svy.Design(**DESIGNS["single"])).singleton.collapse(
        using={"d": "a", "e": "c"}
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
    h = _rule_round_trips(s, lambda s: s.singleton.collapse(using={"d": "a", "e": "c"}))
    assert h.design.singleton == SPEC


# ---------------------------------------------------------------------------
# Singleton patterns
# ---------------------------------------------------------------------------


def test_every_stratum_but_one_a_singleton():
    df = make_frame({"a": 3, "b": 1, "c": 1, "d": 1})
    s = svy.Sample(df, svy.Design(**DESIGNS["single"]))
    h = _rule_round_trips(s, lambda s: s.singleton.collapse())
    assert h.design.singleton == SingletonSpec.collapse({"b": "a", "c": "a", "d": "a"})
    ref = _plain(df.with_columns(pl.lit("a").alias("st")))
    assert ses(h) == pytest.approx(ses(ref), rel=1e-12)


def test_all_strata_singletons_collapse_has_no_target():
    s = svy.Sample(make_frame({"d": 1, "e": 1}), svy.Design(**DESIGNS["single"]))
    with pytest.raises(SingletonError) as exc:
        s.singleton.collapse()
    assert exc.value.code == "NO_MERGE_TARGETS"
    assert s.design.singleton is None
    h = _rule_round_trips(s, lambda s: s.singleton.pool())
    assert h.design.singleton == SingletonSpec.pool(["d", "e"])


def test_a_singleton_whose_psu_has_zero_weight():
    df = DATA.with_columns(
        pl.when(pl.col("st") == "d").then(0.0).otherwise(pl.col("w")).alias("w")
    )
    s = svy.Sample(df, svy.Design(**DESIGNS["single"]))
    assert s.singleton.keys() == ["d", "e"]
    h = _rule_round_trips(s, lambda s: s.singleton.collapse(using={"d": "a", "e": "c"}))
    ref = _plain(df.with_columns(pl.col("st").replace({"d": "a", "e": "c"})))
    assert ses(h) == pytest.approx(ses(ref), rel=1e-12)


def test_a_domain_singleton_is_not_a_design_singleton():
    s = svy.Sample(make_frame({"a": 3, "b": 3}), svy.Design(**DESIGNS["single"]))
    assert not s.singleton.exists
    m = s.estimation.mean("y", where=svy.col("psu").is_in(["a0", "b0", "b1"]))
    assert m.estimates[0].se > 0
    assert s.design.singleton is None


def test_several_singletons_and_message_lists_them_all():
    df = make_frame({"a": 3, "b": 3, "c": 3, "d": 1, "e": 1, "f": 1})
    h = svy.Sample(df, svy.Design(**DESIGNS["single"])).singleton.skip()
    assert h.design.singleton == SingletonSpec.skip(["d", "e", "f"])
    r = h.wrangling.filter_records(svy.col("psu").is_in(["a0", "a1"]), negate=True)
    with pytest.warns(UserWarning, match=r"singletons now: 'a', 'd', 'e', 'f'\."):
        r.design


# ---------------------------------------------------------------------------
# Strata named by their own values in the public API
# ---------------------------------------------------------------------------


def _int_sample() -> svy.Sample:
    return svy.Sample(DATA, svy.Design(stratum="code", psu="psu", wgt="w"))


def test_collapse_takes_native_int_strata():
    h = _int_sample().singleton.collapse(using={4: 1, 5: 3})
    assert h.design.singleton == SingletonSpec.collapse({4: 1, 5: 3})
    assert ses(h) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)


def test_collapse_still_takes_key_strings():
    old = _int_sample().singleton.collapse(using={"4": "1", "5": "3"})
    new = _int_sample().singleton.collapse(using={4: 1, 5: 3})
    mixed = _int_sample().singleton.collapse(using={4: "1", "5": 3})
    assert old.design == new.design == mixed.design


def test_collapse_takes_native_bool_and_tuple_strata():
    df = DATA.with_columns((pl.col("code") % 2 == 0).alias("even"))
    s = svy.Sample(df, svy.Design(stratum=("even", "code"), psu="psu", wgt="w"))
    h = s.singleton.collapse(using={(True, 4): (False, 1), (False, 5): (False, 3)})
    # In the order of svy's stratum keys ("false__by__5" < "true__by__4").
    assert h.design.singleton == SingletonSpec.collapse(
        {(False, 5): (False, 3), (True, 4): (False, 1)}
    )
    assert ses(h) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)
    # Lists for tuples, and the key strings, still work.
    k = s.singleton.collapse(using={"true__by__4": "false__by__1", "false__by__5": [False, 3]})
    assert k.design == h.design


def test_collapse_takes_native_date_strata():
    days = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    df = DATA.with_columns(
        pl.col("st").replace_strict({k: dt.date(2020, 1, v) for k, v in days.items()}).alias("day")
    )
    s = svy.Sample(df, svy.Design(stratum="day", psu="psu", wgt="w"))
    h = s.singleton.collapse(
        using={dt.date(2020, 1, 4): dt.date(2020, 1, 1), dt.date(2020, 1, 5): dt.date(2020, 1, 3)}
    )
    assert ses(h) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)


def test_collapse_callable_may_return_a_value_or_a_candidate():
    s = _int_sample()
    by_value = s.singleton.collapse(using=lambda single, cands: 1)
    by_info = s.singleton.collapse(
        using=lambda single, cands: next(c for c in cands if c.stratum_key == "1")
    )
    assert (
        by_value.design.singleton
        == by_info.design.singleton
        == SingletonSpec.collapse({4: 1, 5: 1})
    )


def test_an_unknown_native_stratum_fails_as_before():
    with pytest.raises(ValueError, match="Mapping does not contain key for singleton '5'"):
        _int_sample().singleton.collapse(using={4: 1})
    with pytest.raises(ValueError, match="not found in candidates"):
        _int_sample().singleton.collapse(using={4: 99, 5: 1})


def test_candidates_for_and_compare_take_native_strata():
    s = _int_sample()
    assert s.singleton.candidates_for(4).equals(s.singleton.candidates_for("4"))
    assert s.singleton.compare(4, 1, ["y"]).equals(s.singleton.compare("4", "1", ["y"]))


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
# Re-validation only when the strata can have moved
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
def test_changes_elsewhere_skip_the_revalidation(change, rederive_calls):
    h = handled()
    h.estimation.mean("y")
    rederive_calls.clear()
    r = change(h)
    r.estimation.mean("y")
    assert rederive_calls == []
    assert r.design.singleton == SPEC


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
def test_changes_to_rows_or_strata_revalidate(change, rederive_calls):
    h = handled()
    h.estimation.mean("y")
    rederive_calls.clear()
    r = change(h)
    r.estimation.mean("y")
    assert len(rederive_calls) == 1


def test_a_skipped_revalidation_keeps_the_derived_columns():
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
    assert f.singleton.keys() == ["b"]


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


# ---------------------------------------------------------------------------
# Every rule x every trigger
# ---------------------------------------------------------------------------

METHODS = ["certainty", "skip", "scale", "center", "pool", "collapse"]


def _apply(method: str, s: svy.Sample) -> svy.Sample:
    if method == "collapse":
        return s.singleton.collapse(using={"d": "a", "e": "c"})
    return getattr(s.singleton, method)()


KEEP_TRIGGERS = {
    "join": lambda s: s.wrangling.join(pl.DataFrame({"id": DATA["id"], "k": DATA["y"]}), on="id"),
    "rename_strata": lambda s: s.wrangling.rename_columns({"st": "st9"}),
    "recode_other": lambda s: s.wrangling.recode("dom", {"U": ["u"]}, replace=True),
    "filter_rows": lambda s: s.wrangling.filter_records(svy.col("id") != 2),
    "update_design_wgt": lambda s: s.update_design(wgt="y"),
    "set_data_same": lambda s: s.set_data(s.data),
    "update_data_same": lambda s: s.update_data(s.data),
}

CLEAR_TRIGGERS = {
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


@pytest.mark.parametrize("trigger", list(KEEP_TRIGGERS))
@pytest.mark.parametrize("method", METHODS)
def test_rule_by_trigger_kept(method, trigger):
    h = _apply(method, sample())
    spec = h.design.singleton
    r = KEEP_TRIGGERS[trigger](h)
    assert r.design.singleton == spec
    # The rule applied afresh to the changed sample's data and design.
    fresh = _apply(method, svy.Sample(r.data, r.design.update(singleton=None)))
    if method == "collapse" and trigger == "rename_strata":
        assert r.design.stratum == "st9"
    assert ses(r) == pytest.approx(ses(fresh), rel=1e-12)


@pytest.mark.parametrize("trigger", list(CLEAR_TRIGGERS))
@pytest.mark.parametrize("method", METHODS)
def test_rule_by_trigger_cleared(method, trigger):
    h = _apply(method, sample())
    with pytest.warns(UserWarning, match=rf"singleton handling \({method}\) cleared") as rec:
        r = CLEAR_TRIGGERS[trigger](h)
        r = h if r is None else r
        assert r.design.singleton is None
    assert len(rec) == 1 and rec[0].filename == __file__


# ---------------------------------------------------------------------------
# More edge cases
# ---------------------------------------------------------------------------


def test_a_where_domain_singleton_does_not_touch_the_rule():
    h = handled()
    m = h.estimation.mean("y", where=svy.col("psu").is_in(["a0", "b0", "b1", "d0"]))
    ref = _plain(DATA.with_columns(pl.col("st").replace({"d": "a", "e": "c"})))
    r = ref.estimation.mean("y", where=svy.col("psu").is_in(["a0", "b0", "b1", "d0"]))
    assert m.estimates[0].se == pytest.approx(r.estimates[0].se, rel=1e-12)
    assert h.design.singleton == SPEC


def test_the_strata_column_inside_the_psu():
    s = svy.Sample(DATA, svy.Design(stratum="st", psu=("st", "psu"), wgt="w"))
    h = _rule_round_trips(s, lambda s: s.singleton.collapse(using={"d": "a", "e": "c"}))
    assert h.design.singleton == SPEC
    assert ses(h) == pytest.approx(BASELINE_SES["single/collapse_dict"], rel=1e-12)


@pytest.mark.parametrize("wname", list(WEIGHTINGS))
@pytest.mark.parametrize("method", METHODS)
def test_weighting_after_handling_equals_a_fresh_handled_sample(method, wname):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = WEIGHTINGS[wname](_apply(method, sample()))
    # The weighted data on a fresh sample, the rule applied there.
    fresh = _apply(method, svy.Sample(r.data, r.design.update(singleton=None)))
    assert ses(r) == pytest.approx(ses(fresh), rel=1e-12)


def test_create_bs_wgts_after_each_rule_is_on_the_design_strata():
    for method in METHODS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = _apply(method, sample()).weighting.create_bs_wgts(n_reps=20, rstate=7)
        assert r.design.singleton is not None
        assert r.design.rep_wgts.n_reps == 20
        assert r.estimation.mean("y", method="replication").estimates[0].se == pytest.approx(
            0.8420318353528821, rel=1e-12
        )


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
    for method in ["certainty", "skip", "scale", "center", "pool"]:
        h = _apply(method, s)
        assert h.design.singleton.handled == ("d", "e")
        # scale has no reference stratum left: NaN SEs, as before.
        restored = ses(svy.Sample(h.data, _saved(h)))
        assert restored == pytest.approx(ses(h), rel=0, nan_ok=True)
