# tests/svy/weighting/test_wgt_api_migration.py
"""The 0.27 weighting API break.

Renames are hard breaks -- there are no aliases and old calls fail. What these
pin is that they fail *legibly*: naming the replacement rather than reporting an
unknown keyword, since `by=` appears in essentially every existing script.
"""

import polars as pl
import pytest

from svy import Design, Sample


@pytest.fixture
def sample():
    df = pl.DataFrame({"w": [1.0, 2.0, 3.0, 4.0], "g": ["a", "a", "b", "b"]})
    return Sample(df, Design(wgt="w"))


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("normalize", {"by": "g"}),
        ("poststratify", {"controls": {"a": 1.0, "b": 1.0}, "by": "g"}),
        ("adjust", {"resp_status": "g", "by": "g"}),
    ],
)
def test_by_names_its_replacement(sample, method, kwargs):
    with pytest.raises(Exception, match="`by=` was renamed to `cells=`"):
        getattr(sample.weighting, method)(**kwargs)


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("poststratify", {"factors": {"a": 0.5, "b": 0.5}}),
        ("rake", {"factors": {"g": {"a": 1.0, "b": 1.0}}}),
    ],
)
def test_factors_names_its_replacement(sample, method, kwargs):
    with pytest.raises(Exception, match="`factors=` was renamed to `shares=`"):
        getattr(sample.weighting, method)(**kwargs)


def test_factors_message_flags_the_semantic_change(sample):
    """factors that did not sum to 1 rescaled the total; shares do not."""
    with pytest.raises(Exception, match="normalized internally"):
        sample.weighting.poststratify(factors={"a": 0.5, "b": 0.5})


def test_unknown_keyword_still_raises_plain_type_error(sample):
    """The guard must not swallow ordinary typos."""
    with pytest.raises(TypeError, match="unexpected keyword argument 'bogus'"):
        sample.weighting.normalize(bogus=1)
