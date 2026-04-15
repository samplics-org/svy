# tests/svy/core/conftest.py
from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import pytest


@dataclass
class CallRecord:
    fn: str
    args: tuple
    kwargs: dict


class DummySvyIO:
    """
    In-memory stand-in for svy-io.
    Updated to accept generic **kwargs to match the flexible engine architecture.
    """

    def __init__(self) -> None:
        self.calls: list[CallRecord] = []

        self._base_df = pl.DataFrame({"id": [1, 2, 3], "sex": [1, 2, 9], "wgt": [0.5, 1.5, 1.0]})
        self._meta = {
            "variables": {
                "id": {"label": "Record id"},
                "sex": {
                    "label": "Sex of respondent",
                    "values": {1: "Male", 2: "Female"},
                    "missing": [9],
                },
                "wgt": {"label": "Final weight"},
            }
        }
        self._file_info = {"source": "dummy"}

    # -------- Reads --------
    def _read_template(self, fnname, **kwargs):
        # We simply return the full dataset and metadata.
        # The Engine (svy.engine.io) is now responsible for column selection
        # and filtering, so the Mock doesn't need to do it.
        res = {
            "data": self._base_df,
            "metadata": self._meta,
            "file_info": self._file_info,
        }

        # Record the exact arguments passed by the engine
        self.calls.append(
            CallRecord(
                fn=fnname,
                args=(),
                kwargs=kwargs,
            )
        )
        return res

    def read_spss(self, path: str, **kwargs):
        return self._read_template("read_spss", **kwargs)

    def read_stata(self, path: str, **kwargs):
        return self._read_template("read_stata", **kwargs)

    def read_sas(self, path: str, **kwargs):
        return self._read_template("read_sas", **kwargs)

    # -------- Writes --------
    def write_spss(self, table_like, path: str, **kwargs):
        # kwargs captures 'metadata', 'encoding', etc.
        self.calls.append(
            CallRecord(
                fn="write_spss",
                args=(path,),
                kwargs={"nrows": len(table_like), **kwargs},
            )
        )

    def write_stata(self, table_like, path: str, **kwargs):
        # kwargs captures 'var_labels', 'value_labels', 'version', etc.
        self.calls.append(
            CallRecord(
                fn="write_stata",
                args=(path,),
                kwargs={"nrows": len(table_like), **kwargs},
            )
        )

    def write_sas(self, table_like, path: str, **kwargs):
        # kwargs captures 'metadata', 'format', etc.
        self.calls.append(
            CallRecord(
                fn="write_sas",
                args=(path,),
                kwargs={"nrows": len(table_like), **kwargs},
            )
        )


@pytest.fixture()
def dummy_svyio(monkeypatch) -> DummySvyIO:
    from svy.engine.io import core as io_core
    from svy.engine.io import sas as io_sas
    from svy.engine.io import spss as io_spss
    from svy.engine.io import stata as io_stata

    dummy = DummySvyIO()
    # Mock all engines to point to this dummy instance
    monkeypatch.setattr(io_core, "sio", dummy, raising=True)
    monkeypatch.setattr(io_spss, "sio", dummy, raising=True)
    monkeypatch.setattr(io_stata, "sio", dummy, raising=True)
    monkeypatch.setattr(io_sas, "sio", dummy, raising=True)
    return dummy
