from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class VarMeta:
    name: str
    label: Optional[str]
    label_set: Optional[str]
    fmt: Optional[str]
    kind: str


@dataclass
class ValueLabels:
    set_name: str
    mapping: Dict[str, str]


@dataclass
class MissingRule:
    var: str
    discrete: List[str]
    ranges: List[Tuple[str, str]]


@dataclass
class SvyMetadata:
    file_label: Optional[str]
    vars: List[VarMeta]
    value_labels: Dict[str, ValueLabels]
    user_missing: Dict[str, MissingRule]
    n_rows: int
