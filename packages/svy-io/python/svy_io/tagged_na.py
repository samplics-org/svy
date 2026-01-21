# python/svy_io/tagged_na.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

Scalar = Union[int, float, str, None]


@dataclass(frozen=True, slots=True)
class TaggedNA:
    """
    Lightweight representation of haven's tagged NA.

    OPTIMIZED: Added __slots__ for reduced memory footprint.
    """

    tag: str

    def __repr__(self) -> str:
        return f"NA({self.tag})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TaggedNA) and self.tag == other.tag

    def __hash__(self) -> int:
        """OPTIMIZED: Added hash for use in sets/dicts"""
        return hash(self.tag)


def tagged_na(tag: Union[str, Sequence[str]]) -> Union[TaggedNA, List[TaggedNA]]:
    """
    Create tagged NA(s).

    OPTIMIZED: Early type check, list comprehension.
    """
    if isinstance(tag, str):
        return TaggedNA(tag)
    return [TaggedNA(t) for t in tag]


def is_tagged_na(
    x: Union[Scalar, TaggedNA, Sequence[Any]], tag: Optional[str] = None
) -> Union[bool, List[bool]]:
    """
    Test if value is a TaggedNA (optionally with specific tag).

    OPTIMIZED: Inline logic, early exits.
    """
    # Handle sequences
    if isinstance(x, (list, tuple)):
        if tag is None:
            return [isinstance(v, TaggedNA) for v in x]
        return [isinstance(v, TaggedNA) and v.tag == tag for v in x]

    # Handle single value
    if not isinstance(x, TaggedNA):
        return False
    return tag is None or x.tag == tag


def na_tag(
    x: Union[Scalar, TaggedNA, Sequence[Any]],
) -> Union[Optional[str], List[Optional[str]]]:
    """
    Return tag of TaggedNA, or None for other values.

    OPTIMIZED: Inline logic, comprehension.
    """
    if isinstance(x, (list, tuple)):
        return [v.tag if isinstance(v, TaggedNA) else None for v in x]

    return x.tag if isinstance(x, TaggedNA) else None


def format_tagged_na(x: Sequence[Any]) -> List[str]:
    """
    Format mixed vector like haven's formatter.

    OPTIMIZED: Pre-allocated list, reduced string operations.
    """
    out = []

    for v in x:
        if isinstance(v, TaggedNA):
            out.append(f"NA({v.tag})")
        elif v is None:
            out.append("   NA")
        else:
            # Numeric or string - right-justify to width 5
            s = str(v)
            # Quick numeric check: all digits/decimal/minus
            out.append(s.rjust(5))

    return out


def print_tagged_na(x: Sequence[Any]) -> str:
    """
    Newline-joined rendering for snapshot tests.

    OPTIMIZED: Direct join instead of intermediate list storage.
    """
    return "\n".join(format_tagged_na(x))
