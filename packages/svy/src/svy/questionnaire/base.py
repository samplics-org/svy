# src/svy/questionnaire/base.py
from __future__ import annotations

import logging

from enum import Enum
from typing import Any, Mapping, Self, cast

import msgspec

from svy.core.types import Category
from svy.metadata.labels import Label, LabellingCatalog, MissingKind


log = logging.getLogger(__name__)

# =============================
# Question types & conditions
# =============================


class QuestionType(Enum):
    SINGLE = "single"  # single-select categorical
    MULTI = "multi"  # multi-select categorical
    BOOLEAN = "boolean"  # yes/no (can still use a scheme)
    NUMERIC = "numeric"  # integers/floats
    TEXT = "text"  # free text
    DATE = "date"  # ISO string yyyy-mm-dd (keep simple)


class Op(Enum):
    EQ = "=="
    NE = "!="
    IN = "in"
    NOT_IN = "not_in"
    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="
    IS_MISSING = "is_missing"
    NOT_MISSING = "not_missing"


class EnableWhen(msgspec.Struct, frozen=True):
    """Minimal, serializable condition for routing/enablement."""

    question_id: str
    op: Op
    value: Category | list[Category] | None = None


# =============================
# Choice source
# =============================


class Choices(msgspec.Struct, frozen=True):
    """
    Choices can come from:
      - inline mapping (mapping=...)
      - a catalog concept (concept=..., locale=...)
    Provide exactly one of (mapping, concept).
    """

    mapping: dict[Category, str] | None = None
    concept: str | None = None
    locale: str | None = None
    title: str | None = None
    randomize: bool = False
    allow_other: bool = False  # UI hint; 'other' not encoded here

    def resolve_mapping(self, catalog: LabellingCatalog | None) -> dict[Category, str]:
        if self.mapping is not None:
            # Return a shallow copy to preserve immutability expectations
            return dict(self.mapping)
        if self.concept is None:
            return {}
        if catalog is None:
            raise ValueError("Choices reference a concept but no catalog was provided.")
        lbl: Label = catalog.to_label_by_concept(
            var_label=self.title or self.concept, concept=self.concept, locale=self.locale
        )
        # Be explicit for the type checker: categories may be Optional/Mapping-like
        cats: Mapping[Category, str] | None = getattr(lbl, "categories", None)
        return dict(cats) if cats is not None else {}


# =============================
# Questions & Questionnaire
# =============================


class Question(msgspec.Struct, frozen=True):
    """
    A single question. 'name' is the variable name/column id (stable).
    """

    name: str
    text: str
    qtype: QuestionType
    choices: Choices | None = None  # for categorical types
    required: bool = False

    # Numeric/text constraints (optional)
    min_value: float | None = None
    max_value: float | None = None
    min_length: int | None = None
    max_length: int | None = None

    # Skip logic (all conditions must be satisfied for the question to be enabled).
    enable_when_all: list[EnableWhen] | None = None

    # Which missing kinds are permitted to be recorded for this question (optional)
    allow_missing_kinds: set[MissingKind] | None = None

    def validate(self) -> None:
        # type-consistency checks
        if self.qtype in (QuestionType.SINGLE, QuestionType.MULTI, QuestionType.BOOLEAN):
            # choices optional for BOOLEAN; recommended for SINGLE/MULTI
            if self.qtype != QuestionType.BOOLEAN and self.choices is None:
                raise ValueError(f"Question {self.name!r}: categorical type requires choices.")
        else:
            if self.choices is not None:
                raise ValueError(
                    f"Question {self.name!r}: choices not allowed for {self.qtype.value}."
                )

        if self.qtype == QuestionType.NUMERIC:
            if self.min_value is not None and self.max_value is not None:
                if self.min_value > self.max_value:
                    raise ValueError(f"Question {self.name!r}: min_value > max_value.")
        if self.qtype == QuestionType.TEXT:
            if self.min_length is not None and self.max_length is not None:
                if self.min_length > self.max_length:
                    raise ValueError(f"Question {self.name!r}: min_length > max_length.")

    def is_enabled(self, answers: Mapping[str, Any]) -> bool:
        """Check skip logic against already-collected answers."""
        conds = self.enable_when_all or []
        for c in conds:
            val = answers.get(c.question_id, None)

            if c.op == Op.IS_MISSING:
                if val is not None:
                    return False
                continue
            if c.op == Op.NOT_MISSING:
                if val is None:
                    return False
                continue

            if c.op in (Op.EQ, Op.NE, Op.GT, Op.GE, Op.LT, Op.LE):
                # numeric/text equality/ordering; simple compare
                if c.op == Op.EQ and not (val == c.value):
                    return False
                if c.op == Op.NE and not (val != c.value):
                    return False
                if c.op == Op.GT and not (
                    val is not None and c.value is not None and val > c.value
                ):
                    return False
                if c.op == Op.GE and not (
                    val is not None and c.value is not None and val >= c.value
                ):
                    return False
                if c.op == Op.LT and not (
                    val is not None and c.value is not None and val < c.value
                ):
                    return False
                if c.op == Op.LE and not (
                    val is not None and c.value is not None and val <= c.value
                ):
                    return False
                continue

            if c.op == Op.IN:
                if not isinstance(c.value, list) or val not in c.value:
                    return False
                continue
            if c.op == Op.NOT_IN:
                if isinstance(c.value, list) and val in c.value:
                    return False
                continue

            raise ValueError(f"Unsupported op {c.op} in condition for {self.name!r}.")
        return True

    def resolved_choices(self, catalog: LabellingCatalog | None) -> dict[Category, str]:
        # The empty dict literal needs a cast for precise key type (Category)
        return (
            self.choices.resolve_mapping(catalog)
            if self.choices
            else cast(dict[Category, str], {})
        )

    def clone(self, **overrides) -> Self:
        from msgspec.structs import replace

        return replace(self, **overrides)


class Questionnaire(msgspec.Struct):
    """
    Minimal questionnaire container.
    - Store metadata
    - Hold questions (ordered)
    - Provide resolution/validation helpers
    """

    title: str
    locale: str | None = None
    version: str | None = None
    questions: list[Question] = msgspec.field(default_factory=list)

    # ---------- management ----------
    def add(self, *qs: Question) -> Self:
        self.questions.extend(qs)
        return self

    def add_question(self, **kwargs: Any) -> Self:
        """
        Build-and-append from kwargs:
        Questionnaire().add_question(
            name="consent",
            text="Do you agree to participate?",
            qtype=QuestionType.SINGLE,
            choices=Choices(concept="yes_no"),
            required=True,
        )
        """
        q = Question(**kwargs)
        q.validate()
        self.questions.append(q)
        return self

    def add_questions(self, *items: dict[str, Any]) -> Self:
        """Batch add from kwargs dicts."""
        for kw in items:
            self.add_question(**kw)
        return self

    def remove(self, name: str) -> Self:
        """Remove a question by name."""
        original_len = len(self.questions)
        self.questions = [q for q in self.questions if q.name != name]
        if len(self.questions) == original_len:
            raise KeyError(f"Question not found: {name!r}")
        return self

    def update(self, name: str, **kwargs: Any) -> Self:
        """Update a question's fields by name."""
        if "name" in kwargs:
            raise ValueError("Cannot change question name via update().")
        for i, q in enumerate(self.questions):
            if q.name == name:
                self.questions[i] = q.clone(**kwargs)
                return self
        raise KeyError(f"Question not found: {name!r}")

    def find(self, name: str) -> Question:
        for q in self.questions:
            if q.name == name:
                return q
        raise KeyError(name)

    # ---------- validation ----------
    def validate(self) -> None:
        seen = set()
        for q in self.questions:
            if q.name in seen:
                raise ValueError(f"Duplicate question name: {q.name!r}")
            seen.add(q.name)
            q.validate()

    # ---------- rendering helpers ----------
    def choice_map(self, name: str, catalog: LabellingCatalog | None) -> dict[Category, str]:
        return self.find(name).resolved_choices(catalog)

    def to_codebook(self, catalog: LabellingCatalog | None) -> list[dict[str, Any]]:
        """
        Produce a simple codebook for UI/exports.
        """
        out: list[dict[str, Any]] = []
        for q in self.questions:
            row: dict[str, Any] = {
                "name": q.name,
                "text": q.text,
                "type": q.qtype.value,
            }
            cm = q.resolved_choices(catalog)
            if cm:
                row["choices"] = cm
            out.append(row)
        return out

    # ---------- simple answering ----------
    def apply_skip_logic(self, answers: Mapping[str, Any]) -> set[str]:
        """
        Return the set of question names that are enabled given current answers.
        """
        enabled: set[str] = set()
        for q in self.questions:
            if q.is_enabled(answers):
                enabled.add(q.name)
        return enabled

    def validate_answer(
        self, name: str, value: Any, catalog: LabellingCatalog | None = None
    ) -> None:
        q = self.find(name)
        if value is None:
            if q.required:
                raise ValueError(f"{name} is required.")
            return

        if q.qtype == QuestionType.SINGLE or q.qtype == QuestionType.BOOLEAN:
            cm = q.resolved_choices(catalog)
            if value not in cm:
                raise ValueError(f"{name}: invalid code {value!r}. Allowed: {list(cm.keys())}")
        elif q.qtype == QuestionType.MULTI:
            cm = q.resolved_choices(catalog)
            if not isinstance(value, list):
                raise ValueError(f"{name}: expected list for MULTI.")
            bad = [v for v in value if v not in cm]
            if bad:
                raise ValueError(f"{name}: invalid codes {bad!r}. Allowed: {list(cm.keys())}")
        elif q.qtype == QuestionType.NUMERIC:
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name}: expected numeric.")
            if q.min_value is not None and value < q.min_value:
                raise ValueError(f"{name}: value < min_value.")
            if q.max_value is not None and value > q.max_value:
                raise ValueError(f"{name}: value > max_value.")
        elif q.qtype == QuestionType.TEXT:
            if not isinstance(value, str):
                raise ValueError(f"{name}: expected text.")
            n = len(value)
            if q.min_length is not None and n < q.min_length:
                raise ValueError(f"{name}: text shorter than min_length.")
            if q.max_length is not None and n > q.max_length:
                raise ValueError(f"{name}: text longer than max_length.")
        elif q.qtype == QuestionType.DATE:
            if not isinstance(value, str):
                raise ValueError(f"{name}: expected ISO date string.")
        else:
            raise ValueError(f"Unsupported question type: {q.qtype}.")

    def to_row(self, answers: Mapping[str, Any]) -> dict[str, Any]:
        """
        Convert validated answers to a row (dict). For MULTI, store as a list (up to caller to expand).
        This function assumes `validate_answer` has been called per item.
        """
        row: dict[str, Any] = {}
        for q in self.questions:
            row[q.name] = answers.get(q.name, None)
        return row
