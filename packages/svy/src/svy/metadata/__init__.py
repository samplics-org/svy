# src/svy/metadata/__init__.py
from svy.core.enumerations import MetadataSource
from svy.metadata.labels import (
    CategoryScheme,
    Label,
    LabellingCatalog,
)
from svy.metadata.variable_meta import (
    # Registry
    MetadataStore,
    # Core metadata types
    ResolvedLabels,
    SchemeRef,
    VariableMeta,
)


__all__ = [
    "ResolvedLabels",
    "SchemeRef",
    "VariableMeta",
    "MetadataStore",
    "MetadataSource",
    "Label",
    "LabellingCatalog",
    "CategoryScheme",
]
