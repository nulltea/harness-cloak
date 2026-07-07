"""Runtime substitution type registry.

Fine detector leaves are externally visible substitution types. `DEM` is kept here only
as a legacy coarse-artifact type; fine-mode runtime code should preserve the leaf label.
"""
import re

FINE_DEM_TYPES = (
    "nationality",
    "ethnicity",
    "religion",
    "profession",
    "age",
    "gender",
    "marital-status",
    "health-condition",
    "sexual-orientation",
    "family-role",
    "demographic-other",
)

DIRECT_TYPES = ("PERSON", "CODE")
COARSE_RUNTIME_TYPES = ("ORG", "LOC", "DATETIME", "QUANTITY", "MISC")
DOMAIN_RUNTIME_TYPES = ("drug", "medical-procedure", "organization-medical-facility")
RUNTIME_TYPES = DIRECT_TYPES + COARSE_RUNTIME_TYPES + FINE_DEM_TYPES + DOMAIN_RUNTIME_TYPES
LEGACY_ROLLUP_TYPES = ("DEM",)
PLACEHOLDER_ONLY_TYPES = ("gender", "marital-status", "sexual-orientation")
FORCED_PLACEHOLDER_TYPES = DIRECT_TYPES + PLACEHOLDER_ONLY_TYPES

PLACEHOLDER_RE = re.compile(r"<[A-Z][A-Z0-9_]*_\d+>")


def placeholder_type_token(type_name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(type_name).upper()).strip("_")
    return token or "OTHER"


def placeholder_token(type_name: str, index: int) -> str:
    return f"<{placeholder_type_token(type_name)}_{index}>"
