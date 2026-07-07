from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProfileRow:
    runtime_type: str
    surface: str
    aliases: list[str] = field(default_factory=list)
    levels: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    count: float = 1000.0


def norm(text: str) -> str:
    return " ".join(str(text).strip().lower().split())
