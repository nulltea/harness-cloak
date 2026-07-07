from lattice_sources.common import ProfileRow, norm


def alias_rows(runtime_type: str, rows: dict[str, list[str]]) -> list[ProfileRow]:
    return [
        ProfileRow(
            runtime_type=runtime_type,
            surface=norm(surface),
            aliases=[norm(a) for a in aliases],
            levels=[],
            source_ids=[f"manual:{runtime_type}:{norm(surface)}"],
            count=1.0,
        )
        for surface, aliases in rows.items()
    ]
