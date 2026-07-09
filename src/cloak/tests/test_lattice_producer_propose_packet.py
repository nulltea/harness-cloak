from cloak.lattice_producer.propose import assemble_context_packet


def test_packet_defines_count_as_anonymity_set_and_requires_two_levels(tmp_path):
    profiles = tmp_path / "p.json"
    profiles.write_text('{"profiles": {}}')
    packet = assemble_context_packet(
        {"runtime_type": "health-condition", "surface": "eczema"},
        profiles_path=profiles, run_dir=tmp_path, prompt_version="v", max_context_rows=8,
    )
    text = packet["count_semantics_instruction"].lower()
    assert "distinct" in text and ("not" in text and ("people" in text or "prevalence" in text))
    assert packet["min_levels"] == 2
