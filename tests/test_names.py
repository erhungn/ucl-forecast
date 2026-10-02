"""The name mapping is the only link between sources, so it must be exact."""

import pandas as pd
import pytest

from ucl.data import names


def test_mapping_has_no_ambiguous_alias():
    mapping = names.load_mapping()
    counts = mapping.groupby(["source", "source_name"]).canonical.nunique()
    assert counts.max() == 1


def test_ambiguous_alias_is_rejected(tmp_path):
    path = tmp_path / "team_names.csv"
    pd.DataFrame(
        {
            "canonical": ["Roma", "Lazio"],
            "country": ["ITA", "ITA"],
            "source": ["openfootball", "openfootball"],
            "source_name": ["Roma", "Roma"],
            "provenance": ["", ""],
        }
    ).to_csv(path, index=False)
    with pytest.raises(names.AmbiguousAliasError, match="openfootball:Roma"):
        names.load_mapping(path)


def test_all_thirty_six_teams_are_mapped():
    teams = names.load_teams()
    assert len(teams) == 36
    mapping = names.load_mapping()
    assert set(teams.team) == set(mapping.canonical)


def test_both_openfootball_spellings_resolve_to_one_team():
    aliases = names.canonical_aliases("openfootball")
    assert set(aliases["Bodo/Glimt"]) == {"Bodo/Glimt", "FK Bodø/Glimt"}
    frame = pd.DataFrame({"home": ["Bodo/Glimt", "FK Bodø/Glimt"], "away": ["Lens", "RC Lens"]})
    out = names.add_canonical(frame, "openfootball")
    assert list(out.home_canonical) == ["Bodo/Glimt", "Bodo/Glimt"]
    assert list(out.away_canonical) == ["Lens", "Lens"]


def test_unmapped_names_are_reported_not_dropped():
    frame = pd.DataFrame({"home": ["Arsenal", "Some New Club"], "away": ["Lens", "Lens"]})
    missing = names.unmapped_names(frame, "openfootball")
    assert list(missing.index) == ["Some New Club"]


def test_teams_with_a_football_data_code_resolve_in_the_cache():
    teams = names.load_teams()
    aliases = names.canonical_aliases("football_data")
    coded = teams[teams.football_data_code != ""]
    assert set(coded.team) == set(aliases)
